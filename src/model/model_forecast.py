from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.agent_embedding import AgentEmbeddingLayer
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.multimodal_decoder import MultimodalDecoder
from .layers.transformer_blocks import Block
from .layers.transformer_decoder import InitialDecoder, InteractionDecoder, FutureEncoder


class ModelForecast(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        future_steps: int = 60,
    ) -> None:
        super().__init__()
        self.hist_embed = AgentEmbeddingLayer(
            4, embed_dim // 4, drop_path_rate=drop_path
        )
        self.lane_embed = LaneEmbeddingLayer(3, embed_dim)

        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(encoder_depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.actor_type_embed = nn.Parameter(torch.Tensor(4, embed_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.mode = 6
        self.decoder = MultimodalDecoder(embed_dim, future_steps)
        self.dense_predictor = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, future_steps * 2)
        )

        self.first_layer_decoder = InitialDecoder(6, 0, future_steps)
        self.decoder_layer_num = 3
        future_encoder = FutureEncoder()
        self.interaction_stage = nn.ModuleList([InteractionDecoder(future_encoder, future_steps) for _ in range(self.decoder_layer_num)])
        # self.decoder_embed = nn.Linear(embed_dim, embed_dim, bias=True)
        # self.decoder_pos_embed = nn.Sequential(
        #     nn.Linear(4, embed_dim),
        #     nn.GELU(),
        #     nn.Linear(embed_dim, embed_dim),
        # )
        #
        # self.dense_future_encoder = nn.Sequential(nn.Linear(2*60,128), nn.ReLU())
        # self.fusion_future = nn.Sequential(nn.Linear(256,128), nn.ReLU())
        #
        # self.query_encoder = nn.Sequential(nn.Linear(120,128), nn.ReLU())
        # self.multi_modal_query_embedding = nn.Embedding(6,128)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)
        nn.init.normal_(self.lane_type_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("net.") :]: v for k, v in ckpt.items() if k.startswith("net.")
        }
        print('load_ckpy_modules: ',state_dict.keys())
        return self.load_state_dict(state_dict=state_dict, strict=False)

    def forward(self, data):
        hist_padding_mask = data["x_padding_mask"][:, :, :50] #[B,N,50]
        hist_key_padding_mask = data["x_key_padding_mask"] #[B,N]当历史50个轨迹点全为padding时(全部点无效时)才为True
        hist_feat = torch.cat(
            [
                data["x"],
                data["x_velocity_diff"][..., None],
                ~hist_padding_mask[..., None],
            ],
            dim=-1,
        )  # [batch,obj_num,50,4(2+1+1)]

        B, N, L, D = hist_feat.shape
        #[batch,35,50,4]
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat_key_padding = hist_key_padding_mask.view(B * N)
        actor_feat = self.hist_embed(
            hist_feat[~hist_feat_key_padding].permute(0, 2, 1).contiguous()
        )
        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], device=actor_feat.device
        )
        actor_feat_tmp[~hist_feat_key_padding] = actor_feat  # 将历史轨迹点全缺失的actor_feat设置为全零
        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1]) #[batch,obj_num,128]

        lane_padding_mask = data["lane_padding_mask"]
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_normalized = torch.cat(
            [lane_normalized, ~lane_padding_mask[..., None]], dim=-1
        )
        B, M, L, D = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(B, M, -1)


        #vis input
        # input = data['lane_positions']
        # print(input.shape)
        # mask = ~data['x_padding_mask'][:, :, :50, None]
        # print(mask.shape)
        # valid = data['x'] * mask
        # print(valid.shape)
        # vel_diff = data['x_velocity_diff'][0, ...].cpu().detach().numpy()
        # lane_norm = lane_normalized.cpu().detach().numpy()
        # x_position = data['x_positions'].cpu().detach().numpy()
        # input = input.cpu().detach().numpy()
        # valid = valid.cpu().detach().numpy()
        # y = data['y'].cpu().detach().numpy()
        # y_ = (data['y']+data['x_positions'][:,:,-1,:].unsqueeze(-2)).cpu().detach().numpy()
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(4, 2)
        # for i in range(input.shape[1]):
        #     ax[0, 0].plot(input[0, i, :, 0], input[0, i, :, 1])
        #     ax[0, 0].set_title('target-centric坐标下的车道线lane')
        # for i in range(valid.shape[1]):
        #     ax[0, 1].scatter(valid[0, i, :, 0], valid[0, i, :, 1], s=1)
        #     ax[0, 1].set_title('data["x"]历史轨迹相对前一帧位移his_traj_diff')
        # ax[1, 0].matshow(vel_diff)
        # for i in range(lane_norm.shape[1]):
        #     ax[1, 1].plot(lane_norm[0, i, :, 0], lane_norm[0, i, :, 1])
        #     ax[1, 1].set_title('车道线规范化后lane_norm')
        # for i in range(x_position.shape[1]):
        #     ax[2, 0].scatter(x_position[0, i, :, 0], x_position[0, i, :, 1], s=1)
        #     ax[2, 0].set_title('target-centric坐标下历史轨迹his_traj')
        # for i in range(y.shape[1]):
        #     ax[3, 0].scatter(y[0, i, :, 0], y[0, i, :, 1], s=1)
        #     ax[3, 0].set_title('agent-centric坐标下未来轨迹fut_traj')
        # for i in range(y_.shape[1]):
        #     ax[3, 1].scatter(y_[0, i, :, 0], y_[0, i, :, 1], s=1)
        #     ax[3, 1].set_title('target-centric坐标下未来轨迹fut_traj')
        # plt.show()

        x_centers = torch.cat([data["x_centers"], data["lane_centers"]], dim=1)
        angles = torch.cat([data["x_angles"][:, :, 49], data["lane_angles"]], dim=1)
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat)

        actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2].long()]
        lane_type_embed = self.lane_type_embed.repeat(B, M, 1)
        actor_feat += actor_type_embed
        lane_feat += lane_type_embed

        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)
        key_padding_mask = torch.cat(
            [data["x_key_padding_mask"], data["lane_key_padding_mask"]], dim=1
        )

        x_encoder = x_encoder + pos_embed
        #[batch,181,128]
        for blk in self.blocks:
            x_encoder = blk(x_encoder, key_padding_mask=key_padding_mask)
        x_encoder = self.norm(x_encoder)

        #与pretrain一致，在decoder之前加上线性映射与pos_embed
        # x_encoder = self.decoder_embed(x_encoder)
        # decode_pos_embed = self.decoder_pos_embed(pos_feat)
        # x_encoder += decode_pos_embed


        # target_encoding = x_encoder[:, 0, :]
        # obj_encoding = x_encoder[:, :N, :]
        # map_encoding = x_encoder[:, N:, :]
        #
        # target_mask = torch.zeros([B])
        # obj_mask = hist_padding_mask
        # map_mask = lane_padding_mask
        #
        # target_pos = pos_feat[:,0,:]
        # obj_pos = pos_feat[:,:N,:]
        # map_pos = pos_feat[:,N:,:]











        #sur mlp decoder
        x_others = x_encoder[:, :N] #[batch,35,128]
        y_hat_others = self.dense_predictor(x_others).view(B, -1, 60, 2)

        # dense predict 进行agent-centric的预测
        # y_hat_others += data['x_positions'][:, :, -1:, :]

        # 在dense predict之后做future encoder，类似mtr的做法
        # others_reg_mask = ~data["x_padding_mask"][:, :, 50:]
        # valid_y_hat_others = others_reg_mask[:, :, :, None] * y_hat_others
        # valid_future_feat = self.dense_future_encoder(valid_y_hat_others.flatten(2))
        # fusion = torch.cat((valid_future_feat, x_encoder[:, :N]), dim=-1)
        # new_x_encoder = self.fusion_future(fusion)

        # target mlp decoder
        x_agent = x_encoder[:, 0, :]  # [batch,128]
        y_hat, pi = self.decoder(x_agent)




        #mtr_transformer_decoder
        # pred_list = self.apply_transformer_decoder(
        #     center_objects_feature=target_encoding,
        #     center_objects_type=data['x_attr'][:,:,0],
        #     obj_feature=obj_encoding, obj_mask=obj_mask, obj_pos=obj_pos,
        #     map_feature=map_encoding, map_mask=map_mask, map_pos=map_pos
        # )

        # target_feat = x_encoder[0, 0, :].cpu().detach().numpy()
        # import matplotlib.pyplot as plt
        # plt.plot(target_feat)
        # plt.show()
        #
        # obj_feat = x_encoder[0, :N, :].cpu().detach().numpy()
        # import matplotlib.pyplot as plt
        # plt.matshow(obj_feat)
        # plt.colorbar()
        # plt.show()
        #
        # map_feat = x_encoder[0, N:, :].cpu().detach().numpy()
        # import matplotlib.pyplot as plt
        # plt.matshow(map_feat)
        # plt.colorbar()
        # plt.show()

        #first_layer_decoder,将所有的特征作为key与value
        # decoder_outputs = {}
        # pred_traj = []
        # pred_score = []
        # target_current_state = torch.cat((data['x_positions'][:,0,-1,:],data['x_attr'][:,0,0:1]),dim=-1)
        #1）用mlp做第一层decoder
        # last_level, last_scores = self.decoder(x_encoder[:, 0, :])
        # last_content = self.query_encoder(last_level.flatten(2)) + self.multi_modal_query_embedding(torch.tensor([0,1,2,3,4,5]).cuda())[None,:,:].repeat(B,1,1)
        #2)原始Gameformer第一层decoder
        # results = self.first_layer_decoder(0, target_current_state, x_encoder, key_padding_mask)
        # last_content = results[0]
        # last_level = results[1]
        # last_scores = results[2]
        # pred_traj.append(last_level)
        # pred_score.append(last_scores)
        # decoder_outputs['level_0_interactions'] = last_level
        # decoder_outputs['level_0_scores'] = last_scores




        #todo: 完善current_state中所需的参数 Done!
        # self.decoder_layer_num = 3
        # for k in range(1, self.decoder_layer_num+1):
        #     interaction_decoder = self.interaction_stage[k-1]
        #     results = interaction_decoder(0, target_current_state.unsqueeze(1), last_level.unsqueeze(1), last_scores.unsqueeze(1), \
        #                last_content, x_encoder, key_padding_mask)
        #     # last_content: 上一层的[batch,2,6,256]
        #     last_content = results[0]
        #     # last_level: 预测轨迹【batch,2,6,80,4】
        #     last_level = results[1]
        #     # last_score： 预测概率【batch，2,6】
        #     last_scores = results[2]
        #     decoder_outputs[f'level_{k}_interactions'] = last_level
        #     decoder_outputs[f'level_{k}_scores'] = last_scores
        #     pred_traj.append(last_level)
        #     pred_score.append(last_scores)


        # y_hat = last_level[:,:,:,:2]
        # pi = last_scores
        #todo: 加入第一层及各层预测结果的指标记录
        #todo: 对多层decoder计算loss



        #todo: remove other agent loss
        # x_others = previous_encoder[:, 1:N] #[batch,35,128]
        # y_hat_others = self.dense_predictor(x_others).view(B, -1, 60, 2)
        single_layer_result = True
        if single_layer_result:
            return {
                "y_hat": y_hat,
                "pi": pi,
                "y_hat_others": y_hat_others,
            }
        else:
            pass
            # return {
            #     "y_hat": pred_traj,
            #     "pi": pred_score,
            #     "y_hat_others": y_hat_others,
            # }
