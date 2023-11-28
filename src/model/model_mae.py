from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from .layers.agent_embedding import AgentEmbeddingLayer
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.transformer_blocks import Block


class ModelMAE(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        encoder_depth=4,
        decoder_depth=2,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        actor_mask_ratio: float = 0.5,
        lane_mask_ratio: float = 0.5,
        history_steps: int = 50,
        future_steps: int = 60,
        loss_weight: List[float] = [1.0, 1.0, 0.35],
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.actor_mask_ratio = actor_mask_ratio
        self.lane_mask_ratio = lane_mask_ratio
        self.loss_weight = loss_weight

        self.hist_embed = AgentEmbeddingLayer(4, 32, drop_path_rate=drop_path)
        self.future_embed = AgentEmbeddingLayer(3, 32, drop_path_rate=drop_path)
        self.lane_embed = LaneEmbeddingLayer(3, embed_dim)

        # self.hist_lstm_embed = nn.LSTM(4, 128, 2, batch_first=True)
        # self.hist_trans_embed = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=128)
        # self.fut_lstm_embed = nn.LSTM(3, 128, 2, batch_first=True)
        # self.fut_trans_embed = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=128)
        #
        # self.lane_lstm_embed = nn.LSTM(3,128,2,batch_first=True)
        # self.lane_trans_embed = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=128)

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

        # alignment_encoder
        self.alignment_encoder = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(encoder_depth)
        )


        # decoder
        self.reg_embed = nn.Linear(embed_dim, embed_dim, bias=True)
        self.reg_pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # decoder
        self.decoder_embed = nn.Linear(embed_dim, embed_dim, bias=True)
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),)

        #
        dpr = [x.item() for x in torch.linspace(0, drop_path, decoder_depth)]
        self.regression_blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(decoder_depth)
        )

        self.decoder_blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(decoder_depth)
        )


        self.decoder_norm = nn.LayerNorm(embed_dim)

        self.actor_type_embed = nn.Parameter(torch.Tensor(4, embed_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        self.lane_mask_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.future_mask_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.history_mask_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        self.future_pred = nn.Linear(embed_dim, future_steps * 2)
        self.history_pred = nn.Linear(embed_dim, history_steps * 2)
        self.lane_pred = nn.Linear(embed_dim, 20 * 2)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)
        nn.init.normal_(self.lane_type_embed, std=0.02)
        nn.init.normal_(self.future_mask_token, std=0.02)
        nn.init.normal_(self.lane_mask_token, std=0.02)
        nn.init.normal_(self.history_mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def agent_bert_random_masking(
        hist_tokens, fut_tokens, mask_ratio, future_padding_mask, num_actors
    ):
        # num_actors: Bacth有历史轨迹信息的agent [B,A_his]
        # pred_masks: Batch有未来轨迹信息的agent [B,A_fut],且A_fut<=A_his
        # mask_ratio: 未来被mask的agent比例(从所有未来有效的agent中)，剩下的所有agent为保留历史的
        # 所有agent要么保留历史信息，要么保留未来信息，而无未来信息的agent显然不能去掉历史信息
        # 从保留agent历史或未来的角度上考虑（而非mask的角度）：所有被记录的agent都至少有历史信息，但可能没有未来信息。所以先对有未来信息的agent随机取mask_ratio保留未来，再对剩下其他的agent做历史信息,因为Batch中有效agent数量及有历史信息agent数量的影响，最终Batch两者相加的结果不太一定
        pred_masks = ~future_padding_mask.all(-1) # 筛选：将未来全为空的agent标志为False,取反得到未来轨迹有效位, [B, A]
        fut_num_tokens = pred_masks.sum(-1)  # [B]未来值存在的agent个数

        len_keeps = (fut_num_tokens * (1 - mask_ratio)).int() # mask未来之后保留下的agent数量,向下取整
        hist_masked_tokens, fut_masked_tokens = [], []
        hist_keep_ids_list, fut_keep_ids_list = [], []
        hist_key_padding_mask, fut_key_padding_mask = [], []

        device = hist_tokens.device
        agent_ids = torch.arange(hist_tokens.shape[1], device=device)
        for i, (fut_num_token, len_keep, future_pred_mask) in enumerate(
            zip(fut_num_tokens, len_keeps, pred_masks)
        ):  # 对batch维度循环
            # 随机挑选对未来mask的agent
            pred_agent_ids = agent_ids[future_pred_mask]  # 从每个样本中标记哪些agent有效
            noise = torch.rand(fut_num_token, device=device)
            ids_shuffle = torch.argsort(noise)
            fut_ids_keep = ids_shuffle[:len_keep]  # 由随机数排序得到随机保留的agent
            # 取消随机打乱
            sorted_fut_ids_keep, _ = torch.sort(fut_ids_keep)
            fut_ids_keep = pred_agent_ids[sorted_fut_ids_keep]  # 未被mask未来的agent id
            fut_keep_ids_list.append(fut_ids_keep)

            # 随机挑选对历史mask的agent
            hist_keep_mask = torch.zeros_like(agent_ids).bool()
            hist_keep_mask[: num_actors[i]] = True
            hist_keep_mask[fut_ids_keep] = False  # 对已经mask未来的车辆设为false
            hist_ids_keep = agent_ids[hist_keep_mask]
            hist_keep_ids_list.append(hist_ids_keep)

            #创建新的token，将mask——token置零
            hist_masked_token = torch.zeros([hist_tokens.shape[1],hist_tokens.shape[-1]],device=device)
            hist_masked_token[hist_ids_keep,:] = hist_tokens[i, hist_ids_keep,:]
            fut_masked_token = torch.zeros([fut_tokens.shape[1],fut_tokens.shape[-1]],device=device)
            fut_masked_token[fut_ids_keep,:] = fut_tokens[i, fut_ids_keep,:]

            fut_masked_tokens.append(fut_masked_token)
            hist_masked_tokens.append(hist_masked_token)

            fut_key_padding_mask.append(torch.zeros(fut_tokens.shape[1], device=device))
            hist_key_padding_mask.append(torch.zeros(hist_tokens.shape[1], device=device))

        fut_masked_tokens = pad_sequence(fut_masked_tokens, batch_first=True)
        hist_masked_tokens = pad_sequence(hist_masked_tokens, batch_first=True)
        fut_key_padding_mask = pad_sequence(
            fut_key_padding_mask, batch_first=True, padding_value=True
        )
        hist_key_padding_mask = pad_sequence(
            hist_key_padding_mask, batch_first=True, padding_value=True
        )

        return (
            hist_masked_tokens,
            hist_keep_ids_list,
            hist_key_padding_mask,
            fut_masked_tokens,
            fut_keep_ids_list,
            fut_key_padding_mask,
        )


    @staticmethod
    def agent_random_masking(
        hist_tokens, fut_tokens, mask_ratio, future_padding_mask, num_actors
    ):
        # num_actors: Bacth有历史轨迹信息的agent [B,A_his]
        # pred_masks: Batch有未来轨迹信息的agent [B,A_fut],且A_fut<=A_his
        # mask_ratio: 未来被mask的agent比例(从所有未来有效的agent中)，剩下的所有agent为保留历史的
        # 所有agent要么保留历史信息，要么保留未来信息，而无未来信息的agent显然不能去掉历史信息
        # 从保留agent历史或未来的角度上考虑（而非mask的角度）：所有被记录的agent都至少有历史信息，但可能没有未来信息。所以先对有未来信息的agent随机取mask_ratio保留未来，再对剩下其他的agent做历史信息,因为Batch中有效agent数量及有历史信息agent数量的影响，最终Batch两者相加的结果不太一定
        pred_masks = ~future_padding_mask.all(-1) # 筛选：将未来全为空的agent标志为False,取反得到未来轨迹有效位, [B, A]
        fut_num_tokens = pred_masks.sum(-1)  # [B]mask之前的agent个数

        len_keeps = (fut_num_tokens * (1 - mask_ratio)).int() #mask之后保留下的agent数量,向下取整
        hist_masked_tokens, fut_masked_tokens = [], []
        hist_bemasked_tokens, fut_bemasked_tokens = [], []
        hist_keep_ids_list, fut_keep_ids_list = [], []
        hist_masked_ids_list, fut_masked_ids_list = [], []
        hist_key_padding_mask, fut_key_padding_mask = [], []
        fut_bemasked_key_padding_mask, hist_bemasked_key_padding_mask = [], []

        device = hist_tokens.device
        agent_ids = torch.arange(hist_tokens.shape[1], device=device)
        for i, (fut_num_token, len_keep, future_pred_mask) in enumerate(
            zip(fut_num_tokens, len_keeps, pred_masks)
        ):  # 对batch维度循环
            # 随机挑选对未来mask的agent
            pred_agent_ids = agent_ids[future_pred_mask] #从每个样本中标记哪些agent有效
            noise = torch.rand(fut_num_token, device=device)
            ids_shuffle = torch.argsort(noise)
            fut_ids_keep = ids_shuffle[:len_keep] # 由随机数排序得到随机保留的agent
            # fut_ids_bemasked = ids_shuffle[len_keep:] #1）将无效的future——tokens剔除在外
            fut_ids_bemasked = torch.tensor(
                np.setdiff1d(np.arange(0, fut_num_token.cpu().detach().numpy()), fut_ids_keep.cpu().detach().numpy()),
                device=device) #2）将无效的future——tokens包含在内
            fut_ids_keep = pred_agent_ids[fut_ids_keep]  # 未被mask未来的agent id
            fut_keep_ids_list.append(fut_ids_keep)
            fut_masked_ids_list.append(pred_agent_ids[fut_ids_bemasked])


            # 随机挑选对历史mask的agent
            hist_keep_mask = torch.zeros_like(agent_ids).bool()
            hist_keep_mask[: num_actors[i]] = True
            hist_keep_mask[fut_ids_keep] = False  # 对已经mask未来的车辆设为false
            hist_ids_keep = agent_ids[hist_keep_mask]
            hist_keep_ids_list.append(hist_ids_keep)
            hist_masked_ids_list.append(agent_ids[fut_ids_keep])
            # hist_ids_remove = torch.tensor(
            #     np.setdiff1d(np.arange(0, agent_ids.shape[0]), hist_ids_keep.cpu().detach().numpy()),
            #     device=device)

            fut_masked_tokens.append(fut_tokens[i, fut_ids_keep])
            hist_masked_tokens.append(hist_tokens[i, hist_ids_keep])
            fut_bemasked_tokens.append(fut_tokens[i, fut_ids_bemasked])
            hist_bemasked_tokens.append(hist_tokens[i, fut_ids_keep])

            fut_key_padding_mask.append(torch.zeros(len_keep, device=device))
            hist_key_padding_mask.append(torch.zeros(len(hist_ids_keep), device=device))
            fut_bemasked_key_padding_mask.append(torch.zeros(fut_bemasked_tokens[i].shape[0], device=device))
            hist_bemasked_key_padding_mask.append(torch.zeros(hist_bemasked_tokens[i].shape[0], device=device))

        fut_masked_tokens = pad_sequence(fut_masked_tokens, batch_first=True)
        hist_masked_tokens = pad_sequence(hist_masked_tokens, batch_first=True)
        fut_bemasked_tokens = pad_sequence(fut_bemasked_tokens, batch_first=True) #将未来所有时间步都无效的future_tokens剔除在外
        hist_bemasked_tokens = pad_sequence(hist_bemasked_tokens, batch_first=True)
        fut_bemasked_key_padding_mask = pad_sequence(
            fut_bemasked_key_padding_mask, batch_first=True, padding_value=True
        )
        hist_bemasked_key_padding_mask = pad_sequence(
            hist_bemasked_key_padding_mask, batch_first=True, padding_value=True
        )
        fut_key_padding_mask = pad_sequence(
            fut_key_padding_mask, batch_first=True, padding_value=True
        )
        hist_key_padding_mask = pad_sequence(
            hist_key_padding_mask, batch_first=True, padding_value=True
        )


        return (
            hist_masked_tokens,
            hist_keep_ids_list,
            hist_key_padding_mask,
            fut_masked_tokens,
            fut_keep_ids_list,
            fut_key_padding_mask,
            hist_bemasked_tokens,
            fut_bemasked_tokens,
            hist_bemasked_key_padding_mask,
            fut_bemasked_key_padding_mask,
            hist_masked_ids_list,
            fut_masked_ids_list
        )


    @staticmethod
    def lane_random_masking(x, future_mask_ratio, key_padding_mask):
        num_tokens = (~key_padding_mask).sum(1)  # (B, )
        len_keeps = torch.ceil(num_tokens * (1 - future_mask_ratio)).int()

        x_masked, new_key_padding_mask, ids_keep_list, new_key_padding_masked = [], [], [], []
        x_bemasked,ids_masked_list = [], []
        for i, (num_token, len_keep) in enumerate(zip(num_tokens, len_keeps)):
            noise = torch.rand(num_token, device=x.device)
            ids_shuffle = torch.argsort(noise)

            ids_keep = ids_shuffle[:len_keep]
            ids_bemasked = ids_shuffle[len_keep:]
            ids_keep_list.append(ids_keep)
            ids_masked_list.append(ids_bemasked)
            x_masked.append(x[i, ids_keep])
            x_bemasked.append(x[i, ids_bemasked])
            new_key_padding_mask.append(torch.zeros(len_keep, device=x.device))
            new_key_padding_masked.append(torch.zeros(ids_bemasked.shape[0], device=x.device))

        x_bemasked = pad_sequence(x_bemasked, batch_first=True)
        x_masked = pad_sequence(x_masked, batch_first=True)
        new_key_padding_masked = pad_sequence(
            new_key_padding_masked, batch_first=True, padding_value=True
        )
        new_key_padding_mask = pad_sequence(
            new_key_padding_mask, batch_first=True, padding_value=True
        )

        return x_masked, new_key_padding_mask, ids_keep_list, x_bemasked, new_key_padding_masked,ids_masked_list

    @staticmethod
    def lane_bert_random_masking(x, future_mask_ratio, key_padding_mask):
        num_tokens = (~key_padding_mask).sum(1)  # (B, )
        len_keeps = torch.ceil(num_tokens * (1 - future_mask_ratio)).int()

        x_masked, new_key_padding_mask, ids_keep_list = [], [], []
        for i, (num_token, len_keep) in enumerate(zip(num_tokens, len_keeps)):
            noise = torch.rand(num_token, device=x.device)
            ids_shuffle = torch.argsort(noise)
            ids_keep = ids_shuffle[:len_keep]
            sorted_ids_keep, _ = torch.sort(ids_keep)
            ids_keep_list.append(sorted_ids_keep)
            masked_lane_token = torch.zeros([x.shape[1], x.shape[-1]], device=x.device)
            masked_lane_token[sorted_ids_keep, :] = x[i, sorted_ids_keep, :]
            x_masked.append(masked_lane_token)
            new_key_padding_mask.append(torch.zeros(x.shape[1], device=x.device))

        x_masked = pad_sequence(x_masked, batch_first=True)
        new_key_padding_mask = pad_sequence(
            new_key_padding_mask, batch_first=True, padding_value=True
        )

        return x_masked, new_key_padding_mask, ids_keep_list

    def alignment_parameter_update(self):
        """parameter update of the alignment_encoder network."""
        for param_encoder, param_alignment_encoder in zip(self.blocks.parameters(),
                                                          self.alignment_encoder.parameters()):
            param_alignment_encoder.data = param_encoder.data  # completely copy

    def forward(self, data):

        hist_padding_mask = data["x_padding_mask"][:, :, :50]
        hist_feat = torch.cat(
            [
                data["x"],
                data["x_velocity_diff"][..., None],
                ~hist_padding_mask[..., None],
            ],
            dim=-1,
        )
        # random_point_wise_mask
        hist_feat, gt_hist_feat, hist_pointwise_mask = get_random_masked_pointwise(hist_feat,0.5)
        #相当于最后一维的feature
        hist_feature_mask = hist_pointwise_mask * (~hist_padding_mask[...,None])
        #point_wise_mask中需要被预测的point
        hist_pointwise_pred_mask = (~hist_pointwise_mask) * (~hist_padding_mask[..., None])
        #vis
        # import matplotlib.pyplot as plt
        # before = gt_hist_feat[0, :, :, -1].cpu().detach().numpy()
        # after = hist_feat[0, :, :, -1].cpu().detach().numpy()
        # mask1 = feature_mask[0, :, :, 0].cpu().detach().numpy()
        # mask2 = pointwise_pred_mask[0, :, :, 0].cpu().detach().numpy()
        # plt.matshow(before)
        # plt.matshow(after)
        # plt.matshow(mask1)
        # plt.matshow(mask2)
        # plt.colorbar()
        # plt.show()
        B, N, L, D = hist_feat.shape
        # [batch,36,50,4]
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat = self.hist_embed(hist_feat.permute(0, 2, 1).contiguous())
        # hist_feat, _ = self.hist_lstm_embed(hist_feat)
        # hist_feat = self.hist_trans_embed(hist_feat)[:,-1,:]
        hist_feat = hist_feat.view(B, N, hist_feat.shape[-1])  # [batch,36,128]

        future_padding_mask = data["x_padding_mask"][:, :, 50:]
        future_feat = torch.cat([data["y"], ~future_padding_mask[..., None]], dim=-1) #[batch, 36, 128]
        # random_point_wise_mask
        future_feat, gt_fut_feat, fut_pointwise_mask = get_random_masked_pointwise(future_feat, 0.5)
        # 相当于最后一维的feature
        fut_feature_mask = fut_pointwise_mask * (~future_padding_mask[..., None])
        # point_wise_mask中需要被预测的point
        fut_pointwise_pred_mask = (~fut_pointwise_mask) * (~future_padding_mask[..., None])
        B, N, L, D = future_feat.shape
        future_feat = future_feat.view(B * N, L, D)
        future_feat = self.future_embed(future_feat.permute(0, 2, 1).contiguous())
        # future_feat, _ = self.fut_lstm_embed(future_feat)
        # future_feat = self.fut_trans_embed(future_feat)[:,-1,:]
        future_feat = future_feat.view(B, N, future_feat.shape[-1])

        lane_padding_mask = data["lane_padding_mask"]
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_feat = torch.cat([lane_normalized, ~lane_padding_mask[..., None]], dim=-1)
        # random_point_wise_mask
        lane_feat, gt_lane_feat, lane_pointwise_mask = get_random_masked_pointwise(lane_feat, 0.5)
        # 相当于最后一维的feature
        lane_feature_mask = lane_pointwise_mask * (~lane_padding_mask[..., None])
        # point_wise_mask中需要被预测的point
        lane_pointwise_pred_mask = (~lane_pointwise_mask) * (~lane_padding_mask[..., None])
        B, M, L, D = lane_feat.shape  # [batch,200,20,3] #todo:200stand for what? lanes num
        lane_feat = self.lane_embed(lane_feat.view(-1, L, D).contiguous())
        # lane_feat, _ = self.lane_lstm_embed(lane_feat.view(B*M,L,D))
        # lane_feat = self.lane_trans_embed(lane_feat)[:,-1,:]
        lane_feat = lane_feat.view(B, M, -1)

        # type_embedding
        actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2].long()]

        hist_feat += actor_type_embed
        lane_feat += self.lane_type_embed
        future_feat += actor_type_embed

        # pos_embedding
        x_centers = torch.cat(
            [data["x_centers"], data["x_centers"], data["lane_centers"]], dim=1
        )
        angles = torch.cat(
            [
                data["x_angles"][..., 49],
                data["x_angles"][..., 49],
                data["lane_angles"],
            ],
            dim=1,
        )
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)

        pos_embed = self.pos_embed(pos_feat)
        hist_feat += pos_embed[:, :N]
        lane_feat += pos_embed[:, -M:]
        future_feat += pos_embed[:, N: N + N]
        if M==178 and N==28:
            print('find ')
        (
            hist_masked_tokens,
            hist_keep_ids_list,
            hist_key_padding_mask,
            fut_masked_tokens,
            fut_keep_ids_list,
            fut_key_padding_mask,
            hist_bemasked_tokens,
            fut_bemasked_tokens,
            hist_bemasked_key_padding_mask,
            fut_bemasked_key_padding_mask,
            hist_masked_ids_list,
            fut_masked_ids_list
        ) = self.agent_random_masking(
            hist_feat,
            future_feat,
            self.actor_mask_ratio,
            future_padding_mask,
            data["num_actors"],
        )  # mask之后保留下来的历史、未来特征

        # [batch,36,128]-->[batch,20,128]，这里的变化比例随valid车辆个数及未来有效valid个数影响。
        lane_mask_ratio = self.lane_mask_ratio
        (
            lane_masked_tokens,
            lane_key_padding_mask,
            lane_ids_keep_list,
            lane_bemasked_tokens,
            lane_key_padding_masked,
            lane_ids_masked_list
        ) = self.lane_random_masking(
            lane_feat, lane_mask_ratio, data["lane_key_padding_mask"]
        )

        # [batch,145,128]-->[batch,73,128]
        x = torch.cat(
            [hist_masked_tokens, fut_masked_tokens, lane_masked_tokens], dim=1
        )
        key_padding_mask = torch.cat(
            [hist_key_padding_mask, fut_key_padding_mask, lane_key_padding_mask],
            dim=1,
        )
        # [batch,110(20+17+73),128]
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)

        # 2）
        with torch.no_grad():
            bemasked_tokens = torch.cat([hist_bemasked_tokens,fut_bemasked_tokens,lane_bemasked_tokens],dim=1)
            bemasked_key_padding_mask = torch.cat([hist_bemasked_key_padding_mask,fut_bemasked_key_padding_mask,lane_key_padding_masked],dim=1)
            for blk in self.blocks:
                bemasked_tokens = blk(bemasked_tokens, key_padding_mask=bemasked_key_padding_mask)
            # latent_target = self.alignment_encoder(bemasked_tokens)
            self.alignment_parameter_update()


        # decoding
        x_decoder = self.reg_embed(x)
        Nh, Nf, Nl = (
            hist_masked_tokens.shape[1],
            fut_masked_tokens.shape[1],
            lane_masked_tokens.shape[1],
        )
        assert x_decoder.shape[1] == Nh + Nf + Nl
        hist_tokens = x_decoder[:, :Nh]
        fut_tokens = x_decoder[:, Nh : Nh + Nf]
        lane_tokens = x_decoder[:, -Nl:]
        #
        decoder_hist_token = self.history_mask_token.repeat(B, N, 1)
        hist_pred_mask = ~data["x_key_padding_mask"]
        for i, idx in enumerate(hist_keep_ids_list):
            decoder_hist_token[i, idx] = hist_tokens[i, : len(idx)]
            hist_pred_mask[i, idx] = False
        #
        decoder_fut_token = self.future_mask_token.repeat(B, N, 1)
        future_pred_mask = ~data["x_key_padding_mask"]
        for i, idx in enumerate(fut_keep_ids_list):
            decoder_fut_token[i, idx] = fut_tokens[i, : len(idx)]
            future_pred_mask[i, idx] = False
        #
        decoder_lane_token = self.lane_mask_token.repeat(B, M, 1)
        lane_pred_mask = ~data["lane_key_padding_mask"]
        for i, idx in enumerate(lane_ids_keep_list):
            decoder_lane_token[i, idx] = lane_tokens[i, : len(idx)]
            lane_pred_mask[i, idx] = False
        #

        x_decoder = torch.cat(
            [decoder_hist_token, decoder_fut_token, decoder_lane_token], dim=1
        )
        x_decoder = x_decoder + self.reg_pos_embed(pos_feat)
        decoder_key_padding_mask = torch.cat(
            [
                data["x_key_padding_mask"],
                future_padding_mask.all(-1),
                data["lane_key_padding_mask"],
            ],
            dim=1,
        )

        for blk in self.regression_blocks:
            x_decoder = blk(x_decoder, key_padding_mask=decoder_key_padding_mask)
        latent_pred = x_decoder
        latent_hist_pred,latent_fut_pred,latent_lane_pred = [],[],[]
        for i, idx in enumerate(hist_masked_ids_list):
            latent_hist_pred.append(latent_pred[:,:N][i,idx])

        for i, idx in enumerate(fut_masked_ids_list):
            latent_fut_pred.append(latent_pred[:,N:2*N][i,idx])

        for i, idx in enumerate(lane_ids_masked_list):
            latent_lane_pred.append(latent_pred[:,2*N:][i,idx])
        latent_hist_pred = pad_sequence(latent_hist_pred, batch_first=True)
        latent_fut_pred = pad_sequence(latent_fut_pred, batch_first=True)
        latent_lane_pred = pad_sequence(latent_lane_pred, batch_first=True)
        latent_pred = torch.cat([latent_hist_pred,latent_fut_pred,latent_lane_pred], dim=1)


        # 3) stage decoder
        Nhm, Nfm, Nlm = latent_hist_pred.shape[1], latent_fut_pred.shape[1], latent_lane_pred.shape[1]
        latent_pred = self.decoder_embed(latent_pred)
        pos_feat_hist, pos_feat_fut, pos_feat_lane = [], [], []
        for i, idx in enumerate(hist_masked_ids_list):
            pos_feat_hist.append(pos_feat[:, :N][i, idx])
        for i, idx in enumerate(fut_masked_ids_list):
            pos_feat_fut.append(pos_feat[:, N:2 * N][i,idx])
        for i, idx in enumerate(lane_ids_masked_list):
            pos_feat_lane.append(pos_feat[:, 2 * N:][i,idx])
        pos_feat_hist = pad_sequence(pos_feat_hist, batch_first=True)
        pos_feat_fut = pad_sequence(pos_feat_fut, batch_first=True)
        pos_feat_lane = pad_sequence(pos_feat_lane, batch_first=True)
        pos_feat = torch.cat([pos_feat_hist,pos_feat_fut,pos_feat_lane], dim=1)
        latent_pred += self.decoder_pos_embed(pos_feat)

        for blk in self.decoder_blocks:
            #仅将masked tokens解码
            latent_pred = blk(latent_pred, key_padding_mask=bemasked_key_padding_mask)

        latent_pred = self.decoder_norm(latent_pred)
        hist_token = latent_pred[:, :Nhm].reshape(-1, self.embed_dim)
        future_token = latent_pred[:, Nhm: Nhm+Nfm].reshape(-1, self.embed_dim)
        lane_token = latent_pred[:, -Nlm:]

        # lane pred loss
        lane_pred = self.lane_pred(lane_token).view(B, Nlm, 20, 2)
        lane_reg_mask = ~lane_padding_mask
        lane_reg_mask[~lane_pred_mask] = False
        #pred point_wise_mask point
        # for i, idx in enumerate(lane_ids_keep_list):
        #     lane_reg_mask[i,idx] = lane_pointwise_pred_mask[i,idx,:,0]
        #对车道lane在仅保留被masked token下的有效值过滤，此时无法使用pointwise mask
        lane_mask_list = []
        lane_gt_list = []
        for i, idx in enumerate(lane_ids_masked_list):
            lane_mask_list.append(lane_reg_mask[i,idx])
            lane_gt_list.append(lane_normalized[i,idx])
        lane_reg_mask = pad_sequence(lane_mask_list,batch_first=True)
        lane_normalized = pad_sequence(lane_gt_list,batch_first=True)

        # lane_reg_mask = lane_reg_mask[]
        lane_pred_loss = F.mse_loss(
            lane_pred[lane_reg_mask], lane_normalized[lane_reg_mask]
        )

        # hist pred loss
        x_hat = self.history_pred(hist_token).view(-1, 50, 2)
        x = (data["x_positions"] - data["x_centers"].unsqueeze(-2)).view(-1, 50, 2)
        x_reg_mask = ~data["x_padding_mask"][:, :, :50]
        x_reg_mask[~hist_pred_mask] = False
        # before_mask = x_reg_mask.clone()
        # for i, idx in enumerate(hist_keep_ids_list):
        #     x_reg_mask[i,idx] = hist_pointwise_pred_mask[i,idx,:,0]
        hist_mask_list = []
        for i, idx in enumerate(hist_masked_ids_list):
            hist_mask_list.append(x_reg_mask[i, idx])
        masked_token_hist_valid = pad_sequence(hist_mask_list, batch_first=True).view(-1, 50)
        #vis
        # import matplotlib.pyplot as plt
        # before = before_mask[0, :, :].cpu().detach().numpy()
        # after = x_reg_mask[0, :, :].cpu().detach().numpy()
        # point = hist_pointwise_pred_mask[0, :, :, 0].cpu().detach().numpy()
        # # plt.matshow(before)
        # plt.matshow(point)
        # # plt.matshow(mask1)
        # # plt.matshow(mask2)
        # plt.colorbar()
        # plt.show()
        x_reg_mask = x_reg_mask.view(-1, 50)
        # print('his_pred_len:',x_hat[x_reg_mask].shape)
        hist_loss = F.l1_loss(x_hat[masked_token_hist_valid], x[x_reg_mask])

        # future pred loss
        y_hat = self.future_pred(future_token).view(-1, 60, 2)  # B*N, 120
        y = data["y"].view(-1, 60, 2)
        reg_mask = ~data["x_padding_mask"][:, :, 50:]
        reg_mask[~future_pred_mask] = False
        # for i, idx in enumerate(fut_keep_ids_list):
        #     reg_mask[i,idx] = fut_pointwise_pred_mask[i,idx,:,0]
        fut_mask_list = []
        for i, idx in enumerate(fut_masked_ids_list):
            fut_mask_list.append(reg_mask[i, idx])
        masked_token_fut_valid = pad_sequence(fut_mask_list,batch_first=True).view(-1, 60)
        reg_mask = reg_mask.view(-1, 60)
        future_loss = F.l1_loss(y_hat[masked_token_fut_valid], y[reg_mask])


        latent_loss = F.mse_loss(latent_pred.float(), bemasked_tokens.detach().float(), reduction="mean")

        loss = (
            self.loss_weight[0] * future_loss
            + self.loss_weight[1] * hist_loss
            + self.loss_weight[2] * lane_pred_loss
            # + latent_loss
        )

        # gt_his = (x.view(B, N, 50, 2) + data["x_centers"].unsqueeze(-2)).cpu().detach().numpy()
        # gt_fut = (y.view(B, N, 60, 2)+ data["x_centers"].unsqueeze(-2)).cpu().detach().numpy()
        # gt_lane = (lane_normalized + data["lane_centers"].unsqueeze(-2)).cpu().detach().numpy()
        # mask_his = x_reg_mask.view(B,N,50)
        # mask_fut = reg_mask.view(B,N,60)
        # mask_lane = lane_reg_mask
        #
        # pred_his = (x_hat.view(B, N, 50, 2) + data["x_centers"].unsqueeze(-2))[0,mask_his[0,...]].cpu().detach().numpy()
        # pred_fut = (y_hat.view(B, N, 60, 2) + data["x_centers"].unsqueeze(-2))[0,mask_fut[0,...]].cpu().detach().numpy()
        # pred_lane = (lane_pred + data["lane_centers"].unsqueeze(-2))[0,mask_lane[0,...]].cpu().detach().numpy()
        #
        # masked_his = (x.view(B, N, 50, 2) + data["x_centers"].unsqueeze(-2))[0,~mask_his[0,...]].cpu().detach().numpy()
        # masked_fut = (y.view(B, N, 60, 2) + data["x_centers"].unsqueeze(-2))[0,~mask_fut[0,...]].cpu().detach().numpy()
        # masked_lane = (lane_normalized + data["lane_centers"].unsqueeze(-2))[0,~mask_lane[0,...]].cpu().detach().numpy()


        # import matplotlib.pyplot as plt
        # plt.scatter(gt_his[0,:,:,0],gt_his[0,:,:,1],c='orange')
        # plt.scatter(gt_fut[0, :, :, 0], gt_fut[0, :, :, 1],c='blue')
        # plt.scatter(gt_lane[0,:,:,0],gt_lane[0,:,:,1],c='gray')
        # plt.scatter(pred_his[:, 0], pred_his[:, 1],c='green')
        # plt.scatter(pred_fut[:, 0], pred_fut[:, 1],c='green')
        # plt.scatter(pred_lane[:, 0], pred_lane[:, 1],c='green')
        # plt.scatter(masked_his[:, 0], masked_his[:, 1], c='orange')
        # plt.scatter(masked_fut[:, 0], masked_fut[:, 1], c='blue')
        # plt.scatter(masked_lane[:, 0], masked_lane[:, 1], c='gray')


        # plt.axis('equal')
        # plt.show()



        # if viz:
        #     # target/ego bounding_box&traj
        #     for i in range(ego.shape[0]):
        #         rect = plt.Rectangle((ego[i,-1, 0]-ego[i,-1, 5]/2, ego[i,-1, 1]-ego[i,-1, 6]/2), ego[i,-1, 5], ego[i,-1, 6], linewidth=2, color='r', alpha=0.6, zorder=3,
        #                             transform=mpl.transforms.Affine2D().rotate_around(*(ego[i,-1, 0], ego[i,-1, 1]), ego[i,-1, 2]) + plt.gca().transData)
        #         plt.gca().add_patch(rect)
        #
        #         future = ground_truth[i][ground_truth[i][:, 0] != 0]
        #         plt.plot(future[:, 0], future[:, 1], 'r', linewidth=1, zorder=3)
        #     # neighbor bounding_box
        #     for i in range(neighbors.shape[0]):
        #         if neighbors[i, -1, 0] != 0:
        #             rect = plt.Rectangle((neighbors[i, -1, 0]-neighbors[i, -1, 5]/2, neighbors[i, -1, 1]-neighbors[i, -1, 6]/2),
        #                                   neighbors[i, -1, 5], neighbors[i, -1, 6], linewidth=1.5, color='m', alpha=0.6, zorder=3,
        #                                   transform=mpl.transforms.Affine2D().rotate_around(*(neighbors[i, -1, 0], neighbors[i, -1, 1]), neighbors[i, -1, 2]) + plt.gca().transData)
        #             plt.gca().add_patch(rect)
        #     # lanes
        #     # map_lanes[2,6,300,17]
        #     # 对2个agent的lane循环
        #     for i in range(map_lanes.shape[0]):
        #         lanes = map_lanes[i]
        #         crosswalks = map_crosswalks[i]
        #
        #         #对6条polyline循环绘图
        #         for j in range(map_lanes.shape[1]):
        #             lane = lanes[j]
        #             if lane[0][0] != 0:
        #                 centerline = lane[:, 0:2]
        #                 centerline = centerline[centerline[:, 0] != 0]
        #                 left = lane[:, 3:5]
        #                 left = left[left[:, 0] != 0]
        #                 right = lane[:, 6:8]
        #                 right = right[right[:, 0] != 0]
        #                 plt.plot(centerline[:, 0], centerline[:, 1], 'k', linewidth=0.5)  # plot centerline
        #                 plt.plot(left[:,0], left[:,1],'y', linewidth=0.5)  # plot left boundary
        #                 plt.plot(right[:, 0], right[:, 1], 'orange', linewidth=0.5)  # plot right boundary
        #         # crosswalks
        #         # map_crosswalks[2,4,100,3]
        #         # 对最近4条crosswalks循环绘图
        #         for k in range(map_crosswalks.shape[1]):
        #             crosswalk = crosswalks[k]
        #             if crosswalk[0][0] != 0:
        #                 crosswalk = crosswalk[crosswalk[:, 0] != 0]
        #                 plt.plot(crosswalk[:, 0], crosswalk[:, 1], 'b', linewidth=1) # plot crosswalk
        #     # todo：不知道在画什么点
        #     if self.point_dir != '':
        #         for i in range(region_dict[32].shape[0]):
        #             plt.scatter(region_dict[32][i,:,0],region_dict[32][i,:,1],marker='*',s=10)
        #
        #     plt.gca().set_aspect('equal')
        #     plt.tight_layout()
        #     plt.show()
        #     print(' ')






        return {
            "loss": loss,
            "hist_loss": hist_loss.item(),
            "future_loss": future_loss.item(),
            "lane_pred_loss": lane_pred_loss.item(),
            "latent_pred_loss": latent_loss.item()
        }

def get_random_masked_pointwise(gt_agents, mask_percentage):
    time_dim = 1
    # ego_in, ego_out, agents_in, agents_out
    mask = torch.rand((gt_agents[:,:, :,-1].shape)).to(gt_agents.device)
    mask = (mask > mask_percentage).unsqueeze(-1) # [B, T, M, 1]

    masked_agent = gt_agents * mask
    # vis
    # import matplotlib.pyplot as plt
    # # before = ego_data[0,:,:].cpu().detach().numpy()
    # # after = ego_masked[0,:,:].cpu().detach().numpy()
    # before = agents_data[0, :, :, 1].cpu().detach().numpy()
    # after = agents_masked[0, :, :, 1].cpu().detach().numpy()
    # # plt.matshow(before)
    # plt.matshow(after)
    # plt.show()
    return masked_agent, gt_agents, mask
