import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection

from src.metrics import MR, minADE, minFDE,multilayer_minADE,multilayer_minFDE
from src.utils.optim import WarmupCosLR
from src.utils.submission_av2 import SubmissionAv2

from .model_forecast import ModelForecast


class Trainer(pl.LightningModule):
    def __init__(
        self,
        dim=128,
        historical_steps=50,
        future_steps=60,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        pretrained_weights: str = None,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
    ) -> None:
        super(Trainer, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.submission_handler = SubmissionAv2()
        self.decoder_layers = 4
        self.net = ModelForecast(
            embed_dim=dim,
            encoder_depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path=drop_path,
            future_steps=future_steps,
        )

        if pretrained_weights is not None:
            self.net.load_from_checkpoint(pretrained_weights)
            print('load from checkpoint: ', pretrained_weights)
        else:
            print('NO checkpoint, train from scratch!')

        metrics_dict = {
            "minADE1": minADE(k=1),
            "minADE6": minADE(k=6),
            "minFDE1": minFDE(k=1),
            "minFDE6": minFDE(k=6),
            "MR": MR(),
        }
        # metrics_dict.update({f"minADE1 layer{n}": multilayer_minADE(n, k=1) for n in range(self.decoder_layers)})
        # metrics_dict.update({f"minADE6 layer{n}": multilayer_minADE(n, k=6) for n in range(self.decoder_layers)})
        # metrics_dict.update({f"minFDE1 layer{n}": multilayer_minFDE(n, k=1) for n in range(self.decoder_layers)})
        # metrics_dict.update({f"minFDE6 layer{n}": multilayer_minFDE(n, k=6) for n in range(self.decoder_layers)})
        metrics = MetricCollection(metrics_dict)

        self.val_metrics = metrics.clone(prefix="val_")

    def forward(self, data):
        return self.net(data)

    def predict(self, data):
        with torch.no_grad():
            out = self.net(data)
        predictions, prob = self.submission_handler.format_data(
            data, out["y_hat"], out["pi"], inference=True
        )
        return predictions, prob


    #3) cal target_loss & all(taget&sur)_dense_loss
    # def cal_loss(self, out, data):
    #     y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
    #     y, y_others = data["y"][:, 0], data["y"][:, :]
    #
    #     l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
    #     best_mode = torch.argmin(l2_norm, dim=-1)
    #     y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]
    #
    #     agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
    #     agent_cls_loss = F.cross_entropy(pi, best_mode.detach())
    #
    #     others_reg_mask = ~data["x_padding_mask"][:, :, 50:]
    #     others_reg_loss = F.smooth_l1_loss(
    #         y_hat_others[others_reg_mask], y_others[others_reg_mask]
    #     )
    #
    #     loss = agent_reg_loss + agent_cls_loss + others_reg_loss
    #
    #     return {
    #         "loss": loss,
    #         "reg_loss": agent_reg_loss.item(),
    #         "cls_loss": agent_cls_loss.item(),
    #         "others_reg_loss": others_reg_loss.item(),
    #     }


    #1)cal target_loss & sur_dense_predict_loss
    # def cal_loss(self, out, data):
    #     y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
    #     y, y_others = data["y"][:, 0], data["y"][:, :]
    #
    #     l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
    #     best_mode = torch.argmin(l2_norm, dim=-1)
    #     y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]
    #
    #     agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
    #     agent_cls_loss = F.cross_entropy(pi, best_mode.detach())
    #
    #     others_reg_mask = ~data["x_padding_mask"][:, :, 50:]
    #     others_reg_loss = F.smooth_l1_loss(
    #         y_hat_others[others_reg_mask], y_others[others_reg_mask]
    #     )
    #
    #     loss = agent_reg_loss + agent_cls_loss + others_reg_loss
    #
    #     return {
    #         "loss": loss,
    #         "reg_loss": agent_reg_loss.item(),
    #         "cls_loss": agent_cls_loss.item(),
    #         "others_reg_loss": others_reg_loss.item(),
    #     }

    # 2) cal target_loss
    # def cal_loss(self, out, data):
    #     y_hat, pi = out["y_hat"], out["pi"]
    #     y, y_others = data["y"][:, 0], data["y"][:, 1:]
    #
    #     l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
    #     best_mode = torch.argmin(l2_norm, dim=-1)
    #     y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]
    #
    #     agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
    #     agent_cls_loss = F.cross_entropy(pi, best_mode.detach())
    #
    #     loss = agent_reg_loss + agent_cls_loss
    #
    #     return {
    #         "loss": loss,
    #         "reg_loss": agent_reg_loss.item(),
    #         "cls_loss": agent_cls_loss.item(),
    #     }

    #4) multilayer cal target_loss & sur_dense_predict_loss
    # def cal_loss(self, out, data):
    #     pred_traj_list = out["y_hat"]
    #     pred_score_list = out["pi"]
    #     # y_hat_others = out["y_hat_others"]
    #     y, y_others = data["y"][:, 0], data["y"][:, :]
    #     assert len(pred_traj_list)==len(pred_score_list)==self.decoder_layers
    #     target_reg = torch.tensor([0.0]).cuda()
    #     target_cls = torch.tensor([0.0]).cuda()
    #     layer_weight = []
    #     loss_result = {}
    #     for layer in range(self.decoder_layers):
    #         y_hat, pi = pred_traj_list[layer], pred_score_list[layer]
    #
    #         l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
    #         best_mode = torch.argmin(l2_norm, dim=-1)
    #         y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]
    #
    #         agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
    #         agent_cls_loss = F.cross_entropy(pi, best_mode.detach())
    #
    #         target_reg += agent_reg_loss
    #         target_cls += agent_cls_loss
    #         loss_result.update({f"reg_loss_layer{layer}": agent_reg_loss})
    #         loss_result.update({f"cls_loss_layer{layer}": agent_cls_loss})
    #
    #
    #     # others_reg_mask = ~data["x_padding_mask"][:, :, 50:]
    #     # others_reg_loss = F.smooth_l1_loss(
    #     #     y_hat_others[others_reg_mask], y_others[others_reg_mask]
    #     # )
    #     #
    #     loss = target_reg + target_cls
    #     loss_result.update({
    #         "loss": loss,
    #         # "others_reg_loss": others_reg_loss.item(),
    #                    })
    #
    #     return loss_result

    # AutomaticWeightedLoss
    def cal_loss(self, out, data):
        from AutomaticWeightedLoss.AutomaticWeightedLoss import AutomaticWeightedLoss
        multi_task_loss = AutomaticWeightedLoss(3)
        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        y, y_others = data["y"][:, 0], data["y"][:, :]

        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]

        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())

        others_reg_mask = ~data["x_padding_mask"][:, :, 50:]
        others_reg_loss = F.smooth_l1_loss(
            y_hat_others[others_reg_mask], y_others[others_reg_mask]
        )

        loss = multi_task_loss(agent_reg_loss,agent_cls_loss,others_reg_loss)

        return {
            "loss": loss,
            "reg_loss": agent_reg_loss.item(),
            "cls_loss": agent_cls_loss.item(),
            "others_reg_loss": others_reg_loss.item(),
        }

    def training_step(self, data, batch_idx):
        # 验证mask作用的demo function
        # input = data['lane_positions']
        # print(input.shape)
        # mask = ~data['x_padding_mask'][:, :, :50, None]
        # print(mask.shape)
        # valid = data['x'] * mask
        # print(valid.shape)
        # vel_diff = data['x_velocity_diff'][0, ...].cpu().detach().numpy()
        # input = input.cpu().detach().numpy()
        # valid = valid.cpu().detach().numpy()
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(2, 2)
        # for i in range(input.shape[1]):
        #     ax[0, 0].plot(input[0, i, :, 0], input[0, i, :, 1])
        # for i in range(valid.shape[1]):
        #     ax[0, 1].plot(valid[0, i, :, 0], valid[0, i, :, 1])
        # ax[1, 0].matshow(vel_diff)
        # plt.show()

        out = self(data)
        losses = self.cal_loss(out, data)

        for k, v in losses.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        return losses["loss"]

    def validation_step(self, data, batch_idx):
        out = self(data)
        losses = self.cal_loss(out, data)
        metrics = self.val_metrics(out, data["y"][:, 0])

        for k, v in losses.items():
            self.log(
                f"val/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        for k, v in metrics.items():
            self.log(
                f"val/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

    def on_test_start(self) -> None:
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        self.submission_handler = SubmissionAv2(
            save_dir=save_dir
        )

    def test_step(self, data, batch_idx) -> None:
        out = self(data)
        self.submission_handler.format_data(data, out["y_hat"], out["pi"])

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            if 'decoder' in module_name:
                for param_name, param in module.named_parameters():
                    full_param_name = (
                        "%s.%s" % (module_name, param_name) if module_name else param_name
                    )
                    if "bias" in param_name:
                        if 'net.interaction_stage.0.future_encoder.mlp.0.bias' in no_decay and 'future_encoder' in full_param_name:
                            pass
                        else:
                            no_decay.add(full_param_name)
                    elif "weight" in param_name:
                        if isinstance(module, whitelist_weight_modules):
                            decay.add(full_param_name)
                        elif isinstance(module, blacklist_weight_modules):
                            no_decay.add(full_param_name)
                    elif not ("weight" in param_name or "bias" in param_name):
                        no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        # assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        return [optimizer], [scheduler]
