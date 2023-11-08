from typing import Any, Callable, Dict, Optional

import torch
from torchmetrics import Metric

from .utils import sort_predictions


class multilayer_minADE(Metric):
    """Minimum Average Displacement Error
    minADE: The average L2 distance between the best forecasted trajectory and the ground truth.
            The best here refers to the trajectory that has the minimum endpoint error.
    """

    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        layer_num,
        k=6,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super(multilayer_minADE, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.layer_num = layer_num
        self.k = k
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> None:
        with torch.no_grad():
            pred_traj_list = outputs["y_hat"]
            pred_score_list = outputs["pi"]
            assert len(pred_score_list)==len(pred_traj_list)
            pred, _ = sort_predictions(pred_traj_list[self.layer_num], pred_score_list[self.layer_num], k=self.k)
            ade = torch.norm(
                pred[..., :2] - target.unsqueeze(1)[..., :2], p=2, dim=-1
            ).mean(-1)
            min_ade = ade.min(-1)[0]
            self.sum += min_ade.sum()
            self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


    #     self.add_state("sum", default=torch.zeros([self.layer_nums, 1]), dist_reduce_fx="sum")
    #     self.add_state("count", default=torch.zeros([self.layer_nums, 1]), dist_reduce_fx="sum")
    #
    # def update(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> None:
    #     with torch.no_grad():
    #         pred_traj_list = outputs["y_hat"]
    #         pred_score_list = outputs["pi"]
    #         assert len(pred_score_list)==len(pred_traj_list)==self.layer_nums
    #         for layer in range(self.layer_nums):
    #             pred, _ = sort_predictions(pred_traj_list[layer], pred_score_list[layer], k=self.k)
    #             ade = torch.norm(
    #                 pred[..., :2] - target.unsqueeze(1)[..., :2], p=2, dim=-1
    #             ).mean(-1)
    #             min_ade = ade.min(-1)[0]
    #             self.sum[layer,:] += min_ade.sum()
    #             self.count[layer,:] += pred.size(0)
    #
    # def compute(self) -> torch.Tensor:
    #     return self.sum / self.count
