import os
import hydra
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf):
    pl.seed_everything(conf.seed, workers=True)
    output_dir = HydraConfig.get().runtime.output_dir

    if conf.wandb != "disable":
        logger = WandbLogger(
            project="Forecast-MAE",
            name=conf.output,
            mode=conf.wandb,
            log_model="all",
            resume=conf.checkpoint is not None,
        )
    else:
        task_name = conf.task_name
        logger = TensorBoardLogger(save_dir=output_dir, name=f"{task_name}")

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="{epoch}",
            monitor=f"{conf.monitor}",
            mode="min",
            save_top_k=conf.save_top_k,
            save_last=True,
        ),
        RichModelSummary(max_depth=1),
        RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        logger=logger,
        gradient_clip_val=conf.gradient_clip_val,
        gradient_clip_algorithm=conf.gradient_clip_algorithm,
        max_epochs=conf.epochs,
        accelerator="gpu",
        devices=conf.gpus,
        strategy="ddp_find_unused_parameters_false" if conf.gpus > 1 else None,
        callbacks=callbacks,
        limit_train_batches=conf.limit_train_batches,
        limit_val_batches=conf.limit_val_batches,
        sync_batchnorm=conf.sync_bn,
    )
    print('conf_checkpoint',conf.checkpoint)
    model = instantiate(conf.model.target)
    print(model)
    # print('pretrained_checkpoint', conf.checkpoint)

    print(conf)
    logger.log_graph(model)
    datamodule = instantiate(conf.datamodule)
    trainer.fit(model, datamodule, ckpt_path=conf.checkpoint)


if __name__ == "__main__":
    main()


# data_root=/home/fu/argoverse2_forcast_mae
# model=model_forecast
# gpus=1
# batch_size=32
# monitor=val_minFDE6
# pretrained_weights="/home/fu/argoverse2_forcast_mae/pretrain_ckpt/last.ckpt"

#
# data_root=/home/fu/argoverse2_forcast_mae
# model=model_mae
# gpus=1
# batch_size=32
# monitor=val_minFDE6