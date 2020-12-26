#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import random
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import AgentDataset
from l5kit.evaluation import compute_metrics_csv, write_pred_csv
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.rasterization import build_rasterizer
from l5kit.visualization import TARGET_POINTS_COLOR, draw_trajectory
from torch.utils.data import DataLoader

import lyft_loss
import lyft_models
import lyft_utils

ALL_DATA_SIZE = 198474478
VAL_INTERVAL_SAMPLES = 250000
CFG_PATH = "./agent_motion_config.yaml"

# for using the same sampling as test dataset agents,
# these two FRAME settings are requried.
# minimum number of frames an agents must have in the past to be picked
MIN_FRAME_HISTORY = 0
# minimum number of frames an agents must have in the future to be picked
MIN_FRAME_FUTURE = 10
VAL_SELECTED_FRAME = (99,)

# output path for test mode
CSV_PATH = "./submission.csv"


class LyftMpredDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        l5kit_data_folder: str,
        cfg: dict,
        batch_size: int = 440,
        num_workers: int = 16,
        downsample_train: bool = False,
        is_test: bool = False,
        is_debug: bool = False,
    ) -> None:
        super().__init__()
        os.environ["L5KIT_DATA_FOLDER"] = l5kit_data_folder
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.downsample_train = downsample_train
        self.is_test = is_test
        self.is_debug = is_debug

    def prepare_data(self):
        # called only on 1 GPU
        self.dm = LocalDataManager(None)
        self.rasterizer = build_rasterizer(cfg, self.dm)

    def setup(self):
        # called on every GPU
        if self.is_test:
            print("test mode setup")
            self.test_path, test_zarr, self.test_dataset = self.load_zarr_dataset(
                loader_name="test_data_loader"
            )
        else:
            print("train mode setup")
            self.train_path, train_zarr, self.train_dataset = self.load_zarr_dataset(
                loader_name="train_data_loader"
            )
            self.val_path, val_zarr, self.val_dataset = self.load_zarr_dataset(
                loader_name="val_data_loader"
            )
            self.plot_dataset(self.train_dataset)

            if self.downsample_train:
                print(
                    "downsampling agents, using only {} frames from each scene".format(
                        len(lyft_utils.TRAIN_DSAMPLE_FRAMES)
                    )
                )
                train_agents_list = lyft_utils.downsample_agents(
                    train_zarr,
                    self.train_dataset,
                    selected_frames=lyft_utils.TRAIN_DSAMPLE_FRAMES,
                )
                self.train_dataset = torch.utils.data.Subset(
                    self.train_dataset, train_agents_list
                )
            # downsampling the validation dataset same as test dataset or
            # l5kit.evaluation.create_chopped_dataset
            val_agents_list = lyft_utils.downsample_agents(
                val_zarr, self.val_dataset, selected_frames=VAL_SELECTED_FRAME
            )
            self.val_dataset = torch.utils.data.Subset(
                self.val_dataset, val_agents_list
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def load_zarr_dataset(
        self, loader_name: str = "train_data_loder"
    ) -> Tuple[str, ChunkedDataset, AgentDataset]:

        zarr_path = self.dm.require(self.cfg[loader_name]["key"])
        print("load zarr data:", zarr_path)
        zarr_dataset = ChunkedDataset(zarr_path).open()
        if loader_name == "test_data_loader":
            mask_path = os.path.join(os.path.dirname(zarr_path), "mask.npz")
            agents_mask = np.load(mask_path)["arr_0"]
            agent_dataset = AgentDataset(
                self.cfg, zarr_dataset, self.rasterizer, agents_mask=agents_mask
            )
        else:
            agent_dataset = AgentDataset(
                self.cfg,
                zarr_dataset,
                self.rasterizer,
                min_frame_history=MIN_FRAME_HISTORY,
                min_frame_future=MIN_FRAME_FUTURE,
            )
        print(zarr_dataset)
        return zarr_path, zarr_dataset, agent_dataset

    def plot_dataset(self, agent_dataset: AgentDataset, plot_num: int = 10) -> None:
        print("Ploting dataset")
        ind = np.random.randint(0, len(agent_dataset), size=plot_num)
        for i in range(plot_num):
            data = agent_dataset[ind[i]]
            im = data["image"].transpose(1, 2, 0)
            im = agent_dataset.rasterizer.to_rgb(im)
            target_positions_pixels = transform_points(
                data["target_positions"], data["raster_from_agent"]
            )
            draw_trajectory(
                im,
                target_positions_pixels,
                TARGET_POINTS_COLOR,
                yaws=data["target_yaws"],
            )
            plt.imshow(im[::-1])
            if self.is_debug:
                plt.show()


class LitModel(pl.LightningModule):
    def __init__(
        self,
        cfg: dict,
        num_modes: int = 3,
        ba_size: int = 128,
        lr: float = 3.0e-4,
        backbone_name: str = "efficientnet_b1",
        epochs: int = 1,
        total_steps: int = 100,
        data_size: int = ALL_DATA_SIZE,
        optim_name: str = "adam",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            "lr",
            "backbone_name",
            "num_modes",
            "ba_size",
            "epochs",
            "optim_name",
            "data_size",
            "total_steps",
        )
        self.model = lyft_models.LyftMultiModel(
            cfg, num_modes=num_modes, backbone_name=backbone_name
        )
        self.test_keys = ("world_from_agent", "centroid", "timestamp", "track_id")

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs = batch["image"]
        target_availabilities = batch["target_availabilities"].unsqueeze(-1)
        targets = batch["target_positions"]

        outputs, confidences = self.model(inputs)
        loss = lyft_loss.pytorch_neg_multi_log_likelihood_batch(
            targets,
            outputs,
            confidences.squeeze(),
            target_availabilities.squeeze(),
        )
        self.log(
            "train_epoch_loss",
            loss,
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["image"]
        target_availabilities = batch["target_availabilities"].unsqueeze(-1)
        targets = batch["target_positions"]

        outputs, confidences = self.model(inputs)
        loss = lyft_loss.pytorch_neg_multi_log_likelihood_batch(
            targets,
            outputs,
            confidences.squeeze(),
            target_availabilities.squeeze(),
        )
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        inputs = batch["image"]
        outputs, confidences = self.model(inputs)
        test_batch = {key_: batch[key_] for key_ in self.test_keys}

        return outputs, confidences, test_batch

    def test_epoch_end(self, outputs):
        """from https://www.kaggle.com/pestipeti/pytorch-baseline-inference"""
        pred_coords_list = []
        confidences_list = []
        timestamps_list = []
        track_id_list = []

        # convert into world coordinates and compute offsets
        for outputs, confidences, batch in outputs:
            outputs = outputs.cpu().numpy()

            world_from_agents = batch["world_from_agent"].cpu().numpy()
            centroids = batch["centroid"].cpu().numpy()
            for idx in range(len(outputs)):
                for mode in range(3):
                    outputs[idx, mode, :, :] = (
                        transform_points(
                            outputs[idx, mode, :, :], world_from_agents[idx]
                        )
                        - centroids[idx][:2]
                    )
            pred_coords_list.append(outputs)

            confidences_list.append(confidences)
            timestamps_list.append(batch["timestamp"])
            track_id_list.append(batch["track_id"])

        coords = np.concatenate(pred_coords_list)
        confs = torch.cat(confidences_list).cpu().numpy()
        track_ids = torch.cat(track_id_list).cpu().numpy()
        timestamps = torch.cat(timestamps_list).cpu().numpy()

        write_pred_csv(
            CSV_PATH,
            timestamps=timestamps,
            track_ids=track_ids,
            coords=coords,
            confs=confs,
        )
        print(f"Saved to {CSV_PATH}")

    def configure_optimizers(self):
        if self.hparams.optim_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=0.9,
                weight_decay=4e-5,
            )
        elif self.hparams.optim_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        else:
            raise NotImplementedError
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.hparams.lr, total_steps=self.hparams.total_steps
        )
        return [optimizer], [scheduler]


def main(cfg: dict, args: argparse.Namespace) -> None:
    # set random seeds
    SEED = 42
    os.environ["PYTHONHASHSEED"] = str(SEED)
    # set Python random seed
    random.seed(SEED)
    # set NumPy random seed
    np.random.seed(SEED)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    # ===== Configure LYFT dataset
    # mypy error due to pl.DataModule.transfer_batch_to_device
    mpred_dm = LyftMpredDatamodule(  # type: ignore[abstract]
        args.l5kit_data_folder,
        cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        downsample_train=args.downsample_train,
        is_test=args.is_test,
        is_debug=args.is_debug,
    )
    mpred_dm.prepare_data()
    mpred_dm.setup()

    if args.is_test:
        print("\t\t ==== TEST MODE ====")
        print("load from: ", args.ckpt_path)
        model = LitModel.load_from_checkpoint(args.ckpt_path, cfg=cfg)
        trainer = pl.Trainer(gpus=len(args.visible_gpus.split(",")))
        trainer.test(model, datamodule=mpred_dm)

        test_gt_path = os.path.join(os.path.dirname(mpred_dm.test_path), "gt.csv")
        if os.path.exists(test_gt_path):
            print("test mode with validation chopped dataset, and check the metrics")
            print("validation ground truth path: ", test_gt_path)
            metrics = compute_metrics_csv(
                test_gt_path, CSV_PATH, [neg_multi_log_likelihood, time_displace]
            )
            for metric_name, metric_mean in metrics.items():
                print(metric_name, metric_mean)

    else:
        print("\t\t ==== TRAIN MODE ====")
        print(
            "training samples: {}, valid samples: {}".format(
                len(mpred_dm.train_dataset), len(mpred_dm.val_dataset)
            )
        )
        total_steps = args.epochs * len(mpred_dm.train_dataset) // args.batch_size
        val_check_interval = VAL_INTERVAL_SAMPLES // args.batch_size

        model = LitModel(
            cfg,
            lr=args.lr,
            backbone_name=args.backbone_name,
            num_modes=args.num_modes,
            optim_name=args.optim_name,
            ba_size=args.batch_size,
            epochs=args.epochs,
            data_size=len(mpred_dm.train_dataset),
            total_steps=total_steps,
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            save_last=True,
            mode="min",
            verbose=True,
        )
        pl.trainer.seed_everything(seed=SEED)
        trainer = pl.Trainer(
            gpus=len(args.visible_gpus.split(",")),
            max_steps=total_steps,
            val_check_interval=val_check_interval,
            precision=args.precision,
            benchmark=True,
            deterministic=False,
            checkpoint_callback=checkpoint_callback,
        )

        # Run lr finder
        if args.find_lr:
            lr_finder = trainer.tuner.lr_find(
                model, datamodule=mpred_dm, num_training=500
            )
            lr_finder.plot(suggest=True)
            plt.show()
            sys.exit()

        # Run Training
        trainer.fit(model, datamodule=mpred_dm)


if __name__ == "__main__":
    cfg = load_config_data(CFG_PATH)

    parser = argparse.ArgumentParser(
        description="Run lyft motion prediction learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--l5kit_data_folder",
        default="/your/dataset/path",
        type=str,
        help="root directory path for lyft motion prediction dataset",
    )
    parser.add_argument(
        "--optim_name",
        choices=["adam", "sgd"],
        default="sgd",
        help="optimizer name",
    )
    parser.add_argument(
        "--num_modes",
        type=int,
        default=3,
        help="number of the modes on each prediction",
    )
    parser.add_argument("--lr", default=7.0e-4, type=float, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=220, help="batch size")
    parser.add_argument("--epochs", type=int, default=1, help="epochs for training")
    parser.add_argument(
        "--backbone_name",
        choices=["efficientnet_b1", "seresnext26d_32x4d"],
        default="seresnext26d_32x4d",
        help="backbone name",
    )
    parser.add_argument(
        "--downsample_train",
        action="store_true",
        help="using only 4 frames from each scene, the loss converge is \
much faster than using all data, but it will get larger loss",
    )
    parser.add_argument("--is_test", action="store_true", help="test mode")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./model.pth",
        help="path for model checkpoint at test mode",
    )
    parser.add_argument(
        "--precision",
        default=16,
        choices=[16, 32],
        type=int,
        help="float precision at training",
    )
    parser.add_argument(
        "--visible_gpus",
        type=str,
        default="0",
        help="Select gpu ids with comma separated format",
    )
    parser.add_argument(
        "--find_lr",
        action="store_true",
        help="find lr with fast ai implementation",
    )
    parser.add_argument(
        "--num_workers",
        default="16",
        type=int,
        help="number of cpus for DataLoader",
    )
    parser.add_argument("--is_debug", action="store_true", help="debug mode")

    args = parser.parse_args()

    if args.is_debug:
        DEBUG = True
        print("\t ---- DEBUG RUN ---- ")
        cfg["train_data_loader"]["key"] = "scenes/sample.zarr"
        cfg["val_data_loader"]["key"] = "scenes/sample.zarr"
        VAL_INTERVAL_SAMPLES = 5000
        args.batch_size = 16
    else:
        DEBUG = False
        print("\t ---- NORMAL RUN ---- ")
    lyft_utils.print_argparse_arguments(args)
    main(cfg, args)
