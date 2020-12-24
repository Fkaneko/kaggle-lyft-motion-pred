#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import random
import sys

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
from tqdm import tqdm

import lyft_loss
import lyft_models
import lyft_utils

ALL_DATA_SIZE = 198474478
VAL_INTERVAL_SAMPLES = 250000
CFG_PATH = "./agent_motion_config.yaml"

# leaderborad test dataset configuration for agents, same as
# l5kit.evaluation.create_chopped_dataset
# minimum number of frames an agents must have in the past to be picked
MIN_FRAME_HISTORY = 0
# minimum number of frames an agents must have in the future to be picked
MIN_FRAME_FUTURE = 10
VAL_SELECTED_FRAME = 99


class LyftMpredModel(pl.LightningModule):
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

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs = batch["image"]
        target_availabilities = batch["target_availabilities"].unsqueeze(-1)
        targets = batch["target_positions"]
        outputs = self.model(inputs)

        outputs, confidences = outputs
        loss = self.criterion(outputs, targets, target_availabilities, confidences)
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

        outputs = self.model(inputs)
        outputs, confidences = outputs
        loss = self.criterion(outputs, targets, target_availabilities, confidences)
        self.log("val_loss", loss)
        return loss

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

    def criterion(self, outputs, targets, target_availabilities, confidences):
        loss = lyft_loss.pytorch_neg_multi_log_likelihood_batch(
            targets,
            outputs,
            confidences.squeeze(),
            target_availabilities.squeeze(),
        )
        return loss


def run_prediction(model: pl.LightningModule, data_loader: DataLoader) -> tuple:
    """from https://www.kaggle.com/pestipeti/pytorch-baseline-inference"""
    DEVICE = torch.device("cuda")
    model.to(DEVICE)
    model.eval()
    model.freeze()

    pred_coords_list = []
    confidences_list = []
    timestamps_list = []
    track_id_list = []

    with torch.no_grad():
        dataiter = tqdm(data_loader)
        for data in dataiter:
            image = data["image"].to(DEVICE)
            preds, confidences = model(image)

            # fix for the new environment
            preds = preds.cpu().numpy()
            world_from_agents = data["world_from_agent"].numpy()
            centroids = data["centroid"].numpy()

            # convert into world coordinates and compute offsets
            for idx in range(len(preds)):
                for mode in range(3):
                    preds[idx, mode, :, :] = (
                        transform_points(preds[idx, mode, :, :], world_from_agents[idx])
                        - centroids[idx][:2]
                    )

            pred_coords_list.append(preds)
            confidences_list.append(confidences.cpu().numpy().copy())
            timestamps_list.append(data["timestamp"].numpy().copy())
            track_id_list.append(data["track_id"].numpy().copy())
    timestamps = np.concatenate(timestamps_list)
    track_ids = np.concatenate(track_id_list)
    coords = np.concatenate(pred_coords_list)
    confs = np.concatenate(confidences_list)
    return timestamps, track_ids, coords, confs


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
    os.environ["L5KIT_DATA_FOLDER"] = args.l5kit_data_folder
    dm = LocalDataManager(None)
    rasterizer = build_rasterizer(cfg, dm)

    if args.is_test:
        test_cfg = cfg["test_data_loader"]
        test_zarr_path = dm.require(test_cfg["key"])
        print("test path", test_zarr_path)
        test_mask_path = os.path.join(os.path.dirname(test_zarr_path), "mask.npz")
        test_gt_path = os.path.join(os.path.dirname(test_zarr_path), "gt.csv")

        test_zarr = ChunkedDataset(test_zarr_path).open()
        test_mask = np.load(test_mask_path)["arr_0"]

        # ===== INIT DATASET AND LOAD MASK
        test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
        test_dataloader = DataLoader(
            test_dataset,
            shuffle=test_cfg["shuffle"],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(test_dataset)

        print("load from: ", args.ckpt_path)
        model = LyftMpredModel.load_from_checkpoint(args.ckpt_path, cfg=cfg)

        # --- Inference ---
        timestamps, track_ids, coords, confs = run_prediction(model, test_dataloader)
        csv_path = "submission.csv"
        write_pred_csv(
            csv_path,
            timestamps=timestamps,
            track_ids=track_ids,
            coords=coords,
            confs=confs,
        )
        print(f"Saved to {csv_path}")
        if os.path.exists(test_gt_path):
            metrics = compute_metrics_csv(
                test_gt_path, csv_path, [neg_multi_log_likelihood, time_displace]
            )
            for metric_name, metric_mean in metrics.items():
                print(metric_name, metric_mean)

    else:
        # ===== INIT DATASET
        train_cfg = cfg["train_data_loader"]
        train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
        train_dataset = AgentDataset(
            cfg,
            train_zarr,
            rasterizer,
            min_frame_history=MIN_FRAME_HISTORY,
            min_frame_future=MIN_FRAME_FUTURE,
        )

        val_cfg = cfg["val_data_loader"]
        val_zarr = ChunkedDataset(dm.require(val_cfg["key"])).open()
        val_dataset = AgentDataset(
            cfg,
            val_zarr,
            rasterizer,
            min_frame_history=MIN_FRAME_HISTORY,
            min_frame_future=MIN_FRAME_FUTURE,
        )
        print(train_dataset)
        print(val_dataset)

        print("Ploting dataset")
        PLOT_NUM = 10
        ind = np.random.randint(0, len(train_dataset), size=PLOT_NUM)
        for i in range(PLOT_NUM):
            data = train_dataset[ind[i]]
            im = data["image"].transpose(1, 2, 0)
            im = train_dataset.rasterizer.to_rgb(im)
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
            if DEBUG:
                plt.show()

        if args.downsample_train:
            print("downsampling train agents, using only 4 frames from each scene")
            train_agents_list = lyft_utils.downsample_agents(
                train_zarr,
                train_dataset,
                selected_frames=lyft_utils.TRAIN_DSAMPLE_FRAMES,
            )
            train_dataset = torch.utils.data.Subset(train_dataset, train_agents_list)

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # downsampling the validation dataset same as test dataset or
        # l5kit.evaluation.create_chopped_dataset
        val_agents_list = lyft_utils.downsample_agents(
            val_zarr, val_dataset, selected_frames=[VAL_SELECTED_FRAME]
        )
        val_dataset = torch.utils.data.Subset(val_dataset, val_agents_list)
        val_dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(
            f"training samples: {len(train_dataset)}, valid samples: {len(val_dataset)}"
        )

        total_steps = args.epochs * len(train_dataset) // args.batch_size
        val_check_interval = VAL_INTERVAL_SAMPLES // args.batch_size

        model = LyftMpredModel(
            cfg,
            lr=args.lr,
            backbone_name=args.backbone_name,
            num_modes=args.num_modes,
            optim_name=args.optim_name,
            ba_size=args.batch_size,
            epochs=args.epochs,
            data_size=len(train_dataset),
            total_steps=total_steps,
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            save_last=True,
            mode="min",
            verbose=True,
        )
        pl.trainer.seed_everything(seed=SEED)
        precision = 16
        trainer = pl.Trainer(
            gpus=len(args.visible_gpus.split(",")),
            max_steps=total_steps,
            val_check_interval=val_check_interval,
            precision=precision,
            benchmark=True,
            deterministic=False,
            checkpoint_callback=checkpoint_callback,
        )

        # Run lr finder
        if args.find_lr:
            lr_finder = trainer.tuner.lr_find(model, train_dataloader, num_training=500)
            lr_finder.plot(suggest=True)
            plt.show()
            sys.exit()

        # Run Training
        trainer.fit(model, train_dataloader, val_dataloader)


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
        print("\t ---- DEBUG RUN")
        cfg["train_data_loader"]["key"] = "scenes/sample.zarr"
        cfg["val_data_loader"]["key"] = "scenes/sample.zarr"
    else:
        DEBUG = False
        print("\t ---- NORMAL RUN")
    lyft_utils.print_argparse_arguments(args)
    main(cfg, args)
