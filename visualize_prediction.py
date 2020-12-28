#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from l5kit.configs import load_config_data
from l5kit.dataset import AgentDataset
from l5kit.geometry import transform_points
from l5kit.visualization import (
    PREDICTED_POINTS_COLOR,
    TARGET_POINTS_COLOR,
    draw_trajectory,
)
from torch.utils.data import DataLoader

import lyft_loss
import lyft_utils
import run_lyft_mpred

CFG_PATH = "./agent_motion_config.yaml"
VIS_SELECTED_FRAMES = (99,)


def visualize_prediction(
    model,
    data_loader: DataLoader,
    rasterizer: AgentDataset,
    args: argparse.Namespace,
    device,
    save_name: str = "agent_train_feature",
) -> None:
    model.eval()
    model.to(device)
    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        dataiter = tqdm.tqdm(data_loader)
        for i, data in enumerate(dataiter):
            image = data["image"].to(device)
            pred, confidences = model(image)
            losses = lyft_loss.pytorch_neg_multi_log_likelihood_batch(
                data["target_positions"].to(device),
                pred,
                confidences,
                data["target_availabilities"].to(device),
                is_reduce=False,
            )

            #  model output
            losses = losses.cpu().numpy().squeeze()
            pred = pred.cpu().numpy()
            confidences = confidences.cpu().numpy()
            # meta data
            images = image.cpu().numpy()
            raster_from_agents = data["raster_from_agent"].numpy()
            timestamps = data["timestamp"].numpy()
            track_ids = data["track_id"].numpy()
            # target data
            target_availabilities = data["target_availabilities"].numpy()
            target_positions = data["target_positions"].numpy()

            # ploting
            for ba_ind in range(len(pred)):
                _, axarr = plt.subplots(1, 4, figsize=(5 * 4, 5))
                # input image
                im = images[ba_ind].transpose(1, 2, 0)
                im = rasterizer.to_rgb(im)
                # plotting ground truth
                im_gt = im.copy()
                target_positions_pixels = transform_points(
                    target_positions[ba_ind],
                    raster_from_agents[ba_ind],
                )
                draw_trajectory(im_gt, target_positions_pixels, TARGET_POINTS_COLOR)
                axarr[0].imshow(im_gt[::-1])
                axarr[0].set_title(
                    "ground truth/pink, loss:{:>3.1f}".format(losses[ba_ind])
                )
                # plotting prediction of each mode
                mode_inds = np.argsort(confidences[ba_ind])[::-1]
                for p in range(1, 4):
                    im_pred = im.copy()
                    mode_ind = mode_inds[p - 1]
                    pred_positions_pixels = transform_points(
                        pred[ba_ind][mode_ind],
                        raster_from_agents[ba_ind],
                    )
                    draw_trajectory(
                        im_pred,
                        pred_positions_pixels[target_availabilities[ba_ind] > 0],
                        PREDICTED_POINTS_COLOR,
                    )
                    axarr[p].imshow(im_pred[::-1])
                    axarr[p].set_title(
                        "mode:{}, confidence:{:>3.2f}/cyan".format(
                            p, confidences[ba_ind][mode_ind]
                        )
                    )
                # save path
                save_path = os.path.join(
                    args.output_dir,
                    f"{timestamps[ba_ind]}_{track_ids[ba_ind]}.png",
                )
                plt.savefig(save_path)
                plt.show()


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
    mpred_dm = run_lyft_mpred.LyftMpredDatamodule(  # type: ignore[abstract]
        args.l5kit_data_folder,
        cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        downsample_train=args.downsample_train,
        is_test=False,
        is_debug=args.is_debug,
    )
    mpred_dm.prepare_data()
    mpred_dm.setup()
    val_datalader = mpred_dm.val_dataloader()
    rasterizer = mpred_dm.rasterizer

    print("load from: ", args.ckpt_path)
    model = run_lyft_mpred.LitModel.load_from_checkpoint(args.ckpt_path, cfg=cfg)
    device = torch.device("cuda")
    visualize_prediction(model, val_datalader, rasterizer, args, device)


if __name__ == "__main__":
    cfg = load_config_data(CFG_PATH)

    parser = argparse.ArgumentParser(
        description="Visualize lyft motion prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--l5kit_data_folder",
        default="/your/dataset/path",
        type=str,
        help="root directory path for lyft motion prediction dataset",
    )
    parser.add_argument(
        "--num_modes",
        type=int,
        default=3,
        help="number of the modes on each prediction",
    )
    parser.add_argument("--batch_size", type=int, default=220, help="batch size")
    parser.add_argument(
        "--downsample_train",
        action="store_true",
        help="using only 4 frames from each scene, the loss converge is \
much faster than using all data, but it will get larger loss",
    )
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
        "--num_workers",
        default="16",
        type=int,
        help="number of cpus for DataLoader",
    )
    parser.add_argument("--is_debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./prediction_images",
        help="directory for visualized images",
    )

    args = parser.parse_args()

    if args.is_debug:
        DEBUG = True
        print("\t ---- DEBUG RUN ---- ")
        cfg["train_data_loader"]["key"] = "scenes/sample.zarr"
        cfg["val_data_loader"]["key"] = "scenes/sample.zarr"
        args.batch_size = 16
    else:
        DEBUG = False
        print("\t ---- NORMAL RUN ---- ")
    lyft_utils.print_argparse_arguments(args)
    main(cfg, args)
