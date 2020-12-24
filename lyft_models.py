from typing import Union

import timm
import torch
import torch.nn as nn

BACKBONE_OUT = {
    "efficientnet_b0": 1280,
    "efficientnet_b1": 1280,
    "efficientnet_b3": 1536,
    "seresnext26d_32x4d": 2048,
}
ModelTypes = Union[timm.models.efficientnet.EfficientNet, timm.models.resnet.ResNet]


def extend_input_channel(model: ModelTypes, channel_scale: int = 2) -> nn.Module:
    """
    Extned 1st conv layer weight of LyftMpredModel pretrained weight
    for longger history input.
    """
    w = model.backbone.conv_stem.weight
    num_hist = (w.shape[1] - 3) // 2 - 1

    other_part = w[:, :num_hist, :, :] * (channel_scale - 1)
    other_part = [other_part, w[:, : num_hist + 1, :, :]]
    target_part = w[:, num_hist + 1 : -4, :, :] * (channel_scale - 1)
    target_part = [target_part, w[:, num_hist + 1 : -3, :, :]]
    map_part = [w[:, -3:, :, :]]

    conv1_weight = nn.Parameter(torch.cat(other_part + target_part + map_part, dim=1))
    num_in_channels = (num_hist * channel_scale + 1) * 2 + 3
    model.backbone.conv_stem = nn.Conv2d(
        num_in_channels,
        model.backbone.conv_stem.out_channels,
        kernel_size=model.backbone.conv_stem.kernel_size,
        stride=model.backbone.conv_stem.stride,
        padding=model.backbone.conv_stem.padding,
        bias=False,
    )
    model.backbone.conv_stem.weight = conv1_weight
    return model


def extend_1st_convw_ch(
    backbone: ModelTypes,
    backbone_name: str,
    num_in_channels: int,
) -> nn.Module:
    """
    Extned 1st conv layer weight of imagenet pretrained weight for multi channel input
    """
    extend_ch = num_in_channels // 3
    if backbone_name.find("efficientnet") > -1:
        w = backbone.conv_stem.weight
    else:
        w = backbone.conv1[0].weight

    if num_in_channels - extend_ch > 0:
        conv1_weight = nn.Parameter(
            torch.cat(
                [w] * extend_ch + [w[:, : (num_in_channels - extend_ch * 3), :, :]],
                dim=1,
            )
        )
    else:
        conv1_weight = nn.Parameter(torch.cat([w] * extend_ch, dim=1))

    if backbone_name.find("efficientnet") > -1:
        backbone.conv_stem = nn.Conv2d(
            num_in_channels,
            backbone.conv_stem.out_channels,
            kernel_size=backbone.conv_stem.kernel_size,
            stride=backbone.conv_stem.stride,
            padding=backbone.conv_stem.padding,
            bias=False,
        )
        backbone.conv_stem.weight = conv1_weight
        backbone.classifier = nn.Identity()
    else:
        backbone.conv1[0] = nn.Conv2d(
            num_in_channels,
            backbone.conv1[0].out_channels,
            kernel_size=backbone.conv1[0].kernel_size,
            stride=backbone.conv1[0].stride,
            padding=backbone.conv1[0].padding,
            bias=False,
        )
        backbone.conv1[0].weight = conv1_weight
        backbone.fc = nn.Identity()

    return backbone


class LyftMultiModel(nn.Module):
    def __init__(
        self, cfg: dict, num_modes: int = 3, backbone_name: str = "efficientnet_b1"
    ) -> None:
        """
        Multi mode prediction net with imagenet pretrained backbone.
        1st conv weight is extedned to multi channel > 3 input.
        """
        super().__init__()
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels
        self.backbone = timm.create_model(backbone_name, pretrained=True)
        backbone_out_features = BACKBONE_OUT[backbone_name]
        self.backbone = extend_1st_convw_ch(
            self.backbone, backbone_name, num_in_channels
        )

        # X, Y coords for the future positions (output shape: batch_sizex50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len

        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        self.logit = nn.Linear(
            backbone_out_features, out_features=self.num_preds + num_modes
        )

    def forward(self, x):
        feature = self.backbone(x)
        x = self.logit(feature)
        # pred (batch_size)x(modes)x(time)x(2D coords)
        # confidences (batch_size)x(modes)
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences
