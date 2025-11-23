"""Loss functions for adversarial patch training.

This module contains various loss functions used in adversarial patch generation:
- MaxProbExtractor: Extracts max class probability from YOLO output
- NPSCalculator: Non-printability score calculation
- TotalVariation: Total variation loss for smoothness
- ContentLoss: Content loss for style transfer
- AdaINStyleLoss: Adaptive Instance Normalization style loss
"""

from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MaxProbExtractor(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id: int, num_cls: int, config: Any) -> None:
        super(MaxProbExtractor, self).__init__()
        self.cls_id: int = cls_id
        self.num_cls: int = num_cls
        self.config: Any = config

    def forward(self, YOLOoutput: torch.Tensor) -> torch.Tensor:
        # get values neccesary for transformation
        if YOLOoutput.dim() == 3:
            YOLOoutput = YOLOoutput.unsqueeze(0)
        batch = YOLOoutput.size(0)
        assert (YOLOoutput.size(1) == (5 + self.num_cls ) * 5)
        h = YOLOoutput.size(2)
        w = YOLOoutput.size(3)
        # transform the output tensor from [batch, 425, 19, 19] to [batch, 80, 1805]
        output = YOLOoutput.view(batch, 5, 5 + self.num_cls , h * w)  # [batch, 5, 85, 361]
        output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 361]
        output = output.view(batch, 5 + self.num_cls , 5 * h * w)  # [batch, 85, 1805]
        output_objectness = torch.sigmoid(output[:, 4, :])  # [batch, 1805]
        output = output[:, 5:5 + self.num_cls , :]  # [batch, 80, 1805]
        # perform softmax to normalize probabilities for object classes to [0,1]
        normal_confs = torch.nn.Softmax(dim=1)(output)
        # we only care for probabilities of the class of interest (person)
        confs_for_class = normal_confs[:, self.cls_id, :]
        confs_if_object = output_objectness #confs_for_class * output_objectness
        confs_if_object = confs_for_class * output_objectness
        confs_if_object = self.config.loss_target(output_objectness, confs_for_class)
        # find the max probability for person
        max_conf, max_conf_idx = torch.max(confs_if_object, dim=1)

        return max_conf


class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file: str, patch_side: int) -> None:
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, patch_side),requires_grad=False)

    def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
        # calculate euclidian distance between colors in patch and colors in printability_array
        # square root of sum of squared difference
        color_dist = (adv_patch - self.printability_array+0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1)+0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0] #test: change prod for min (find distance to closest color)
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod,0)
        nps_score = torch.sum(nps_score,0)
        return nps_score/torch.numel(adv_patch)

    def get_printability_array(self, printability_file: str, side: int) -> torch.Tensor:
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self) -> None:
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001),0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001),0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
        tv = tvcomp1 + tvcomp2
        return tv/torch.numel(adv_patch)


class ContentLoss(nn.Module):
    """ContentLoss: calculates the content loss.
    """

    def __init__(self) -> None:
        super(ContentLoss, self).__init__()

    def forward(self, adv_patch: torch.Tensor, orig_img: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(adv_patch, orig_img)


class AdaINStyleLoss(nn.Module):
    """Adaptive Instance Normalization Style Loss.

    Uses VGG19 encoder layers to compute style loss based on
    feature statistics (mean and standard deviation).
    """

    def __init__(self, device: str = 'cuda:0') -> None:
        super(AdaINStyleLoss, self).__init__()
        encoder_layers = list(models.vgg19(pretrained=True).features)
        encoder_1 = torch.nn.Sequential(*encoder_layers[:2])
        encoder_2 = torch.nn.Sequential(*encoder_layers[2:8])
        encoder_3 = torch.nn.Sequential(*encoder_layers[8:14])
        encoder_4 = torch.nn.Sequential(*encoder_layers[14:26])
        self.encoder_layers_list: List[torch.nn.Sequential] = [
            encoder_1.to(device),
            encoder_2.to(device),
            encoder_3.to(device),
            encoder_4.to(device),
        ]

    def _encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        outputs = [input]
        for encoder_layer in self.encoder_layers_list:
            input = encoder_layer(input)
            outputs.append(input)
        return outputs

    def _statistics(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        size = features.size()
        batch_size, n_channels = size[0], size[1]
        features_flatten = features.view(
            batch_size, n_channels, -1)

        mean = features_flatten.mean(2)
        mean = mean.view(batch_size, n_channels, 1, 1)

        eps = 1e-7
        features_var = features_flatten.var(dim=2) + eps
        std = features_var.sqrt().view(batch_size, n_channels, 1, 1)
        return mean, std

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        content_features_list = self._encode(content)
        style_features_list = self._encode(style)

        style_loss = 0
        for i in range(len(content_features_list)):
            content_features = content_features_list[i]
            style_features = style_features_list[i]
            size = content_features.size()
            batch_size, n_channels = size[0], size[1]
            content_mean, content_std = self._statistics(content_features)
            style_mean, style_std = self._statistics(style_features)

            style_loss += F.mse_loss(content_mean, style_mean) + \
                F.mse_loss(content_std, style_std)
        return style_loss
