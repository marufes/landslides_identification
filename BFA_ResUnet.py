########################################################################################################################
# BFA_ResUnet (with edge detection at input stage)
########################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict

from backbone_resnet import resnet50
from Unet_decode import Up, OutConv

# ============================================================
# HED Network (already in your code)
# ============================================================
class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.netVggOne = torch.nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=False)
        )
        self.netVggTwo = torch.nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=False)
        )
        self.netVggThr = torch.nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=False)
        )
        self.netVggFou = torch.nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=False)
        )
        self.netVggFiv = torch.nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=False)
        )

        self.netScoreOne = nn.Conv2d(64, 1, 1)
        self.netScoreTwo = nn.Conv2d(128, 1, 1)
        self.netScoreThr = nn.Conv2d(256, 1, 1)
        self.netScoreFou = nn.Conv2d(512, 1, 1)
        self.netScoreFiv = nn.Conv2d(512, 1, 1)

        self.netCombine = nn.Sequential(
            nn.Conv2d(5, 1, 1),
            nn.Sigmoid()
        )

        self.load_state_dict({
            strKey.replace('module', 'net'): tenWeight
            for strKey, tenWeight in torch.hub.load_state_dict_from_url(
                url='http://content.sniklaus.com/github/pytorch-hed/network-bsds500.pytorch',
                file_name='hed-bsds500'
            ).items()
        })

    def forward(self, tenInput):
        tenInput = tenInput * 255.0
        tenInput = tenInput - torch.tensor(
            [104.00698793, 116.66876762, 122.67891434],
            dtype=tenInput.dtype,
            device=tenInput.device
        ).view(1, 3, 1, 1)

        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = F.interpolate(tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreTwo = F.interpolate(tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreThr = F.interpolate(tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFou = F.interpolate(tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFiv = F.interpolate(tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)

        return self.netCombine(torch.cat([tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv], 1))

# Singleton pattern for edge detector
netNetwork = None
def estimate(tenInput):
    global netNetwork
    if netNetwork is None:
        netNetwork = Network().cuda().eval()
    return netNetwork(tenInput)[0: , :, :, :]

# ============================================================
# BFA Module (feature + edge map fusion)
# ============================================================
class BFA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BFA, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels + 1, out_channels, kernel_size=1) 
        self.relu = nn.ReLU()

    def forward(self, feature_map, edge_image):
        # Resize edge map to match feature spatial size
        edge_resized = F.interpolate(edge_image, size=feature_map.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate along channel dimension
        concatenated = torch.cat([feature_map, edge_resized], dim=1)

        output = self.conv1x1(concatenated)
        output = self.relu(output)
        return output

# ============================================================
# Helper: IntermediateLayerGetter
# ============================================================
class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

# ============================================================
# Full BFA-ResUNet
# ============================================================
class BFA_resunet(nn.Module):
    def __init__(self, num_classes, pretrain_backbone: bool = False):
        super(BFA_resunet, self).__init__()
        backbone = resnet50()

        if pretrain_backbone:
            backbone.load_state_dict(torch.load("resnet50.pth", map_location='cpu'), strict=False)

        self.stage_out_channels = [64, 256, 512, 1024, 2048]
        return_layers = {
            'relu': 'out0',
            'layer1': 'out1',
            'layer2': 'out2',
            'layer3': 'out3',
            'layer4': 'out4'
        }
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        # BFA block for deepest feature
        self.bfa = BFA(in_channels=self.stage_out_channels[4], out_channels=self.stage_out_channels[4])

        c = self.stage_out_channels[4] + self.stage_out_channels[3]
        self.up1 = Up(c, self.stage_out_channels[3])
        c = self.stage_out_channels[3] + self.stage_out_channels[2]
        self.up2 = Up(c, self.stage_out_channels[2])
        c = self.stage_out_channels[2] + self.stage_out_channels[1]
        self.up3 = Up(c, self.stage_out_channels[1])
        c = self.stage_out_channels[1] + self.stage_out_channels[0]
        self.up4 = Up(c, self.stage_out_channels[0])

        self.conv = OutConv(64, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_shape = x.shape[-2:]

        # ---- Edge detection on raw input ----
        edge_image = estimate(x)  # shape [B, 1, H, W]

        # ---- Backbone ----
        backbone_out = self.backbone(x)

        # ---- Apply BFA at deepest layer ----
        backbone_out['out4'] = self.bfa(backbone_out['out4'], edge_image)

        # ---- Decoder ----
        x = self.up1(backbone_out['out4'], backbone_out['out3'])
        x = self.up2(x, backbone_out['out2'])
        x = self.up3(x, backbone_out['out1'])
        x = self.up4(x, backbone_out['out0'])
        x = self.conv(x)

        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        return {"out": x}
