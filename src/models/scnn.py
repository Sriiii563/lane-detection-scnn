import torch
import torch.nn as nn
import torchvision.models as models
from config import config

# ----- 1. Directional SCNN Layer -----
class SCNNLayer(nn.Module):
    def __init__(self, channels, direction='down'):
        super(SCNNLayer, self).__init__()
        self.direction = direction
        self.conv = nn.Conv2d(channels, channels, kernel_size=9, padding=4, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        n, c, h, w = x.size()

        if self.direction in ['down', 'up']:
            for i in range(1, h) if self.direction == 'down' else range(h - 2, -1, -1):
                x[:, :, i, :] += self.relu(self.conv(x[:, :, i - 1, :].unsqueeze(2)).squeeze(2))
        elif self.direction in ['right', 'left']:
            for j in range(1, w) if self.direction == 'right' else range(w - 2, -1, -1):
                x[:, :, :, j] += self.relu(self.conv(x[:, :, :, j - 1].unsqueeze(3)).squeeze(3))
        return x

# ----- 2. SCNN Model -----
class SCNN(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, num_classes=5):
        super(SCNN, self).__init__()

        # Load backbone
        if backbone == 'resnet18':
            net = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            net = models.resnet34(pretrained=pretrained)
        else:
            raise ValueError("Unsupported backbone")

        # Remove the FC layer and avgpool
        self.backbone = nn.Sequential(*list(net.children())[:-2])  # Output: (B, 512, H/32, W/32)

        self.reduce = nn.Conv2d(512, 128, kernel_size=1)  # Reduce to smaller channel for SCNN

        # SCNN Layers
        self.scnn = nn.Sequential(
            SCNNLayer(128, 'down'),
            SCNNLayer(128, 'up'),
            SCNNLayer(128, 'right'),
            SCNNLayer(128, 'left'),
        )

        # Final prediction head
        self.classifier = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

        self.upsample = nn.Upsample(size=(config.image_height, config.image_width), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.backbone(x)         # ResNet feature map
        x = self.reduce(x)           # Reduce channels
        x = self.scnn(x)             # SCNN message passing
        x = self.classifier(x)       # Classifier head
        x = self.upsample(x)         # Restore to image size
        return x

