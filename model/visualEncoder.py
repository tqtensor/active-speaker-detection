##
# ResNet18 Pretrained network to extract lip embedding
# This code is modified based on https://github.com/lordmartian/deep_avsr
##

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetLayer(nn.Module):
    """A ResNet layer consisting of two residual blocks.

    This layer implements the building block for ResNet-18 architecture with
    the following structure::

        --> conv-bn-relu -> conv -> + -> bn-relu -> conv-bn-relu -> conv -> + -> bn-relu -->
         |                        |   |                                    |
         -----> downsample ------>    ------------------------------------->

    Args:
        inplanes: Number of input channels.
        outplanes: Number of output channels.
        stride: Stride for the first convolution and downsample operation.
    """

    def __init__(self, inplanes, outplanes, stride):
        super(ResNetLayer, self).__init__()
        self.conv1a = nn.Conv2d(
            inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(
            outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.stride = stride
        self.downsample = nn.Conv2d(
            inplanes, outplanes, kernel_size=(1, 1), stride=stride, bias=False
        )
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(
            outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(
            outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        return

    def forward(self, inputBatch):
        batch = F.relu(self.bn1a(self.conv1a(inputBatch)))
        batch = self.conv2a(batch)
        if self.stride == 1:
            residualBatch = inputBatch
        else:
            residualBatch = self.downsample(inputBatch)
        batch = batch + residualBatch
        intermediateBatch = batch
        batch = F.relu(self.outbna(batch))

        batch = F.relu(self.bn1b(self.conv1b(batch)))
        batch = self.conv2b(batch)
        residualBatch = intermediateBatch
        batch = batch + residualBatch
        outputBatch = F.relu(self.outbnb(batch))
        return outputBatch


class ResNet(nn.Module):
    """An 18-layer ResNet architecture for visual feature extraction.

    This network consists of 4 ResNet layers with increasing channel dimensions
    (64 -> 128 -> 256 -> 512) followed by average pooling.
    """

    def __init__(self):
        super(ResNet, self).__init__()
        self.layer1 = ResNetLayer(64, 64, stride=1)
        self.layer2 = ResNetLayer(64, 128, stride=2)
        self.layer3 = ResNetLayer(128, 256, stride=2)
        self.layer4 = ResNetLayer(256, 512, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(4, 4), stride=(1, 1))

        return

    def forward(self, inputBatch):
        batch = self.layer1(inputBatch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        batch = self.layer4(batch)
        outputBatch = self.avgpool(batch)
        return outputBatch


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization module.

    Applies layer normalization across channel and temporal dimensions with
    learnable scale (gamma) and shift (beta) parameters.

    Args:
        channel_size: Number of channels to normalize.
    """

    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # [M, 1, 1]
        var = (
            (torch.pow(y - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        )
        gLN_y = self.gamma * (y - mean) / torch.pow(var + 1e-8, 0.5) + self.beta
        return gLN_y


class visualFrontend(nn.Module):
    """Visual frontend for extracting features from video frames.

    Generates a 512-dimensional feature vector per video frame using a 3D
    convolution block followed by an 18-layer ResNet architecture.
    """

    def __init__(self):
        super(visualFrontend, self).__init__()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1,
                64,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(64, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.resnet = ResNet()
        return

    def forward(self, inputBatch):
        inputBatch = inputBatch.transpose(0, 1).transpose(1, 2)
        batchsize = inputBatch.shape[0]
        batch = self.frontend3D(inputBatch)

        batch = batch.transpose(1, 2)
        batch = batch.reshape(
            batch.shape[0] * batch.shape[1],
            batch.shape[2],
            batch.shape[3],
            batch.shape[4],
        )
        outputBatch = self.resnet(batch)
        outputBatch = outputBatch.reshape(batchsize, -1, 512)
        outputBatch = outputBatch.transpose(1, 2)
        outputBatch = outputBatch.transpose(1, 2).transpose(0, 1)
        return outputBatch


class DSConv1d(nn.Module):
    """Depthwise Separable 1D Convolution block with residual connection.

    Implements a sequence of ReLU, BatchNorm, depthwise convolution, PReLU,
    Global Layer Normalization, and pointwise convolution with a skip connection.
    """

    def __init__(self):
        super(DSConv1d, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(
                512, 512, 3, stride=1, padding=1, dilation=1, groups=512, bias=False
            ),
            nn.PReLU(),
            GlobalLayerNorm(512),
            nn.Conv1d(512, 512, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x)
        return out + x


class visualTCN(nn.Module):
    """Visual Temporal Convolutional Network (V-TCN).

    A stack of 5 Depthwise Separable 1D Convolution blocks for temporal
    modeling of visual features.
    """

    def __init__(self):
        super(visualTCN, self).__init__()
        stacks = []
        for x in range(5):
            stacks += [DSConv1d()]
        self.net = nn.Sequential(*stacks)  # Visual Temporal Network V-TCN

    def forward(self, x):
        out = self.net(x)
        return out


class visualConv1D(nn.Module):
    """1D Convolutional head for visual feature projection.

    Projects 512-dimensional visual features down to 128 dimensions through
    two convolutional layers (512 -> 256 -> 128).
    """

    def __init__(self):
        super(visualConv1D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(512, 256, 5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
        )

    def forward(self, x):
        out = self.net(x)
        return out
