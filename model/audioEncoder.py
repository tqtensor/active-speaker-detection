import torch
import torch.nn as nn


class SEBasicBlock(nn.Module):
    """A residual block with Squeeze-and-Excitation module.

    Implements a basic residual block enhanced with channel-wise attention
    through the Squeeze-and-Excitation (SE) mechanism for improved feature
    representation.

    Attributes:
        expansion: Expansion factor for the output channels.
        conv1: First convolutional layer.
        bn1: Batch normalization after first convolution.
        conv2: Second convolutional layer.
        bn2: Batch normalization after second convolution.
        relu: ReLU activation function.
        se: Squeeze-and-Excitation layer for channel attention.
        downsample: Optional downsampling layer for residual connection.
        stride: Stride value for the convolutional layers.
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        """Initializes the SEBasicBlock.

        Args:
            inplanes: Number of input channels.
            planes: Number of output channels for the convolutional layers.
            stride: Stride for the first convolution. Defaults to 1.
            downsample: Optional downsampling module for the residual path.
            reduction: Reduction ratio for the SE layer. Defaults to 8.
        """
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Performs forward pass through the SE basic block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying residual connection with SE attention.
        """
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class SELayer(nn.Module):
    """A Squeeze-and-Excitation layer for channel-wise attention.

    Implements the SE mechanism to recalibrate channel-wise feature responses
    by explicitly modeling interdependencies between channels.

    Attributes:
        avg_pool: Adaptive average pooling layer for global spatial information.
        fc: Fully connected layers for learning channel attention weights.
    """

    def __init__(self, channel, reduction=8):
        """Initializes the SELayer.

        Args:
            channel: Number of input channels.
            reduction: Reduction ratio for the bottleneck in FC layers. Defaults to 8.
        """
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Performs forward pass through the SE layer.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Recalibrated tensor with channel-wise attention applied.
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class audioEncoder(nn.Module):
    """An audio encoder network with SE-ResNet architecture.

    Implements a convolutional neural network for audio feature extraction
    using residual blocks with Squeeze-and-Excitation modules. The architecture
    processes audio spectrograms and outputs temporal features.

    Attributes:
        inplanes: Current number of input channels for layer construction.
        conv1: Initial convolutional layer.
        bn1: Batch normalization after initial convolution.
        relu: ReLU activation function.
        layer1: First residual layer.
        layer2: Second residual layer.
        layer3: Third residual layer.
        layer4: Fourth residual layer.
    """

    def __init__(self, layers, num_filters, **kwargs):
        """Initializes the audioEncoder.

        Args:
            layers: List of integers specifying the number of blocks in each layer.
            num_filters: List of integers specifying output channels for each layer.
            **kwargs: Additional keyword arguments (unused).
        """
        super(audioEncoder, self).__init__()
        block = SEBasicBlock
        self.inplanes = num_filters[0]

        self.conv1 = nn.Conv2d(
            1, num_filters[0], kernel_size=7, stride=(2, 1), padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """Constructs a residual layer with specified number of blocks.

        Args:
            block: The block class to use (e.g., SEBasicBlock).
            planes: Number of output channels for the blocks.
            blocks: Number of blocks in this layer.
            stride: Stride for the first block. Defaults to 1.

        Returns:
            Sequential module containing the residual blocks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Performs forward pass through the audio encoder.

        Args:
            x: Input tensor of shape (batch, 1, freq_bins, time_steps).

        Returns:
            Encoded features of shape (batch, time_steps, channels) suitable
            for temporal modeling.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.mean(x, dim=2, keepdim=True)
        x = x.view((x.size()[0], x.size()[1], -1))
        x = x.transpose(1, 2)

        return x
