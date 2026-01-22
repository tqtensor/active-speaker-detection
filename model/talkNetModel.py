import torch
import torch.nn as nn

from model.attentionLayer import attentionLayer
from model.audioEncoder import audioEncoder
from model.visualEncoder import visualConv1D, visualFrontend, visualTCN


class talkNetModel(nn.Module):
    """A multi-modal neural network for active speaker detection.

    Combines visual and audio encoders with cross-attention and self-attention
    mechanisms to detect active speakers in video sequences.

    Attributes:
        visualFrontend: Visual feature extraction frontend network.
        visualTCN: Visual temporal convolutional network for sequential processing.
        visualConv1D: 1D convolutional layer for visual feature refinement.
        audioEncoder: ResNet-based audio encoder for audio feature extraction.
        crossA2V: Cross-attention layer from audio to visual features.
        crossV2A: Cross-attention layer from visual to audio features.
        selfAV: Self-attention layer for combined audio-visual features.
    """

    def __init__(self):
        """Initializes the talkNetModel instance.

        Sets up the visual frontend, temporal networks, audio encoder, and
        attention layers for audio-visual fusion.
        """
        super(talkNetModel, self).__init__()
        # Visual Temporal Encoder
        self.visualFrontend = visualFrontend()  # Visual Frontend
        # self.visualFrontend.load_state_dict(torch.load('visual_frontend.pt', map_location="cuda"))
        # for param in self.visualFrontend.parameters():
        #     param.requires_grad = False
        self.visualTCN = visualTCN()  # Visual Temporal Network TCN
        self.visualConv1D = visualConv1D()  # Visual Temporal Network Conv1d

        # Audio Temporal Encoder
        self.audioEncoder = audioEncoder(
            layers=[3, 4, 6, 3], num_filters=[16, 32, 64, 128]
        )

        # Audio-visual Cross Attention
        self.crossA2V = attentionLayer(d_model=128, nhead=8)
        self.crossV2A = attentionLayer(d_model=128, nhead=8)

        # Audio-visual Self Attention
        self.selfAV = attentionLayer(d_model=256, nhead=8)

    def forward_visual_frontend(self, x):
        """Processes visual input through the visual encoding pipeline.

        Normalizes input frames, extracts features using the visual frontend,
        and applies temporal convolution networks.

        Args:
            x: Input tensor of shape (B, T, W, H) where B is batch size, T is
                temporal frames, and W, H are frame dimensions.

        Returns:
            Encoded visual features of shape (B, T, 128).
        """
        B, T, W, H = x.shape
        x = x.view(B * T, 1, 1, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualFrontend(x)
        x = x.view(B, T, 512)
        x = x.transpose(1, 2)
        x = self.visualTCN(x)
        x = self.visualConv1D(x)
        x = x.transpose(1, 2)
        return x

    def forward_audio_frontend(self, x):
        """Processes audio input through the audio encoding pipeline.

        Reshapes input and applies the ResNet-based audio encoder to extract
        audio features.

        Args:
            x: Input audio tensor.

        Returns:
            Encoded audio features of shape (B, T, 128).
        """
        x = x.unsqueeze(1).transpose(2, 3)
        x = self.audioEncoder(x)
        return x

    def forward_cross_attention(self, x1, x2):
        """Applies bidirectional cross-attention between audio and visual features.

        Computes attention from audio to visual and from visual to audio
        to capture cross-modal interactions.

        Args:
            x1: Audio features.
            x2: Visual features.

        Returns:
            Tuple of (audio_attended, visual_attended) features after cross-attention.
        """
        x1_c = self.crossA2V(src=x1, tar=x2)
        x2_c = self.crossV2A(src=x2, tar=x1)
        return x1_c, x2_c

    def forward_audio_visual_backend(self, x1, x2):
        """Fuses audio and visual features using self-attention.

        Concatenates audio and visual features and applies self-attention
        to model their joint representation.

        Args:
            x1: Audio features after cross-attention.
            x2: Visual features after cross-attention.

        Returns:
            Fused audio-visual features of shape (B*T, 256).
        """
        x = torch.cat((x1, x2), 2)
        x = self.selfAV(src=x, tar=x)
        x = torch.reshape(x, (-1, 256))
        return x

    def forward_audio_backend(self, x):
        """Reshapes audio features for final processing.

        Args:
            x: Audio features.

        Returns:
            Reshaped audio features of shape (B*T, 128).
        """
        x = torch.reshape(x, (-1, 128))
        return x

    def forward_visual_backend(self, x):
        """Reshapes visual features for final processing.

        Args:
            x: Visual features.

        Returns:
            Reshaped visual features of shape (B*T, 128).
        """
        x = torch.reshape(x, (-1, 128))
        return x
