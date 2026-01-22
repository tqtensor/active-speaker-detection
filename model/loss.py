import torch
import torch.nn as nn
import torch.nn.functional as F


class lossAV(nn.Module):
    """Audio-visual loss module for active speaker detection.

    Combines audio and visual features to predict speaker activity using
    binary classification with cross-entropy loss.

    Attributes:
        criterion: Cross-entropy loss function for training.
        FC: Fully connected layer mapping 256-dimensional features to 2 classes.
    """

    def __init__(self):
        """Initializes the lossAV instance."""
        super(lossAV, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.FC = nn.Linear(256, 2)

    def forward(self, x, labels=None):
        """Computes loss and predictions for audio-visual features.

        Processes input features through a fully connected layer and computes
        either prediction scores (inference mode) or loss with accuracy metrics
        (training mode) based on whether labels are provided.

        Args:
            x: Input features of shape (batch_size, 1, 256).
            labels: Ground truth labels for training. If None, operates in
                inference mode.

        Returns:
            In inference mode (labels=None): Numpy array of prediction scores
                for the positive class.
            In training mode: Tuple containing (loss, prediction scores as
                softmax probabilities, predicted binary labels, count of
                correct predictions).
        """
        x = x.squeeze(1)
        x = self.FC(x)
        if labels is None:
            predScore = x[:, 1]
            predScore = predScore.t()
            predScore = predScore.view(-1).detach().cpu().numpy()
            return predScore
        else:
            nloss = self.criterion(x, labels)
            predScore = F.softmax(x, dim=-1)
            predLabel = torch.round(F.softmax(x, dim=-1))[:, 1]
            correctNum = (predLabel == labels).sum().float()
            return nloss, predScore, predLabel, correctNum


class lossA(nn.Module):
    """Audio-only loss module for speaker detection.

    Processes audio features for binary classification of speaker activity
    using cross-entropy loss.

    Attributes:
        criterion: Cross-entropy loss function for training.
        FC: Fully connected layer mapping 128-dimensional audio features to 2 classes.
    """

    def __init__(self):
        """Initializes the lossA instance."""
        super(lossA, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.FC = nn.Linear(128, 2)

    def forward(self, x, labels):
        """Computes cross-entropy loss for audio features.

        Args:
            x: Input audio features of shape (batch_size, 1, 128).
            labels: Ground truth labels for computing loss.

        Returns:
            Cross-entropy loss value.
        """
        x = x.squeeze(1)
        x = self.FC(x)
        nloss = self.criterion(x, labels)
        return nloss


class lossV(nn.Module):
    """Visual-only loss module for speaker detection.

    Processes visual features for binary classification of speaker activity
    using cross-entropy loss.

    Attributes:
        criterion: Cross-entropy loss function for training.
        FC: Fully connected layer mapping 128-dimensional visual features to 2 classes.
    """

    def __init__(self):
        """Initializes the lossV instance."""
        super(lossV, self).__init__()

        self.criterion = nn.CrossEntropyLoss()
        self.FC = nn.Linear(128, 2)

    def forward(self, x, labels):
        """Computes cross-entropy loss for visual features.

        Args:
            x: Input visual features of shape (batch_size, 1, 128).
            labels: Ground truth labels for computing loss.

        Returns:
            Cross-entropy loss value.
        """
        x = x.squeeze(1)
        x = self.FC(x)
        nloss = self.criterion(x, labels)
        return nloss
