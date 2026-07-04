import torch
import torch.nn as nn

from config.logging_config import get_logger

from .loss import lossAV, lossV
from .Model import ASD_Model

logger = get_logger(__name__)


class lightASD(nn.Module):
    """Inference wrapper for the Light-ASD model.

    Mirrors the interface used at the pipeline's inference call site: builds
    the model and loss heads on CUDA, exposes ``loadParameters`` for the
    upstream checkpoint format, and delegates forward passes to ``self.model``.
    """

    def __init__(self, **kwargs):
        super(lightASD, self).__init__()
        self.model = ASD_Model().cuda()
        self.lossAV = lossAV().cuda()
        self.lossV = lossV().cuda()
        n = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info(f"Light-ASD para number = {n:.2f}M")

    def loadParameters(self, path):
        """Loads upstream Light-ASD weights, tolerating a 'module.' prefix.

        Args:
            path: Path to a Light-ASD .model checkpoint.
        """
        selfState = self.state_dict()
        loadedState = torch.load(path, map_location="cpu")
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    logger.warning(f"{origName} is not in the model.")
                    continue
            if selfState[name].size() != loadedState[origName].size():
                logger.warning(
                    f"Wrong parameter length: {origName}, model: "
                    f"{selfState[name].size()}, loaded: {loadedState[origName].size()}"
                )
                continue
            selfState[name].copy_(param)
