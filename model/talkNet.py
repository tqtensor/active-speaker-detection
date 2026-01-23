import subprocess
import sys
import time

import pandas
import torch
import torch.nn as nn
import tqdm

from config.logging_config import get_logger
from model.talkNetModel import talkNetModel

from .loss import lossA, lossAV, lossV

logger = get_logger(__name__)


class talkNet(nn.Module):
    """Audio-visual active speaker detection network.

    Implements the TalkNet model for detecting active speakers using both
    audio and visual features with cross-attention mechanisms.

    Attributes:
        model: The TalkNet model architecture.
        lossAV: Audio-visual loss function.
        lossA: Audio-only loss function.
        lossV: Visual-only loss function.
        optim: Adam optimizer for training.
        scheduler: Learning rate scheduler with step decay.
    """

    def __init__(self, lr=0.0001, lrDecay=0.95, **kwargs):
        """Initializes the TalkNet network.

        Args:
            lr: Learning rate for the optimizer.
            lrDecay: Learning rate decay factor per epoch.
            **kwargs: Additional keyword arguments.
        """
        super(talkNet, self).__init__()
        self.model = talkNetModel().cuda()
        self.lossAV = lossAV().cuda()
        self.lossA = lossA().cuda()
        self.lossV = lossV().cuda()
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim, step_size=1, gamma=lrDecay
        )
        model_params_mb = (
            sum(param.numel() for param in self.model.parameters()) / 1024 / 1024
        )
        logger.info(f"Model para number = {model_params_mb:.2f}")

    def train_network(self, loader, epoch, **kwargs):
        """Trains the network for one epoch.

        Args:
            loader: DataLoader providing training batches of (audio, visual, labels).
            epoch: Current epoch number.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of (average loss, learning rate) for the epoch.
        """
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]["lr"]
        for num, (audioFeature, visualFeature, labels) in enumerate(loader, start=1):
            self.zero_grad()
            audioEmbed = self.model.forward_audio_frontend(
                audioFeature[0].cuda()
            )  # feedForward
            visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())
            audioEmbed, visualEmbed = self.model.forward_cross_attention(
                audioEmbed, visualEmbed
            )
            outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
            outsA = self.model.forward_audio_backend(audioEmbed)
            outsV = self.model.forward_visual_backend(visualEmbed)
            labels = labels[0].reshape((-1)).cuda()  # Loss
            nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels)
            nlossA = self.lossA.forward(outsA, labels)
            nlossV = self.lossV.forward(outsV, labels)
            nloss = nlossAV + 0.4 * nlossA + 0.4 * nlossV
            loss += nloss.detach().cpu().numpy()
            top1 += prec
            nloss.backward()
            self.optim.step()
            index += len(labels)
            sys.stderr.write(
                time.strftime("%m-%d %H:%M:%S")
                + " [%2d] Lr: %5f, Training: %.2f%%, "
                % (epoch, lr, 100 * (num / loader.__len__()))
                + " Loss: %.5f, ACC: %2.2f%% \r" % (loss / (num), 100 * (top1 / index))
            )
            sys.stderr.flush()
        sys.stdout.write("\n")
        return loss / num, lr

    def evaluate_network(self, loader, evalCsvSave, evalOrig, **kwargs):
        """Evaluates the network on a validation/test set.

        Args:
            loader: DataLoader providing evaluation batches of (audio, visual, labels).
            evalCsvSave: Path to save the evaluation results CSV.
            evalOrig: Path to the original evaluation CSV with ground truth.
            **kwargs: Additional keyword arguments.

        Returns:
            Mean Average Precision (mAP) score for active speaker detection.
        """
        self.eval()
        predScores = []
        for audioFeature, visualFeature, labels in tqdm.tqdm(loader):
            with torch.no_grad():
                audioEmbed = self.model.forward_audio_frontend(audioFeature[0].cuda())
                visualEmbed = self.model.forward_visual_frontend(
                    visualFeature[0].cuda()
                )
                audioEmbed, visualEmbed = self.model.forward_cross_attention(
                    audioEmbed, visualEmbed
                )
                outsAV = self.model.forward_audio_visual_backend(
                    audioEmbed, visualEmbed
                )
                labels = labels[0].reshape((-1)).cuda()
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels)
                predScore = predScore[:, 1].detach().cpu().numpy()
                predScores.extend(predScore)
        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = []
        labels = pandas.Series(["SPEAKING_AUDIBLE" for line in evalLines])
        scores = pandas.Series(predScores)
        evalRes = pandas.read_csv(evalOrig)
        evalRes["score"] = scores
        evalRes["label"] = labels
        evalRes.drop(["label_id"], axis=1, inplace=True)
        evalRes.drop(["instance_id"], axis=1, inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s " % (
            evalOrig,
            evalCsvSave,
        )
        mAP = float(
            str(subprocess.run(cmd, shell=True, capture_output=True).stdout).split(" ")[
                2
            ][:5]
        )
        return mAP

    def saveParameters(self, path):
        """Saves the model parameters to disk.

        Args:
            path: File path where model state dict will be saved.
        """
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        """Loads model parameters from disk.

        Args:
            path: File path to the saved model state dict.
        """
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    logger.warning(f"{origName} is not in the model.")
                    continue
            if selfState[name].size() != loadedState[origName].size():
                logger.warning(
                    f"Wrong parameter length: {origName}, model: {selfState[name].size()}, loaded: {loadedState[origName].size()}"
                )
                continue
            selfState[name].copy_(param)
