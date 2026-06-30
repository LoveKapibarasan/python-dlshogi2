"""Policy-value ResNet used by both the trainer and the players.

The network maps a stack of board feature planes to a policy distribution over
moves and a scalar value (win probability). The architecture is configurable:
the number of residual blocks, channel width, value fully-connected size and
the presence of Squeeze-and-Excitation (SE) modules can all be set at
construction time.

Checkpoints saved by :mod:`pydlshogi2.train` embed their architecture
configuration, so :func:`load_network` can reconstruct the exact network even
when the class defaults change. Legacy checkpoints that predate the embedded
config are loaded with the original ``10 x 192`` SE-free defaults.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from pydlshogi2.features import FEATURES_NUM, MOVE_PLANES_NUM, MOVE_LABELS_NUM

#: Architecture defaults for checkpoints that predate embedded config metadata.
LEGACY_NETWORK_CONFIG = {'blocks': 10, 'channels': 192, 'fcl': 256, 'se': False}


class Bias(nn.Module):
    """Adds a learnable per-element bias to its input."""

    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, input):
        return input + self.bias


class SqueezeExcitation(nn.Module):
    """Channel-wise Squeeze-and-Excitation gate.

    Global-average-pools the spatial dimensions, learns per-channel gates
    through a bottleneck MLP, and rescales the input channels. Cheap to compute
    and a consistent small strength gain in AlphaZero-style networks.

    :param channels: number of input/output channels.
    :param reduction: bottleneck reduction ratio.
    """

    def __init__(self, channels, reduction=8):
        super(SqueezeExcitation, self).__init__()
        bottleneck = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, bottleneck)
        self.fc2 = nn.Linear(bottleneck, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))            # squeeze: global average pool
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))    # excitation: per-channel gates
        return x * s.view(b, c, 1, 1)


class ResNetBlock(nn.Module):
    """A pre-activation-free residual block with optional SE gating.

    :param channels: channel width of the block.
    :param se: whether to apply a :class:`SqueezeExcitation` gate before the
        residual addition.
    """

    def __init__(self, channels, se=False):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels) if se else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.se is not None:
            out = self.se(out)

        return F.relu(out + x)


class PolicyValueNetwork(nn.Module):
    """ResNet backbone with separate policy and value heads.

    :param blocks: number of residual blocks.
    :param channels: channel width of the backbone.
    :param fcl: hidden size of the value head's fully-connected layer.
    :param se: enable Squeeze-and-Excitation in every residual block.
    """

    def __init__(self, blocks=20, channels=256, fcl=256, se=True):
        super(PolicyValueNetwork, self).__init__()
        # 構成を記録しておき、チェックポイントに埋め込めるようにする
        self.config = {'blocks': blocks, 'channels': channels, 'fcl': fcl, 'se': se}

        self.conv1 = nn.Conv2d(in_channels=FEATURES_NUM, out_channels=channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)

        # resnet blocks
        self.blocks = nn.Sequential(*[ResNetBlock(channels, se=se) for _ in range(blocks)])

        # policy head
        self.policy_conv = nn.Conv2d(in_channels=channels, out_channels=MOVE_PLANES_NUM, kernel_size=1, bias=False)
        self.policy_bias = Bias(MOVE_LABELS_NUM)

        # value head
        self.value_conv1 = nn.Conv2d(in_channels=channels, out_channels=MOVE_PLANES_NUM, kernel_size=1, bias=False)
        self.value_norm1 = nn.BatchNorm2d(MOVE_PLANES_NUM)
        self.value_fc1 = nn.Linear(MOVE_LABELS_NUM, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)

    def forward(self, x):
        """Run a forward pass.

        :param x: input feature tensor of shape ``(batch, FEATURES_NUM, 9, 9)``.
        :returns: a ``(policy_logits, value_logit)`` tuple. The value is a raw
            logit; apply ``sigmoid`` to obtain a win probability.
        """
        x = self.conv1(x)
        x = F.relu(self.norm1(x))

        # resnet blocks
        x = self.blocks(x)

        # policy head
        policy = self.policy_conv(x)
        policy = self.policy_bias(torch.flatten(policy, 1))

        # value head
        value = F.relu(self.value_norm1(self.value_conv1(x)))
        value = F.relu(self.value_fc1(torch.flatten(value, 1)))
        value = self.value_fc2(value)

        return policy, value


def build_network(config=None, **overrides):
    """Construct a :class:`PolicyValueNetwork` from a config dict.

    :param config: dict of architecture hyper-parameters (``blocks``,
        ``channels``, ``fcl``, ``se``). ``None`` uses the class defaults.
    :param overrides: individual hyper-parameters overriding ``config``.
    :returns: an un-trained :class:`PolicyValueNetwork`.
    """
    cfg = dict(config) if config else {}
    cfg.update(overrides)
    return PolicyValueNetwork(**cfg)


def load_network(modelfile, device):
    """Load a checkpoint and reconstruct its network.

    The architecture is taken from the checkpoint's embedded ``'network'``
    config when present; checkpoints without it (saved before this feature
    existed) fall back to :data:`LEGACY_NETWORK_CONFIG`.

    :param modelfile: path to a ``.pth`` checkpoint.
    :param device: torch device to map the weights onto.
    :returns: a ``(model, checkpoint)`` tuple. The model is on ``device`` and in
        ``eval`` mode is *not* set (caller decides).
    """
    checkpoint = torch.load(modelfile, map_location=device)
    config = checkpoint.get('network', LEGACY_NETWORK_CONFIG)
    model = build_network(config)
    model.to(device)
    model.load_state_dict(checkpoint['model'])
    return model, checkpoint
