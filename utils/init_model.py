"""Write a randomly initialised checkpoint — the starting point of a zero-knowledge run.

Reinforcement learning from scratch has no supervised model to warm-start
from, but :mod:`pydlshogi2.selfplay` and ``rl_loop.sh`` both expect a
checkpoint.  This writes one in the same format :mod:`pydlshogi2.train` saves
(weights, a fresh optimizer state and the embedded architecture), so the loop
can resume from it like from any other checkpoint.

Example
-------

.. code-block:: bash

    python utils/init_model.py zero/checkpoint-000.pth --blocks 10 --channels 128
"""
import argparse
import os
import sys

import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydlshogi2.network.policy_value_resnet import build_network  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('output', help='checkpoint file to write')
    parser.add_argument('--blocks', type=int, default=10, help='number of residual blocks')
    parser.add_argument('--channels', type=int, default=128, help='channel width')
    parser.add_argument('--fcl', type=int, default=256, help='value head fully-connected size')
    parser.add_argument('--no_se', action='store_true', help='disable Squeeze-and-Excitation blocks')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    config = {'blocks': args.blocks, 'channels': args.channels,
              'fcl': args.fcl, 'se': not args.no_se}
    model = build_network(config)
    # train.py と同じオプティマイザ (resume で状態を読み込むため形を揃える)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({'epoch': 0, 't': 0, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'network': config}, args.output)
    params = sum(p.numel() for p in model.parameters())
    print('wrote {} ({} parameters, {})'.format(args.output, params, config))


if __name__ == '__main__':
    main()
