"""Measure how often a model's top policy move matches the human move.

For human imitation the policy move-match rate on a band's test set is the
metric that matters. Evaluate several checkpoints on several bands in one go,
e.g. a base model against its per-rank fine-tunes:

.. code-block:: bash

    python human/eval_policy.py --models base.pth model-0031-0031-001.pth \\
        --tests 0031-0031/test.hcpe 0036-0036/test.hcpe --max_positions 50000

Prints one TSV row per (model, test): positions, policy top-1 accuracy,
policy loss, value accuracy.
"""
import argparse
import os

import numpy as np
import torch

from pydlshogi2.dataloader import HcpeDataLoader
from pydlshogi2.network.policy_value_resnet import build_network, LEGACY_NETWORK_CONFIG


def load_model(path, device):
    """Build the network recorded in a checkpoint and load its weights."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_network(ckpt.get('network', LEGACY_NETWORK_CONFIG))
    model.load_state_dict(ckpt['model'])
    return model.to(device).eval()


def evaluate(model, loader, amp_dtype):
    """Return ``(positions, policy_acc, policy_loss, value_acc)`` over ``loader``."""
    ce = torch.nn.CrossEntropyLoss(reduction='sum')
    n = hits = value_hits = 0
    loss = 0.0
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=amp_dtype,
                                         enabled=amp_dtype is not None):
        for x, move_label, result in loader:
            y1, y2 = model(x)
            n += len(move_label)
            hits += (y1.argmax(dim=1) == move_label).sum().item()
            loss += ce(y1.float(), move_label).item()
            value_hits += ((y2 >= 0) == (result >= 0.5)).sum().item()
    return n, hits / n, loss / n, value_hits / n


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', nargs='+', required=True, help='checkpoint files')
    parser.add_argument('--tests', nargs='+', required=True, help='test .hcpe files')
    parser.add_argument('--max_positions', type=int, default=0,
                        help='evaluate on a fixed random subset of each test set (0 = all)')
    parser.add_argument('--batchsize', type=int, default=1024)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--amp_dtype', choices=['bfloat16', 'float16', 'none'], default='bfloat16')
    args = parser.parse_args()

    device = torch.device('cuda:{}'.format(args.gpu) if args.gpu >= 0 else 'cpu')
    amp_dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'none': None}[args.amp_dtype]
    if device.type != 'cuda':
        amp_dtype = None

    loaders = []
    for path in args.tests:
        loader = HcpeDataLoader(path, args.batchsize, device)
        if args.max_positions and len(loader.data) > args.max_positions:
            # 毎回同じ部分集合で比べられるよう固定シードで抜き出す
            idx = np.random.default_rng(0).choice(len(loader.data), args.max_positions, replace=False)
            loader.data = loader.data[np.sort(idx)]
        loaders.append((path, loader))

    print('model\ttest\tpositions\tpolicy_acc\tpolicy_loss\tvalue_acc')
    for mpath in args.models:
        model = load_model(mpath, device)
        for tpath, loader in loaders:
            n, acc, loss, vacc = evaluate(model, loader, amp_dtype)
            print('{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}'.format(
                os.path.basename(mpath), tpath, n, acc, loss, vacc), flush=True)


if __name__ == '__main__':
    main()
