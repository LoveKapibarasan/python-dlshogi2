"""Merge and de-duplicate HCPE training files.

Concatenates any number of ``.hcpe`` inputs and removes duplicate positions
(records sharing the same ``hcp`` bytes), keeping the first occurrence. Useful
for cleaning self-play output, where the opening positions recur across games
and would otherwise be heavily over-represented.

Example
-------

.. code-block:: bash

    python utils/hcpe_dedup.py merged.hcpe rl/selfplay-*.hcpe
"""
import argparse

import numpy as np
from cshogi import HuffmanCodedPosAndEval


def dedup(input_paths, output_path):
    """Merge inputs, drop duplicate positions and write the result.

    :param input_paths: list of input ``.hcpe`` file paths.
    :param output_path: destination ``.hcpe`` path.
    :returns: a ``(total, unique)`` tuple of record counts.
    """
    chunks = [np.fromfile(path, dtype=HuffmanCodedPosAndEval) for path in input_paths]
    data = np.concatenate(chunks) if chunks else np.empty(0, HuffmanCodedPosAndEval)
    total = len(data)

    # hcpバイト列をキーに最初の出現のみ残す
    hcp_view = data['hcp'].reshape(total, -1)
    _, first_index = np.unique(hcp_view, axis=0, return_index=True)
    unique_data = data[np.sort(first_index)]

    unique_data.tofile(output_path)
    return total, len(unique_data)


def main():
    """Parse arguments and run :func:`dedup`."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output', help='output (merged, de-duplicated) HCPE file')
    parser.add_argument('inputs', nargs='+', help='input HCPE files')
    args = parser.parse_args()

    total, unique = dedup(args.inputs, args.output)
    print('total={} unique={} removed={}'.format(total, unique, total - unique))


if __name__ == '__main__':
    main()
