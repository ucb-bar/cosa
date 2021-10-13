#!/usr/bin/env python3 
import argparse
import pathlib
import copy
import itertools
import os
import subprocess
import re
import shutil

import utils
import run_arch

_COSA_DIR = os.environ['COSA_DIR']


def construct_argparser():

    parser = argparse.ArgumentParser(description='Run Configuration')
    parser.add_argument('-p',
                        '--dataset_path',
                        type=str,
                        help='Dataset Path',
                        )
    parser.add_argument('-n',
                        '--n_entries',
                        type=str,
                        help='Number of entries',
                        default=None,
                        )

    return parser

if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()

    dataset_path = pathlib.Path(args.dataset_path).resolve()
    run_arch.parse_best_results(dataset_path, n_entries=args.n_entries, obj='edp', func='min')
    run_arch.parse_best_results(dataset_path, n_entries=args.n_entries, obj='edp', func='mean')

