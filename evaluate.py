import sys
sys.path.append('thirdparty/vidar/')
import os

import torch
from evaluation_utils.r3d3_argparser import argparser
from evaluation_utils.eval_dataset import Evaluator


if __name__ == '__main__':
    os.environ['DIST_MODE'] = 'gpu' if torch.cuda.is_available() else 'cpu'
    args = argparser()

    wrapper = Evaluator(args)
    wrapper.eval_datasets()
