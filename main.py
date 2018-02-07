import os
import sys
import argparse
from train import train
from test import test

def main(args):
    train(args)
    test(args)  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Person Re-Identification Reproduce')
    parser.add_argument('--params-filename', type=str, default='reid.pth.tar', help='filename of model parameters.')
    parser.add_argument('--use-gpu', type=int, default=1, help='set 1 if want to use GPU, otherwise 0. (default 1)')
    parser.add_argument('--world-size', type=int, default=1, help='number of distributed processes. (default 1)')
    parser.add_argument('--dist-url', type=str, default='tcp://127.0.0.1:2222', help='the master-node\'s address and port')
    parser.add_argument('--dist-rank', type=int, default=0, help='rank of distributed process. (default 0)')
    parser.add_argument('--last-conv', type=int, default=1, help='whether contains last convolution layter. (default 1)')
    parser.add_argument('--batch-size', type=int, default=64, help='training data batch size. (default 64)')
    parser.add_argument('--num-workers', type=int, default=20, help='number of workers when loading data. (default 20)')
    args = parser.parse_args()
    args.home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
    args.dataset = os.path.join(args.home, 'datasets')
    args.model_file = os.path.join(args.home, 'models', args.params_filename)
    args.distributed = args.world_size > 1
    main(args)