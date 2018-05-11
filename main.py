import os
import sys
import argparse
from train import train
from test import test
import torch

def main(args):
    torch.manual_seed(960202)
    if args.stage == 'all' or args.stage == 'train':
        train(args)
    if args.stage == 'all' or args.stage == 'test':
        test(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Person Re-Identification Reproduce'
    )
    parser.add_argument('--params-filename',
        type=str, default='reid.pth.tar',
        help='filename of model parameters.'
    )
    parser.add_argument('--use-gpu',
        type=int, default=1,
        help='set 1 if want to use GPU, otherwise 0. (default 1)'
    )
    parser.add_argument('--world-size',
        type=int, default=1,
        help='number of distributed processes. (default 1)'
    )
    parser.add_argument('--dist-url',
        type=str, default='tcp://127.0.0.1:2222',
        help='the master-node\'s address and port'
    )
    parser.add_argument('--dist-rank',
        type=int, default=0,
        help='rank of distributed process. (default 0)'
    )
    parser.add_argument('--last-conv',
        type=int, default=1,
        help='whether contains last convolution layter. (default 1)'
    )
    parser.add_argument('--batch-size',
        type=int, default=64,
        help='training data batch size. (default 64)'
    )
    parser.add_argument('--num-workers',
        type=int, default=20,
        help='number of workers when loading data. (default 20)'
    )
    parser.add_argument('--load-once',
        type=int, default=0,
        help='load all of data at once. (default 0)'
    )
    parser.add_argument('--epoch',
        type=int, default=60,
        help="number of epochs. (default 60)"
    )
    parser.add_argument('--stage',
        type=str, default='train',
        help='running stage. train, test or all. (default train)'
    )
    parser.add_argument('--test-type',
        type=str, default='pcb',
        help='model type when testing. pcb, rpp or fnl. (default pcb)'
    )
    parser.add_argument('--rpp-std',
        type=float, default=0.01,
        help='standard deviation of initialization of rpp layer. (default 0.01)'
    )
    parser.add_argument('--conv-std',
        type=float, default=0.001,
        help='standard deviation of initialization of conv layer. (default 0.001)'
    )
    args = parser.parse_args()
    args.use_gpu = args.use_gpu == 1
    args.last_conv = args.last_conv == 1
    args.load_once = args.load_once == 1
    args.distributed = args.world_size > 1
    args.home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
    args.dataset = os.path.join(args.home, 'datasets')
    args.model_file = os.path.join(args.home, 'models', args.params_filename)
    main(args)