import argparse
import os
from utils import util
from dimred import DimRed

#py main.py --alg pca --datapath ./data/digits_data.csv --savepath ./saves/ --verbose 2


parser = argparse.ArgumentParser('argument for running dimred algorithms')

parser.add_argument("--alg", type=str, default="pca", help="name of selected dimred algorithm")
parser.add_argument("--datapath", type=str, help="path of input dataset")
parser.add_argument("--savepath", type=str, help="path to save files")
parser.add_argument("--verbose", type=int, default=1, help="print statement verbosity")

args = parser.parse_args()

if args.datapath is None or os.path.isfile(args.datapath) == False:
    raise AttributeError('Invalid input data path')
else:
    if os.path.isdir(args.savepath) == False:
        args.savepath = None
        print('File save path is not a valid path; saving is disabled.')

    obj = DimRed(args.alg, args.datapath, args.savepath, args.verbose)

    obj.data_proc()
    obj.dimred()
    # data, labels = util.default_input_proc(args.datapath, args.verbose)

    # algorithms = ['pca', 'mds', 'lda', 'le', 'tsne']

    # for a in algorithms:
    #     exec('from algorithms import {}'.format(a))

    # run = '{}.{}(data, labels, args.datapath, args.savepath, args.verbose)'.format(args.alg, args.alg.upper())
    # exec(run)