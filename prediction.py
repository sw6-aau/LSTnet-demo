import argparse
import math
import time

import torch
import torch.nn as nn
from models import LSTNet
import numpy as np;
import importlib

from utils import *;
import Optim

# See main.py for extra explanations
def evaluate(data, X, Y, model, batch_size):
    model.eval();
    predict = None;
    test = None;
    
    for X, Y in data.get_batches(X, Y, batch_size, False):
        output = model(X);
        if predict is None:
            predict = output;
            test = Y;
        else:
            predict = torch.cat((predict,output));
            test = torch.cat((test, Y));
        
        scale = data.scale.expand(output.size(0), data.m)
        
    # I'm thinking this really should be predict * scale instead
    # But it gives a tensor error, something about a tensor being too small
    return (output * scale)

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, required=True,
                    help='location of the data file')
parser.add_argument('--model', type=str, default='LSTNet',
                    help='')
parser.add_argument('--hidCNN', type=int, default=100,
                    help='number of CNN hidden units')
parser.add_argument('--hidRNN', type=int, default=100,
                    help='number of RNN hidden units')
parser.add_argument('--window', type=int, default=24 * 7,
                    help='window size')
parser.add_argument('--CNN_kernel', type=int, default=6,
                    help='the kernel size of the CNN layers')
parser.add_argument('--highway_window', type=int, default=24,
                    help='The window size of the highway component')
parser.add_argument('--clip', type=float, default=10.,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=54321,
                    help='random seed')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--cuda', type=str, default=True)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--horizon', type=int, default=12)
parser.add_argument('--skip', type=float, default=24)
parser.add_argument('--hidSkip', type=int, default=5)
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--output_fun', type=str, default='sigmoid')
args = parser.parse_args()
args.cuda = args.gpu is not None

#Model Loading
Data = Data_utility(args.data, 0.6, 0.2, args.cuda, args.horizon, args.window, args.normalize)
print(Data.rse)
model = eval(args.model).Model(args, Data)
optim = Optim.Optim(
    model.parameters(), args.optim, args.lr, args.clip,
)
with open(args.save, 'rb+') as f:
    checkpoint = torch.load(f)
model.load_state_dict(checkpoint['model_state_dict'])
optim.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.eval()

#Parameter setting
if args.cuda:
    torch.cuda.set_device(args.gpu)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)
if args.cuda:
    model.cuda()

result = evaluate(Data, Data.valid[0], Data.valid[1], model, args.batch_size)
print(result)