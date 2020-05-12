import argparse
import math
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models import AENet
import numpy as np;
import importlib

from utils import *;
import Optim
import torch.distributions as qwe

# Passes the data-set as input to the model in small batches
# After the data-set has been fully parsed, the output is compared to the original data-set.
def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()       # Sets the model to evaluation mode
    total_loss = 0;
    total_loss_l1 = 0;
    n_samples = 0;
    predict = None;
    test = None;
    
    # Iterates through all the batches as inputs and uses the latest model on it. 
    # Appends the batches of X and Y to big tensors, and finds the total loss according to both metrics, RSE and RAE.
    for X, Y in data.get_batches(X, Y, batch_size, False):
        output = model(X.float())
        if predict is None:
            predict = output;
            test = Y;
        else:
            predict = torch.cat((predict,output))
            test = torch.cat((test, Y))
        
        scale = data.scale.expand(output.size(0), data.m)
        total_loss += evaluateL2(output * scale, Y * scale).data    # .data??
        total_loss_l1 += evaluateL1(output * scale, Y * scale).data
        n_samples += (output.size(0) * data.m)
    
    rse = math.sqrt(total_loss / n_samples)/data.rse
    rae = (total_loss_l1/n_samples)/data.rae
    
    # Calculates correlation
    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis = 0)
    sigma_g = (Ytest).std(axis = 0)
    mean_p = predict.mean(axis = 0)
    mean_g = Ytest.mean(axis = 0)
    index = (sigma_g!=0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis = 0)/(sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return rse, rae, correlation;

def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()                  # Sets the model to training mode
    total_loss = 0;
    total_loss2 = 0;
    n_samples = 0;
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()          # Reset gradient'
        output = model(X.float())
        scale = data.scale.expand(output.size(0), data.m)  # Expand the original scale tensor to have row size matching the batch size.
        loss = criterion(output * scale, Y * scale)   # defines the loss / objective function, loss function arguments (input, target)
        loss.backward()                                # Computes the loss for every gradient / weight?
        optim.step()                       # Updates gradients https://discuss.pytorch.org/t/what-does-the-backward-function-do/9944
        total_loss += loss.data;                        # Adds the loss for this batch to the total loss
        n_samples += (output.size(0) * data.m)         # Increments the sample count with this sample size.
        #print(total_loss/n_samples)
    return total_loss / n_samples                       # Returns average loss of all samples
    
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
if args.cuda:
    torch.cuda.set_device(args.gpu)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

Data = Data_utility(args.data, 0.6, 0.2, args.cuda, args.horizon, args.window, args.normalize)
#print(Data.rse)

model = eval("AENet").Model(args, Data)
model.float()

if args.cuda:
    model.cuda()
    
nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

if args.L1Loss:
    criterion = nn.L1Loss(size_average=False)
else:
    criterion = nn.MSELoss(size_average=False)
evaluateL2 = nn.MSELoss(size_average=False)
evaluateL1 = nn.L1Loss(size_average=False)
if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda()
    evaluateL2 = evaluateL2.cuda()
    
    
best_val = 10000000
optim = Optim.Optim(
    model.parameters(), args.optim, args.lr, args.clip,
)

best_val2 = 10000000;
optim2 = Optim.Optim(
    model.parameters(), args.optim, args.lr, args.clip,
)

def add_noise(data):
        noise_factor = 0.5
        train_data = data.data.numpy()
        train_noisy = train_data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train_data.shape) 
        train_noisy = np.clip(train_noisy, 0., 1.)
        train_noisy_torch = torch.from_numpy(train_noisy)
        return train_noisy_torch

# At any point you can hit Ctrl + C to break out of training early.
try:
    print('begin training')
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_noisy = Data.train[0]
        valid_noisy = Data.valid[0]
        test_noisy = Data.test[0]
        
        train_loss = train(Data, train_noisy, Data.train[1], model, criterion, optim, args.batch_size) # Changes the gradients and finds loss, X, Y from bachify
        val_loss, val_rae, val_corr = evaluate(Data, valid_noisy, Data.valid[1], model, evaluateL2, evaluateL1, args.batch_size) # Evaluates loss according to RSE RAE CORR fomulars
        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))
        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val:
            with open(args.save, 'wb+') as f:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.optimizer.state_dict()
                }, f)
            best_val = val_loss
        if epoch % 5 == 0:
            test_acc, test_rae, test_corr  = evaluate(Data, test_noisy, Data.test[1], model, evaluateL2, evaluateL1, args.batch_size)
            print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb+') as f:
    checkpoint = torch.load(f)
model.load_state_dict(checkpoint['model_state_dict'])
optim.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
test_acc, test_rae, test_corr  = evaluate(Data, test_noisy, Data.test[1], model, evaluateL2, evaluateL1, args.batch_size)
print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))

