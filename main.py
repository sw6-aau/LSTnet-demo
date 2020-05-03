import argparse
import math
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models import LSTM, AELSTM, AENet, TAENet
import numpy as np;
import importlib

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from utils import *;
import Optim
import numpy as np

class Trainer:
    def __init__(self):
        # Initial setup with arguments
        self.parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
        self.set_args()
        self.enable_gpu()
        self.set_seed()

        self.Data = Data_utility(self.args.data, 0.6, 0.2, self.args.cuda, self.args.horizon, self.args.window, self.args.normalize);
        print(self.Data.rse);

        self.set_loss_functions()

        # Hyperopt configuration
        self.set_initial_values()
        search_space = self.create_spaces()
        self.trials_setup()

        # Tune each parameter one by one
        self.absolute_best = 10000000
        self.active = ''
        for x in range(0,len(search_space)):
            current_space = search_space[x]
            self.active = current_space

            best, trials = self.tune(current_space)
            print(best)

            self.manage_results(best, trials)
        self.print_results()

        
    
    #####################
    # HYPERTUNING LOGIC #
    #####################
    
    # Performs hyperopt's tuning process
    def tune(self, case):
        set_trials = Trials()
        best = fmin(
            self.tuned_train,
            space=case,
            algo=tpe.suggest,
            max_evals=self.args.evals,
            trials=set_trials
        )
        return best, set_trials

    # Tunes hyperparameters and trains the model
    # Adjust this function anytime a new hyperparameter is added
    def tuned_train(self, tuning):
        self.active_parameter(tuning)
            
        print(self.args.model)
        # Prepares the model for training
        if self.args.model == 'AENet':
            model = eval(self.args.model).Model(self.args, self.Data, self.cnn);
        else:
            model = eval(self.args.model).Model(self.args, self.Data, self.cnn, self.rnn, self.skip, self.activation);

        nParams = sum([p.nelement() for p in model.parameters()])
        print('* number of parameters: %d' % nParams)

        if self.args.cuda:
            model.cuda()
            
        optim = Optim.Optim(
            model.parameters(), self.args.optim, self.lr, self.args.clip
        )

        best_val = 10000000;
        # Performs training for a given hypertuning iteration
        for epoch in range(1, self.hyper_epoch + 1):
            epoch_start_time = time.time()
            train_loss = self.train(self.Data, self.Data.train[0], self.Data.train[1], model, self.criterion, optim, self.args.batch_size)
            print(train_loss)
            val_loss, val_rae, val_corr = self.evaluate(self.Data, self.Data.valid[0], self.Data.valid[1], model, self.evaluateL2, self.evaluateL1, self.args.batch_size);
            print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))
            
            # Save the model if the validation loss is the best we've seen so far.
            if train_loss < best_val:
                with open(self.args.save, 'wb+') as f:
                    torch.save(model, f)
                best_val = train_loss
            if epoch % 5 == 0:
                test_acc, test_rae, test_corr  = self.evaluate(self.Data, self.Data.test[0], self.Data.test[1], model, self.evaluateL2, self.evaluateL1, self.args.batch_size);
                print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))
        
        # Tests best saved model on the test data
        with open(self.args.save, 'rb+') as f:
            model = torch.load(f)
        test_acc, test_rae, test_corr  = self.evaluate(self.Data, self.Data.test[0], self.Data.test[1], model, self.evaluateL2, self.evaluateL1, self.args.batch_size);
        print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))

        ''' This shit meant for saving the absolute best model, but doesn't quite work for reasons I cannot fathom
        if test_acc < self.absolute_best:
            with open(self.args.bestsave, 'wb+') as f:
                torch.save(model, f)
            self.absolute_best = test_acc
        '''
        
        return {'loss': test_acc, 'status': STATUS_OK}



    ##########################
    # NETWORK TRAINING LOGIC #
    ##########################

    def train(self, data, X, Y, model, criterion, optim, batch_size):
        model.train();
        total_loss = 0;
        n_samples = 0;
        for X, Y in data.get_batches(X, Y, batch_size, True):
            model.zero_grad();
            output = model(X);
            scale = data.scale.expand(output.size(0), data.m)
            loss = criterion(output * scale, Y * scale);
            loss.backward();
            grad_norm = optim.step();
            total_loss += loss.data;
            n_samples += (output.size(0) * data.m);
        return total_loss / n_samples

    def evaluate(self, data, X, Y, model, evaluateL2, evaluateL1, batch_size):
        model.eval();
        total_loss = 0;
        total_loss_l1 = 0;
        n_samples = 0;
        predict = None;
        test = None;
        
        # Iterates through all the batches as inputs.
        for X, Y in data.get_batches(X, Y, batch_size, False):
            output = model(X);
            if predict is None:
                predict = output;
                test = Y;
            else:
                predict = torch.cat((predict,output));
                test = torch.cat((test, Y));
            
            # Loss calculation
            scale = data.scale.expand(output.size(0), data.m)
            total_loss += evaluateL2(output * scale, Y * scale).data
            total_loss_l1 += evaluateL1(output * scale, Y * scale).data
            n_samples += (output.size(0) * data.m);
        
        rse = math.sqrt(total_loss / n_samples)/data.rse
        rae = (total_loss_l1/n_samples)/data.rae
        
        # Calculates correlation
        predict = predict.data.cpu().numpy();
        Ytest = test.data.cpu().numpy();
        sigma_p = (predict).std(axis = 0);
        sigma_g = (Ytest).std(axis = 0);
        mean_p = predict.mean(axis = 0)
        mean_g = Ytest.mean(axis = 0)
        index = (sigma_g!=0);
        correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis = 0)/(sigma_p * sigma_g);
        correlation = (correlation[index]).mean();
        return rse, rae, correlation;



    ###################
    # SETUP-FUNCTIONS #
    ###################

    def enable_gpu(self):
        self.args = self.parser.parse_args()
        self.args.cuda = self.args.gpu is not None
        if self.args.cuda:
            torch.cuda.set_device(self.args.gpu)

    # Set the random seed manually for reproducibility.
    def set_seed(self):
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            if not self.args.cuda:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            else:
                torch.cuda.manual_seed(self.args.seed)

    def set_loss_functions(self):
        if self.args.L1Loss:
            self.criterion = nn.L1Loss(size_average=False);
        else:
            self.criterion = nn.MSELoss(size_average=False);
        self.evaluateL2 = nn.MSELoss(size_average=False);
        self.evaluateL1 = nn.L1Loss(size_average=False)
        if self.args.cuda:
            self.criterion = self.criterion.cuda()
            self.evaluateL1 = self.evaluateL1.cuda();
            self.evaluateL2 = self.evaluateL2.cuda();

    # Defines all arguments that are given in shell scripts and the like.
    def set_args(self):
        self.parser.add_argument('--data', type=str, required=True,
                            help='location of the data file')
        self.parser.add_argument('--model', type=str, default='LSTM',
                            help='')
        self.parser.add_argument('--hidCNN', type=int, default=100,
                            help='number of CNN hidden units')
        self.parser.add_argument('--hidRNN', type=int, default=100,
                            help='number of RNN hidden units')
        self.parser.add_argument('--window', type=int, default=24 * 7,
                            help='window size')
        self.parser.add_argument('--CNN_kernel', type=int, default=6,
                            help='the kernel size of the CNN layers')
        self.parser.add_argument('--highway_window', type=int, default=24,
                            help='The window size of the highway component')
        self.parser.add_argument('--clip', type=float, default=10.,
                            help='gradient clipping')
        self.parser.add_argument('--epochs', type=int, default=1,
                            help='upper epoch limit')
        self.parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                            help='batch size')
        self.parser.add_argument('--dropout', type=float, default=0.2,
                            help='dropout applied to layers (0 = no dropout)')
        self.parser.add_argument('--seed', type=int, default=54321,
                            help='random seed')
        self.parser.add_argument('--gpu', type=int, default=None)
        self.parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                            help='report interval')
        self.parser.add_argument('--save', type=str,  default='model/model.pt',
                            help='path to save a temporary model')
        self.parser.add_argument('--bestsave', type=str, default='model/model.pt')
        self.parser.add_argument('--cuda', type=str, default=True)
        self.parser.add_argument('--optim', type=str, default='adam')
        self.parser.add_argument('--lr', type=float, default=0.001)
        self.parser.add_argument('--horizon', type=int, default=12)
        self.parser.add_argument('--skip', type=float, default=24)
        self.parser.add_argument('--hidSkip', type=int, default=5)
        self.parser.add_argument('--L1Loss', type=bool, default=True)
        self.parser.add_argument('--normalize', type=int, default=2)
        self.parser.add_argument('--output_fun', type=str, default='sigmoid')
        self.parser.add_argument('--evals', type=int, default=5)



    ##########################
    # HYPEROPT CONFIGURATION #
    ##########################

    # Selects which set of spaces to use
    # Spaces are defined under SPACE FUNCTIONS
    def create_spaces(self):
        print('In create_spaces, model: ' + self.args.model)
        if self.args.model == 'AENet' or self.args.model == 'TAENet':
            return self.AENet_spaces()
        else:
            return self.standard_spaces()

    # Initializes values from args
    # See set_args() for a list of accepted args
    # Update this function and all functions below if you add/remove parameters
    def set_initial_values(self):
        self.hyper_epoch = self.args.epochs
        self.cnn = self.args.hidCNN
        self.rnn = self.args.hidRNN
        self.skip = self.args.hidSkip
        self.activation = self.args.output_fun
        self.lr = self.args.lr

    # Adjusts the value of the parameter that is currently being tuned
    def active_parameter(self, tuning):
        if self.active == self.case_epoch:
            self.hyper_epoch = int(tuning)
        elif self.active == self.case_cnn:
            self.cnn = int(tuning)
        elif self.active == self.case_rnn:
            self.rnn = int(tuning)
        elif self.active == self.case_skip:
            self.skip = int(tuning)
        elif self.active == self.case_activation:
            self.activation = tuning['type']
        elif self.active == self.case_lr:
            self.lr = tuning

    # Sets up trials for end-of-optimization reviewing
    def trials_setup(self):
        self.epochtrials = Trials()
        self.cnntrials = Trials()
        self.rnntrials = Trials()
        self.skiptrials = Trials()
        self.actitrials = Trials()
        self.lrtrials = Trials()

    # Sets up variables for later use in displaying results
    def manage_results(self, best, trials):
        if self.active == self.case_epoch:
            self.hyper_epoch = int(best['epoch'])
            self.epochtrials = trials
        elif self.active == self.case_cnn:
            self.cnn = int(best['cnn'])
            self.cnntrials = trials
        elif self.active == self.case_rnn:
            self.rnn = int(best['rnn'])
            self.rnntrials = trials
        elif self.active == self.case_skip:
            self.skip = int(best['skip'])
            self.skiptrials = trials
        elif self.active == self.case_activation:
            self.activation = best
            self.actitrials = trials
        elif self.active == self.case_lr:
            self.lr = best
            self.lrtrials = trials
            
    # Prints results of each parameter at the end of tuning
    # If you add new parameters, remember to update this
    def print_results(self):
        print('Model: ' + self.args.model)
        print('Best epoch: ' + str(self.hyper_epoch))
        print(self.epochtrials.trials)
        print('Best cnn: ' + str(self.cnn))
        print(self.cnntrials.trials)
        print('Best rnn: ' + str(self.rnn))
        print(self.rnntrials.trials)
        print('Best skip: ' + str(self.skip))
        print(self.skiptrials.trials)
        print('Best activator: ' + str(self.activation))
        print(self.actitrials.trials)
        print('Best Learning Rate: ' + str(self.lr))
        print(self.lrtrials.trials)


    
    ###################
    # SPACE FUNCTIONS #
    ###################
    
    def standard_spaces(self):
        print('Creating standard_spaces')
        self.case_epoch = hp.uniform('epoch', 10, 250)
        self.case_lr = hp.uniform('lr', 0.0001, 0.01)
        self.case_cnn = hp.uniform('cnn', 30, 800)
        self.case_rnn = hp.uniform('rnn', 30, 800)
        self.case_skip = hp.uniform('skip', 2, 10)
        self.case_activation = hp.choice('activation_type', [
            {
                'type': 'None',
            },
            {
                'type': 'sigmoid',
            },
            {
                'type': 'tanh',
            },
            {
                'type': 'relu',
            },
        ])
        return [self.case_epoch, self.case_lr, self.case_cnn, self.case_rnn, self.case_skip, self.case_activation] # Adjust this to change the order in which parameters are tuned
    
    def AENet_spaces(self):
        print('Creating AENet_spaces')
        self.case_epoch = hp.uniform('epoch', 10, 250)
        self.case_cnn = hp.uniform('cnn', 30, 800)
        return [self.case_epoch, self.case_cnn] # Adjust this to change the order in which parameters are tuned

    ################
    # END OF CLASS #
    ################

trainer = Trainer()