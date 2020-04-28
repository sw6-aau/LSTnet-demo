import torch
import numpy as np;
from torch.autograd import Variable

#import sys
#import numpy

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, cuda, horizon, window, normalize = 2):
        self.cuda = cuda;
        self.P = window;
        self.h = horizon
        fin = open(file_name);
        self.rawdat = np.loadtxt(fin,delimiter=',');
        self.dat = np.zeros(self.rawdat.shape);
        self.n, self.m = self.dat.shape;
        self.normalize = 2
        self.scale = np.ones(self.m);
        self._normalized(normalize);
        self._split(int(train * self.n), int((train+valid) * self.n), self.n);
        
        self.scale = torch.from_numpy(self.scale).float();
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m);
            
        if self.cuda:
            self.scale = self.scale.cuda();
        self.scale = Variable(self.scale);
        
        self.rse = normal_std(tmp);
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp))); # Used to define rae, does ???
    
    def _normalized(self, normalize):
        #normalized by the maximum value of entire matrix.
       
        if (normalize == 0):
            self.dat = self.rawdat
            
        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat);
            
        #normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:,i]));
                self.dat[:,i] = self.rawdat[:,i] / np.max(np.abs(self.rawdat[:,i]));
            
        
    def _split(self, train, valid, test):
        
        train_set = range(self.P+self.h-1, train);  # It dosen't make predictions on the first self.P+self.h-1, because it needs self.P+self.h-1 inputs before making a prediction.
                                                    # It is training so that, given self.P+self.h-1 inputs predict horizon time steps ahead.
        valid_set = range(train, valid);
        test_set = range(valid, self.n);
        self.train = self._batchify(train_set);
        self.valid = self._batchify(valid_set);
        self.test = self._batchify(test_set);
        
        
    def _batchify(self, idx_set):
        
        #TODO Add the gaussian distribution with the numpy method to the numpy array, as done in https://blog.keras.io/building-autoencoders-in-keras.html
        # add the gaussian distribution to train set only, it should still try to validate on the original data. If we add noise to our dataset it will be a different dataset
        # than the LSTNet model trained on, but if we our testset is the same as the original LSTNet i.e. without the added noise, if our model performs better it should still be fine.
        # If our model adds the noise at run-time like it is currently planned, you could also argue that the (input) dataset is the same, the noise factor is just a part of the model. 

        # Step 1: Find where training and validation set for training is specified (this function), and see if it is possible to only change train set. Look at the arguments 
        # train[0] and train[1] in main, that comes from this functions return [X, Y].

        n = len(idx_set);
        X = torch.zeros((n,self.P,self.m)); # Prepares empty torch tensors to insert the result to instead of having it as numpy arrays
        Y = torch.zeros((n,self.m));
        
        for i in range(n):
            end = idx_set[i] - self.h + 1;
                                  
            start = end - self.P;   # Hypothesis: It makes the prediction, horizon steps ahead after the last input in the window
                                    # start 621, end = 789 (for testset) (included index 789 it is currently processing, meaning there is 12 steps ahead to 1000)
            
            # Adds noise to the input values (X), that will be used for training. At the moment it will also affect validation X (evaluation) and test X set, is that okay?
            noise_factor = 0.5
            #x_numpy_array = self.dat[start:end, :]
            #x_train_noisy = x_numpy_array + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_numpy_array.shape)
            #x_train_noisy = np.clip(x_train_noisy, 0., 1.)
            #X[i,:,:] = torch.from_numpy(x_train_noisy)
            X[i,:,:] = torch.from_numpy(self.dat[start:end, :]);    # The input for training, the 168 values it uses as input
            # Make array to ones and remove it
            Y[i,:] = torch.from_numpy(self.dat[idx_set[i], :]);     # The exact time step it tries to predict, window (168) + horizon (12)
        return [X, Y];

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]; Y = targets[excerpt];
            if (self.cuda):
                X = X.cuda();
                Y = Y.cuda();  
            yield Variable(X), Variable(Y);
            start_idx += batch_size