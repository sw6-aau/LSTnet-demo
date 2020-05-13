import torch
import numpy as np;
from torch.autograd import Variable


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, cuda, horizon, window, normalize = 2):
        self.cuda = cuda;
        self.P = window;
        self.h = horizon
        fin = open(file_name);
        self.rawdat = np.loadtxt(fin,delimiter=',')
        self.dat = np.zeros(self.rawdat.shape)  # Prepares a empty numpy array with the shape of our dataset
        self.n, self.m = self.dat.shape         # Sets rows and column size variable. .shape returns the shape in this case (rows, number columns)
        self.normalize = 2
        self.scale = np.ones(self.m)            # np.ones makes a numpy array with ones in the shape given. 
        self._normalized(normalize)           
        self._split(self.n) 
        
        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)
            
        if self.cuda:
            self.scale = self.scale.cuda()
        self.scale = Variable(self.scale)
        
        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))
    
    def _normalized(self, normalize):
        self.dat = self.rawdat
            
        
    def _split(self, test):
        test_set = range(self.P+self.h-1, self.n); #179 (168 + 12 - 1)
        self.test = self._batchify(test_set);
        
    # Splits the data set in windows the net will train on defined in self.P, minus the horizon because it will need horizon ahead to confirm if it predicted correctly
    def _batchify(self, idx_set):
        n = len(idx_set);
        X = torch.zeros((n,self.P,self.m)); #Initiliazes tensor (1000, 178, 8) 
        Y = torch.zeros((n,self.m));        #Initiliazes tensor (1000, 8) 
        
        for i in range(n):
            end = idx_set[i] - self.h + 1;  #  179-988  (168 + 12 - 1 (-1 because index 0 is counted)) # What we want 
            start = end - self.P;           # start: 0 - 820 (+ 179 = 999) # What we want 0 - 999
            X[i,:,:] = torch.from_numpy(self.dat[start:end, :]);    # The first dimension of the X array is  from start to end, excluding the [end] element. X[]
            Y[i,:] = torch.from_numpy(self.dat[idx_set[i], :]);     # The first dimension of the Y array equal to same index in the index array.
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