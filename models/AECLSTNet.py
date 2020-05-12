import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window;
        self.m = data.m
        self.hidR = args.hidRNN;
        self.hidC = args.hidCNN;
        self.hidS = args.hidSkip;
        self.Ck = args.CNN_kernel;
        self.skip = args.skip;
        self.pt = int((self.P - self.Ck)/self.skip)  # (168 - 6) / 24 = 6.75 rounded 6 # If they have made a mistake here it would make a lot more sense. Then 168 - 6 would be
                                                # sequence length after being processed by CNN layer, but the actual shape after CNN layer was 163 (i.e. - 5), but that can be unintuitive to
                                                # determine, since it is sequence length - (Ck - 1) after being processed by CNN layer not sequence length - Ck.
                                                # This would make sense of the RNN-skip arguments, they want to feed the input to the RNN with the knoweledge
                                                # that the inputs are 24 hours in a day, so they divide the window length with 24 hours.
        self.hw = args.highway_window
        #self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m));

        self.encode = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m)); # 168, 8, -5 -7 = 163, 1
        
        self.height_after_conv = (self.P - (self.Ck - 1))           # Conv layer changes shape by, Input size - (height - 1)
        self.height_after_pooling = self.height_after_conv/2        # Max pooling 2x2 gives a input size reduction of input_size / 2
        self.deconv_height = self.P - self.height_after_pooling + 1 # Transpose layer adds the height argument to the input shape -1, after convolution. So  
                                                                    # in order to get the original size of self.P, we find the difference between height after the
                                                                    # pooling layer and adds 1 to negate the -1 that is subtracted when using the method.
        self.decode = nn.ConvTranspose2d(self.hidC, 1, (self.Ck, self.m)) # 163, 1, 
        self.pool = nn.MaxPool2d(1,4)
        

        #self.encode1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m)); 
        #self.encode2 = nn.Conv2d(self.hidC, 25, kernel_size = (self.Ck, 1)); 
        #self.encode3 = nn.Conv2d(25, 12, kernel_size = (self.Ck, 1));
        
        #self.pool = nn.MaxPool2d(2)

        #self.decode3 = nn.ConvTranspose2d(12, 25, (self.Ck, 1))
        #self.decode2 = nn.ConvTranspose2d(25, self.hidC, (self.Ck, 1))
        #self.decode1 = nn.ConvTranspose2d(self.hidC, 1, (self.Ck, self.m))


        self.GRU1 = nn.GRU(self.hidC, self.hidR);
        self.dropout = nn.Dropout(p = args.dropout);

        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS);
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m);
        else:
            self.linear1 = nn.Linear(self.hidR, self.m);
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1);
        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;



    def forward(self, x):
        # Y (128, 8)    (128, 168, 8) (128, 50, 163, 1)
        batch_size = x.size(0);
        c = x.view(-1, 1, self.P, self.m)
        # CNN Autoencoder (3 layer)
        #ae = self.encode1(ae)   # width is 1 this layer, since 8 - (8 - 1) = 1
        #ae = self.encode2(ae)
        #ae = self.encode3(ae)
        #ae = self.decode3(ae)
        #ae = self.decode2(ae)
        #ae = self.decode1(ae)
        
        # CNN
        c = self.encode(c)      # (128, 50, 163, 1)
        
        reconstructed = self.pool(c)                # (128, 50, 81, 1) (163 / 2 = 81, rounding down)
        reconstructed = F.relu(self.decode(c))
        reconstructed = torch.squeeze(reconstructed, 1);

        
        #CNN DROPOUT MAYBE AFTER CNN LAYER
        c = self.dropout(c);
        c = torch.squeeze(c, 3);

        # RNN 
        r = c.permute(2, 0, 1).contiguous();        # The order was changed from (128, 50, 163) to (163, 128, 50)
        _, r = self.GRU1(r);                        # shape parameters from doc: seq_len, batch, input_size, 
                                                    # input size is layer input size, batch is batch. "seq_len - the number of time steps in each input stream." i.e. each input
                                                    # that is fed to the RNN will consist of 168 rows of time stamps. Remember that all the input is processed in each batch.
        r = self.dropout(torch.squeeze(r,0));

        
        #skip-rnn # adds a rnn with periodic patterns, in the current implementation it's a rnn with day to day steps and predictions
        
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous();        # c = 128, 50, 163 | s = c[(all elements from ":") : (128), : 50, -162: (gets the last 162 elements)]  
                                                                        # int calculation = (6 * 24) = 144
            s = s.view(batch_size, self.hidC, self.pt, self.skip);      # (128, 50, 6, 24)
            s = s.permute(2,0,3,1).contiguous();                        # (6, 128, 24, 50) (self.pt, batch-size, self.skip, self.hidC) Permutes so view, combines batch and self.skip
            s = s.view(self.pt, batch_size * self.skip, self.hidC);     # (6, 128*24=3072, 50) removes the last dimension, view does this by multiplying as defined 
           
            _, s = self.GRUskip(s);                                     # shape parameters from doc: seq_len, batch, input_size. Batch increased because each sequence time step
                                                                        # now is one day (24 hours) instead of one hour, between each time stamp. Meaning this configuration would try
                                                                        # to predict a day forward.
                                                                        # output doc:
                                                                        # h_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t = seq_len
                                                                        # output: (1, 3072, 5)
            s = s.view(batch_size, self.skip * self.hidS);              # Reshapes the vector back to batches of 128 (one hour) format, output: 128, 120 (24 * 5)
                                                                        # hidS: The number of features in the hidden state h
            s = self.dropout(s);
            r = torch.cat((r,s),1);
        
        res = self.linear1(r);
        
        # highway # Adds a linear layers directly on the input data (works as a highway). The idea is the dense layer performing autoregresion (attempting to guess the future,
        # in this case by adjusting the result with weights in its dense layer) on the raw data, independeltly from any kind of RNN. This makes it more sensitive to 
        # violated scale fluctuations in data (which is typical in Electricity data possibly due to random events for public holidays or temperature turbulence, etc.)
        # which it can fit its weights to adjust for (given they are a frequent occurence).
        # Also really important for keeping the general base structure of the input, since this part tries to adjusts weights to mimick the original input, without first 
        # deconstructing the input in a RNN hidden state....?
        if (self.hw > 0):
            z = x[:, -self.hw:, :];                                     # (128, 24, 8) self.hw = 24 (normally 168, but looks at last 24 input values)
            z = z.permute(0,2,1).contiguous().view(-1, self.hw);        # (1024, 24)   1024 samples (128 samples with 8 stock markets flattened = 1024)
            z = self.highway(z);                                        # (1024, 1) In the documentation of linear's input shapes (not definition), 
                                                                        # the inputs can be whatever dimensions, as long as the last dimension is input size, 
                                                                        # this last dim is 24, and is changed to 1
            z = z.view(-1,self.m);                                      # (128, 8) view reshapes the output of the highway layer back to 8 dimensions / collumns
            res = res + z;                                              # Adds the result from the two RNNs' with the highway layer.
            
        if (self.output):
            res = self.output(res);
        return res, reconstructed;
     
        
        
