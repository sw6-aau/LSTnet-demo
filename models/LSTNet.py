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
        self.pt = (self.P - self.Ck)/self.skip  # (168 - 6) / 24 = 6.75 rounded 6 # If they have made a mistake here it would make a lot more sense. Then 168 - 6 would be
                                                # sequence length after being processed by CNN layer, but the actual shape after CNN layer was 163 (i.e. - 5), but that can be unintuitive to
                                                # determine, since it is sequence length - (Ck - 1) after being processed by CNN layer not sequence length - Ck.
                                                # This would make sense of the RNN-skip arguments, they want to feed the input to the RNN with the knoweledge
                                                # that the inputs are 24 hours in a day, so they div
        self.hw = args.highway_window
        
        self.encode = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m)); # Kernel size = 6 * 8
        
        self.height_after_conv = (self.P - (self.Ck - 1))           # Conv layer changes shape by, Input size - (height - 1)
        self.height_after_pooling = self.height_after_conv/2        # Max pooling 2x2 gives a input size reduction of input_size / 2
        self.deconv_height = self.P - self.height_after_pooling + 1 # Transpose layer adds the height argument to the input shape -1, after convolution. So  
                                                                    # in order to get the original size of self.P, we find the difference between height after the
                                                                    # pooling layer and adds 1 to negate the -1 that is subtracted when using the method.
        
        self.decode = nn.ConvTranspose2d(self.hidC, 1, (self.deconv_height, self.m))

        self.change_hidden = nn.Linear(in_features=8, out_features=self.hidC)
        
        self.pool = nn.MaxPool2d(2)
        # Linear / dense layer that acts as a hidden 50 unit layer.
        #self.fifty_layer = nn.Linear(8, 50);
        
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
 
        # 163 / 2 + 7 
        # 168 - 81 + 1 (81 input efter pooling)


        # New convolutional layer

        #self.decoder1 = nn.ConvTranspose2d(self.hidC, self.hidC, kernel_size = (self.Ck, self.m))

    def forward(self, x):
        batch_size = x.size(0);
        #print(self.P - (self.P - (self.Ck - 1))/2 + 1)
        #print(self.Ck - 1)
        c = x.view(-1, 1, self.P, self.m);  # (128, 1, 168, 8)  <--- efter dette bliver 8 betragtet som input size (1 * 8), er det okay? virker som om det passer fint med vores
                                            # multi variant model med 8 forskellige markedere (for stocks) 168 height in conv layer
        # CNN Autoencoder
        c = F.relu(self.encode(c))      # (128, 50, 163, 1)
        c = self.pool(c)                # (128, 50, 81, 1) (rip 163 / 2 = 81, rounding down)
        c = F.relu(self.decode(c))  
        c = self.dropout(c);
        #CNN
        c = F.relu(self.change_hidden(c)) # (128, 1, 168, 50) (7*24)
        c = self.dropout(c);
        c = c.view(-1, self.P, self.hidC); # Flattens / reshapes to three arguments which in practice removes the 1
                                           # in second argument by multpiplying the layer with the 50 layer.
        # output before changes: (128, 50, 163)
        # new output after changes: (128, 168, 50)
        c = c.permute(0, 2, 1)     # Permute magic, to old format
        # RNN 
        r = c.permute(2, 0, 1).contiguous();        # The order was changed from (128, 50, 168) to (168, 128, 50) (was 163 when we had conv and not autoencode)
        _, r = self.GRU1(r);                        # shape parameters from doc: seq_len, batch, input_size, 
                                                    # input size is layer input size, batch is batch. "seq_len - the number of time steps in each input stream." i.e. each input
                                                    # that is fed to the RNN will consist of 168 rows of time stamps. Remember that all the input processed in each batch.
        r = self.dropout(torch.squeeze(r,0));

        
        #skip-rnn # adds a rnn with periodic patterns, in the current implementation it's a rnn with day to day steps and predictions
        
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous();  # c = 128, 168, 50 (changed by us), (all elements from ":") s = : (128), : (168), -162: (gets the last 162 elements)  
                                                                  # int calculation = (-6 * 24) = -144
            s = s.view(batch_size, self.hidC, self.pt, self.skip);      # expected (128, 50, -6.75, 24)
            s = s.permute(2,0,3,1).contiguous();                        # expected (6, 128, 24, 50) (self.pt, batch-size, self.skip, self.hidC) Permutes so view, combines batch and self.skip
            s = s.view(self.pt, batch_size * self.skip, self.hidC);     # expected (6, 128*24=3072, 50) removes the last dimension, view does this by multiplying the last dimension 
                                                                        # with the dimension two dimensions behind it.
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
            z = x[:, -self.hw:, :]; # (128, 24, 8) #self.hw = 24 (normally 168, but looks at last 24 input values)
            z = z.permute(0,2,1).contiguous().view(-1, self.hw); # (1024, 24)   #1024 samples (128 samples med 8 markeder flattened = 1024)
            z = self.highway(z);        # In the documentation of linear's input shapes (not definition), 
                                        # the inputs can be whatever dimensions, as long as the last dimension is input size, this last dim is 24, and is changed to 8

            z = z.view(-1,self.m);      # Assumed output: (1024, 8) WRONG shapes it back to (128, 8)
            res = res + z;
            
        if (self.output):
            res = self.output(res);
        return res;
     
        
        
