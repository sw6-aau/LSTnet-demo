import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    def __init__(self, args, data, cnn, rnn, skip, activation, kernel):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window;
        self.m = data.m
        self.hidC = cnn;
        self.hidR = rnn;
        self.hidS = skip;
        self.Ck = args.CNN_kernel;
        self.skip = args.skip;
        self.pt = int((self.P - self.Ck)/self.skip)
        self.hw = args.highway_window
        
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m));

        self.height_after_conv = (self.P - (self.Ck - 1))           # Conv layer changes shape by, Input size - (height - 1)
        self.pooling_factor = 4
        self.height_after_pooling = int(math.ceil(float(self.height_after_conv)/self.pooling_factor))        # Max pooling 2x2 gives a input size reduction of input_size / 2
        self.deconv_height = self.P - self.height_after_pooling + 1 # Transpose layer adds the height argument to the input shape -1, after convolution. So  
                                                                    # in order to get the original size of self.P, we find the difference between height after the
                                                                    # pooling layer and adds 1 to negate the -1 that is subtracted when using the method.
        
        #self.pool = nn.MaxPool2d(1, self.pooling_factor)
        stride = 1
        padding = 0
        output_padding = 0
        dropout = 0
        reluf = False
        kernel_size = kernel  # 2 and 8 gives 0 results funny enough TRY HYPERTUNING KERNEL SIZE DROUPUT STRIDE, they are very sensitive to change

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size, stride=stride, padding=padding),
            nn.Conv1d(128, 64, kernel_size, stride=stride, padding=padding),
            nn.Conv1d(64, 32, kernel_size, stride=stride, padding=padding),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 64, kernel_size, stride=stride, padding=padding, output_padding=output_padding),
            nn.ConvTranspose1d(64, 128, kernel_size, stride=stride, padding=padding, output_padding=output_padding),
            nn.ConvTranspose1d(128, 1, kernel_size, stride=stride, padding=padding, output_padding=output_padding),
            nn.Dropout(dropout),
        )


        self.GRU1 = nn.GRU(self.hidC, self.hidR);
        self.dropout = nn.Dropout(p = args.dropout);
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS);
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m);
        else:
            self.linear1 = nn.Linear(self.hidR, self.m);
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1);

        # Deprecated shit that somehow only works because it's deprecated
        # GG PyTorch
        self.output = None;
        if (activation == 'sigmoid'):
            self.output = F.sigmoid;
        if (activation == 'tanh'):
            self.output = F.tanh;
        if (activation == 'relu'):
            self.output = F.relu;

        
    def forward(self, x):
        batch_size = x.size(0);
        ae = x.view(-1, 1, self.P * self.m)

        # CNN Autoencoder
        ae = self.encoder(ae)
        ae = self.decoder(ae)
        ae = ae.view(-1, 1, self.P, self.m)
        ae_hw = torch.squeeze(ae, 1);

        #CNN
        c = F.relu(self.conv1(ae))
        c = self.dropout(c);
        c = torch.squeeze(c, 3)
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous();
        _, r = self.GRU1(r);
        r = self.dropout(torch.squeeze(r,0));

        
        #skip-rnn
        
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous();  # c = 128, 168, 50 (changed by us), (all elements from ":") s = : (128), : (168), -162: (gets the last 162 elements)  
                                                                  # int calculation = (-6.75 * 24) = -162
            s = s.view(batch_size, self.hidC, self.pt, self.skip);  # expected : (128, 50, )
            s = s.permute(2,0,3,1).contiguous();
            s = s.view(self.pt, batch_size * self.skip, self.hidC);
            _, s = self.GRUskip(s);
            s = s.view(batch_size, self.skip * self.hidS);
            s = self.dropout(s);
            r = torch.cat((r,s),1);
        
        res = self.linear1(r);
        
        #highway
        if (self.hw > 0):
            z = ae_hw[:, -self.hw:, :]; # (128, 24, 8) #self.hw = 24 (normally 168, but looks at last 24 input values)
            z = z.permute(0,2,1).contiguous().view(-1, self.hw); # (1024, 24)   #1024 samples (128 samples med 8 markeder flattened = 1024)
            z = self.highway(z);        # In the documentation of linear's input shapes (not definition), 
                                        # the inputs can be whatever dimensions, as long as the last dimension is input size, this last dim is 24, and is changed to 8
            z = z.view(-1,self.m);      # Assumed output: (1024, 8) WRONG shapes it back to (128, 8)
            res = res + z;
            
        if (self.output):
            res = self.output(res);
        return res;