import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window;
        self.m = data.m
        self.hidC = args.hidCNN;
        self.hidR = args.hidRNN;
        self.hidS = args.hidSkip;
        self.Ck = args.CNN_kernel;
        self.skip = args.skip;
        self.pt = int((self.P - self.Ck)/self.skip)
        self.hw = args.highway_window
        
        self.encode = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m));
        
        self.height_after_conv = (self.P - (self.Ck - 1))
        self.pooling_factor = 4
        self.height_after_pooling = int(math.ceil(float(self.height_after_conv)/self.pooling_factor)) 
        self.deconv_height = self.P - self.height_after_pooling + 1
        
        self.decode = nn.ConvTranspose2d(self.hidC, 1, (self.deconv_height, self.m))

        self.change_hidden = nn.Linear(in_features=self.m, out_features=self.hidC)
        
        self.pool = nn.MaxPool2d(1, self.pooling_factor)
        
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
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;
        if (args.output_fun == 'relu'):
            self.output = F.relu;

    def forward(self, x):
        batch_size = x.size(0);
        c = x.view(-1, 1, self.P, self.m);
        # CNN Autoencoder
        ae = self.encode(c)
        ae = self.pool(ae)
        ae = self.decode(ae)
        ae_hw = torch.squeeze(ae, 1)
        #CNN
        c = F.relu(self.change_hidden(ae))
        c = self.dropout(c);
        c = torch.squeeze(c, 1)
        c = c.permute(0, 2, 1)     # Permute magic, to old format
        
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