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
        self.pt = (self.P - self.Ck)/self.skip
        self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m)); #In channel, out channel, kernel size (1, 50, (6, 8))
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
        batch_size = x.size(0);
        
        #CNN
        c = x.view(-1, 1, self.P, self.m);      #(128, 1, 168, 8) change x shape to: -1 = take whatever previous shape (this shape will vary), 1, self.P = 168 (P-window), m = 8 (columns) 
        #print(c.shape)
        # Hypothesis: the second argument in x is read as output channel, which will be the input channel for the next layer, which is why conv1, takes 1 as input layer.
        #print(c.shape) 
        c = F.relu(self.conv1(c));  
        # conv layer setting:   (1, 50, (6,8)), 
        # input:                (128, 1, 168, 8)
        # output                (128, 50, 163, 1)
        # Hypothesis: after being in conv layer that takes input size 1 it outputtet a layer with size 50. 163, 1 tho?      
        c = self.dropout(c);        # Dropout, does not change the shape
        #print(c.shape)
        c = torch.squeeze(c, 3);    # (128, 50, 163) Fourth dimension "1" is removed (or 3 if you count from 0)
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous();    # #163, 128, 50 Permute: Returns a view of the original tensor with its dimensions permuted 
        # (2, 0, 1) Column 2 is no the first column, column 0 is the second and 1 is the third.
        #When you call contiguous() , it actually makes a copy of tensor so the order of elements would be same as if tensor of same shape created from scratch. 
        #Normally you don't need to worry about this. If PyTorch expects contiguous tensor but if its not then you will get
        # RuntimeError: input is not contiguous and then you just add a call to contiguous().
        print(r.shape) 
        _, r = self.GRU1(r);    #GRU layer expects input HidC = 50, HidR = 50
        r = self.dropout(torch.squeeze(r,0));

        
        #skip-rnn
        
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous();
            s = s.view(batch_size, self.hidC, self.pt, self.skip);
            s = s.permute(2,0,3,1).contiguous();
            s = s.view(self.pt, batch_size * self.skip, self.hidC);7
            _, s = self.GRUskip(s);
            s = s.view(batch_size, self.skip * self.hidS);
            s = self.dropout(s);
            r = torch.cat((r,s),1);
        
        res = self.linear1(r);
        
        #highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :];
            z = z.permute(0,2,1).contiguous().view(-1, self.hw);
            z = self.highway(z);
            z = z.view(-1,self.m);
            res = res + z;
            
        if (self.output):
            res = self.output(res);
        return res;
    
        
        
        