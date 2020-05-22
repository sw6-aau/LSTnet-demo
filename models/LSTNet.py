import torch
import torch.nn as nn
import torch.nn.functional as F

# OG LSTnet model
# Do not change
# Make a new python file instead
class Model(nn.Module):
    def __init__(self, args, data, cnn, rnn, skip, activation):
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
        if (activation == 'sigmoid'):
            self.output = F.sigmoid;
        if (activation == 'tanh'):
            self.output = F.tanh;
        if (activation == 'relu'):
            self.output = F.relu;

    def forward(self, x):
        batch_size = x.size(0);
        
        #CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous() # torch.Size([163, 128, 50])
        _, r = self.GRU1(r);
        r = self.dropout(torch.squeeze(r,0))

        #skip-rnn
        
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous() #torch.Size([128, 50, 144])
            s = s.view(batch_size, self.hidC, self.pt, self.skip) #torch.Size([128, 50, 6, 24])
            s = s.permute(2,0,3,1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC) # torch.Size([6, 3072, 50])
            _, s = self.GRUskip(s) #s: torch.Size([1, 3072, 5]) Tensor containing the the hidden state for t = seq_len, that is the last predicted day of the week
            s = s.view(batch_size, self.skip * self.hidS) # torch.Size([128, 120]) Splits the predictions of each day, to each 128 input again
            s = self.dropout(s)
            r = torch.cat((r,s),1) #r:torch.Size([128, 170]), s: torch.Size([128, 50]), r-after: torch.Size([128, 170])
        
        res = self.linear1(r); # torch.Size([128, 8]) Changes second dimension to 8 through a linear layer.
        
        #highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :]; # torch.Size([128, 24, 8])
            z = z.permute(0,2,1).contiguous().view(-1, self.hw); # torch.Size([1024, 24])
            z = self.highway(z);    # 24 -> 1
            z = z.view(-1,self.m); # torch.Size([128, 8])
            res = res + z;
            
        if (self.output):
            res = self.output(res);
        return res;
     
        
        
