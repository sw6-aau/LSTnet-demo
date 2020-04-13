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
        
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m));
        
        self.encode = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m));
        self.decode = nn.ConvTranspose2d(self.hidC, 1, (self.Ck, self.m))
        
        
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
 


        # New convolutional layer

        #self.decoder1 = nn.ConvTranspose2d(self.hidC, self.hidC, kernel_size = (self.Ck, self.m))

    def forward(self, x):
        batch_size = x.size(0);


        c = x.view(-1, 1, self.P, self.m);
        
        print("BEFORE: ")
        print(c.shape)
        # CNN Autoencoder
        c = F.relu(self.encode(c))
        c = F.relu(self.decode(c))
        #CNN
        c = F.relu(self.conv1(c));
        print("AFTER: ")
        print(c.shape)
        #c = F.relu(self.decode(c))
        # Dropout and squeeze
        c = self.dropout(c);
        print("after dropout")
        print(c.shape)
        c = torch.squeeze(c, 3);   #This will only squeeze 1's so when you add deconv layer the fourth dim returns to being 8.
                                   #Thought: This implies that conv layer flattens the last dim to be 1 instead of 8.

        # RNN 
        print("Shape RNN")
        r = c.permute(2, 0, 1).contiguous();        # We could simply not permute and feed the RNN the original shape of the data, why do they permute? Is the 
                                                    # order the input is fed to the RNN somehow important?
                                                    # The order was changed from (128, 50, 163) to (163, 128, 50)
                                                    # parameters from doc: input_size, hidden_size, num_layers
                                                    # num_layers = 50 output layers from CNN layer
                                                    # 128 should be the batch size, but is put in as hidden_size, makes sense if you consider that each time series step in the batch
                                                    # must have its own hidden layer representation
                                                    # num layers = convoluted(self.P) (P = window, P from report i.e. look back horizon) default without conv 24*7 = 168
                                                    # How is it that conv layer changes (128, 1, 168, 8) to (128, 50, 163, 1)
                                                    # In theory permuting the original input to (128, 8, 168) should be legal. It would mean it takes the 8 input layers, 
                                                    # of the 50 from the convolutional network, but i dont know what effect that will have. You could also apply a linear layer
                                                    # that changes the input 8 layer to a 50 layer, like we did with linear encoding, but again not sure of the effect. 
                                                    # In theory it should be perfectly fine as this is how normal neural networks are made, input layer (typically 
                                                    # 1 if not multivariant like ours), hidden layers, and output layers (typically same amount as input layers)
        print(r.shape)
        _, r = self.GRU1(r);
        r = self.dropout(torch.squeeze(r,0));

        
        #skip-rnn
        
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous();
            s = s.view(batch_size, self.hidC, self.pt, self.skip);
            s = s.permute(2,0,3,1).contiguous();
            s = s.view(self.pt, batch_size * self.skip, self.hidC);
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
     
        
        
