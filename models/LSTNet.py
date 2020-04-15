import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, data, cnn, rnn, skip, activation):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window;
        self.m = data.m
        self.hidR = rnn;
        self.hidC = cnn;
        self.hidS = skip;
        self.Ck = args.CNN_kernel;
        self.skip = args.skip;
        self.pt = (self.P - self.Ck)/self.skip  # (168 - 6) / 24 = 6.75
        self.hw = args.highway_window
        
        #self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m)); # Kernel size = 6 * 8
        
        self.encode = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m));
        self.decode = nn.ConvTranspose2d(self.hidC, 1, (88, self.m))       # Hardcoded 81 for now (163 / 2) (168 - 81 + 1)

        self.change_hidden = nn.Linear(in_features=self.m, out_features=self.hidR)
        
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
        if (activation == 'sigmoid'):
            self.output = F.sigmoid;
        if (activation == 'tanh'):
            self.output = F.tanh;
 


        # New convolutional layer

        #self.decoder1 = nn.ConvTranspose2d(self.hidC, self.hidC, kernel_size = (self.Ck, self.m))

    def forward(self, x):
        batch_size = x.size(0);

        c = x.view(-1, 1, self.P, self.m);  # (128, 1, 168, 8)  <--- efter dette bliver 8 betragtet som input size (1 * 8), er det okay? virker som om det passer fint med vores
                                            # multi variant model med 8 forskellige markedere (for stocks) 168 height in conv layer

        # CNN Autoencoder
        c = F.relu(self.encode(c))      # (128, 50, 163, 1)
        c = self.pool(c)                # (128, 50, 81, 1) (rip 163 / 2 = 81, rounding down)
        c = F.relu(self.decode(c))  
        #CNN
        c = F.relu(self.change_hidden(c)) # (128, 1, 168, 50)
        c = self.dropout(c);
        print(c.shape)
        c = c.view(-1, self.P, self.hidC); # Flattens / reshapes to three arguments which in practice removes the 1
                                           # in second argument by multpiplying the layer with the 50 layer.
        print(c.shape)
        # output before changes: (128, 50, 163)
        # new output after changes: (128, 168, 50)
        c = c.permute(0, 2, 1)     # Permute magic, to old format
        # RNN 
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
            z = x[:, -self.hw:, :]; # (128, 24, 8) #self.hw = 24 (normally 168, but looks at last 24 input values)
            z = z.permute(0,2,1).contiguous().view(-1, self.hw); # (1024, 24)   #1024 samples (128 samples med 8 markeder flattened = 1024)
            z = self.highway(z);        # In the documentation of linear's input shapes (not definition), 
                                        # the inputs can be whatever dimensions, as long as the last dimension is input size, this last dim is 24, and is changed to 8
            z = z.view(-1,self.m);      # Assumed output: (1024, 8) WRONG shapes it back to (128, 8)
            res = res + z;
            
        if (self.output):
            res = self.output(res);
        return res;
     
        
        
