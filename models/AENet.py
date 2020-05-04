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
                                                # that the inputs are 24 hours in a day, so they divide the window length with 24 hours.

        self.encode = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m)); # Kernel size = 6 * 8
        
        self.height_after_conv = (self.P - (self.Ck - 1))           # Conv layer changes shape by, Input size - (height - 1)
        self.height_after_pooling = self.height_after_conv/2        # Max pooling 2x2 gives a input size reduction of input_size / 2
        self.deconv_height = self.P - self.height_after_pooling + 1 # Transpose layer adds the height argument to the input shape -1, after convolution. So  
                                                                    # in order to get the original size of self.P, we find the difference between height after the
                                                                    # pooling layer and adds 1 to negate the -1 that is subtracted when using the method.
        
        self.decode = nn.ConvTranspose2d(self.hidC, 1, (self.deconv_height, self.m))
        self.pool = nn.MaxPool2d(1, 4)
        

        #self.encode1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m)); 
        #self.encode2 = nn.Conv2d(self.hidC, 25, kernel_size = (self.Ck, 1)); 
        #self.encode3 = nn.Conv2d(25, 12, kernel_size = (self.Ck, 1));
        
        #self.pool = nn.MaxPool2d(2)

        #self.decode3 = nn.ConvTranspose2d(12, 25, (self.Ck, 1))
        #self.decode2 = nn.ConvTranspose2d(25, self.hidC, (self.Ck, 1))
        #self.decode1 = nn.ConvTranspose2d(self.hidC, 1, (self.Ck, self.m))


    def forward(self, x):
        # Y (128, 8)    (128, 168, 8) (128, 50, 163, 1)
        batch_size = x.size(0);
        ae = x.view(-1, 1, self.P, self.m)
        # CNN Autoencoder (3 layer)
        #ae = self.encode1(ae)   # width is 1 this layer, since 8 - (8 - 1) = 1
        #ae = self.encode2(ae)
        #ae = self.encode3(ae)
        #ae = self.decode3(ae)
        #ae = self.decode2(ae)
        #ae = self.decode1(ae)
        
        # Old autoencoder + dropped self.dropout
        ae = F.relu(self.encode(ae))      # (128, 50, 163, 1)
        ae = self.pool(ae)                # (128, 50, 81, 1) (163 / 2 = 81, rounding down)
        ae = F.relu(self.decode(ae))
        #print(ae.shape)
        ae_hw = torch.squeeze(ae, 1);
        temp = ae_hw.contiguous()
        return temp[:,-1,:];
     
        
        
