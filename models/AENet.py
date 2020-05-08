import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    def __init__(self, args, data, cnn):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window;
        self.m = data.m
        self.hidC = cnn;
        self.Ck = args.CNN_kernel;
        self.skip = args.skip;
        self.pt = (self.P - self.Ck)/self.skip  # (168 - 6) / 24 = 6.75 rounded 6 # If they have made a mistake here it would make a lot more sense. Then 168 - 6 would be
                                                # sequence length after being processed by CNN layer, but the actual shape after CNN layer was 163 (i.e. - 5), but that can be unintuitive to
                                                # determine, since it is sequence length - (Ck - 1) after being processed by CNN layer not sequence length - Ck.
                                                # This would make sense of the RNN-skip arguments, they want to feed the input to the RNN with the knoweledge
                                                # that the inputs are 24 hours in a day, so they divide the window length with 24 hours.

        #self.encode = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m)); # Kernel size = 6 * 8
        
        self.height_after_conv = (self.P - (self.Ck - 1))           # Conv layer changes shape by, Input size - (height - 1)
        self.pooling_factor = 4
        self.height_after_pooling = int(math.ceil(float(self.height_after_conv)/self.pooling_factor))        # Max pooling 2x2 gives a input size reduction of input_size / 2
        self.deconv_height = self.P - self.height_after_pooling + 1 # Transpose layer adds the height argument to the input shape -1, after convolution. So  
                                                                    # in order to get the original size of self.P, we find the difference between height after the
                                                                    # pooling layer and adds 1 to negate the -1 that is subtracted when using the method.
        
        #self.decode = nn.ConvTranspose2d(self.hidC, 1, (self.deconv_height, self.m))
        self.pool = nn.MaxPool2d(1, self.pooling_factor)


        self.encoder = nn.Sequential(
            #nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m)),
            nn.Conv1d(1, 128, 4),
            nn.Conv1d(128, 256, 4),
            nn.Conv1d(256, 256, 4),
            nn.Conv1d(256, 512, 4),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            #nn.ConvTranspose2d(self.hidC, 1, (self.deconv_height, self.m)),
            nn.ConvTranspose1d(512, 256, 4),
            nn.ConvTranspose1d(256, 256, 4),
            nn.ConvTranspose1d(256, 128, 4),
            nn.ConvTranspose1d(128, 1, 4),
            nn.ReLU(True)
        )
        

    def forward(self, x):
        # Y (128, 8)    (128, 168, 8) (128, 50, 163, 1)
        batch_size = x.size(0);
	print("X:")
	print(x.shape)
        ae = x.view(-1, 1, self.P * self.m)
        #print(batch_size)
        #print(ae.shape)
        # Old autoencoder + dropped self.dropout
        ae = self.encoder(ae)    # (128, 50, 163, 1)
        print('Encoded:')
	print(ae.shape)
	#ae = self.pool(ae)                # (128, 50, 81, 1) (163 / 2 = 81, rounding down)
        ae = self.decoder(ae)
	print('Decoded:')
	print(ae.shape)
	ae = ae.view(-1, 1, self.m)
        
	
        ae = torch.squeeze(ae, 1);
        temp = ae.contiguous()
	print('Contiguous:')
	print(temp.shape)
        return temp[:,:];
