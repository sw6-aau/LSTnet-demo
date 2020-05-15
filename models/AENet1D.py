import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    def __init__(self, args, data, cnn, kernel):
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
      
        self.height_after_conv = (self.P - (self.Ck - 1))           # Conv layer changes shape by, Input size - (height - 1)
        self.pooling_factor = 4
        self.height_after_pooling = int(math.ceil(float(self.height_after_conv)/self.pooling_factor))        # Max pooling 2x2 gives a input size reduction of input_size / 2
        self.deconv_height = self.P - self.height_after_pooling + 1 # Transpose layer adds the height argument to the input shape -1, after convolution. So  
                                                                    # in order to get the original size of self.P, we find the difference between height after the
                                                                    # pooling layer and adds 1 to negate the -1 that is subtracted when using the method.
        
        self.pool = nn.MaxPool2d(1, self.pooling_factor)
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
        
    # AE 128, 168, 8
    def forward(self, x): # 128, 168, 8
        batch_size = x.size(0)
        ae = x.view(-1, 1, self.P * self.m)
        ae = self.encoder(ae)
        ae = self.decoder(ae)
        ae = ae.view(-1, 1, self.P, self.m)
        ae = torch.squeeze(ae, 1);
        new_torch = torch.zeros(batch_size, self.m)
        new_torch = ae[:,-1,:]
        #F.relu(new_torch)
        return new_torch;
