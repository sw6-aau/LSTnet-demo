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
        self.pool = nn.MaxPool2d(2, 2)
        

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

        
        # Old autoencoder + dropped self.dropout
        ae = F.relu(self.encode(ae))      # (128, 50, 163, 1)
        ae = self.pool(ae)                # (128, 50, 81, 1) (163 / 2 = 81, rounding down)
        ae = F.relu(self.decode(ae))
        ae_hw = torch.squeeze(ae, 1);
        temp = ae_hw.contiguous()
        return temp[:,-1,:];

### Sequitor Recurrent Encoder Decoder ###

class Encoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()

    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )

  def forward(self, x):
    x = x.reshape((1, self.seq_len, self.n_features))

    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)

    return hidden_n.reshape((self.n_features, self.embedding_dim))


class Decoder(nn.Module):

  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))

    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((self.seq_len, self.hidden_dim))

    return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(RecurrentAutoencoder, self).__init__()

    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)

    return x