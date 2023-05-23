import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class RNN(nn.Module):
    def __init__(self, output_dim=64, input_size=64, hidden_size=64, num_layers=3):
        '''
            Output_dim (int): dimension of vector output by TextEncoder
            Input_size (int): Dimension of vector pass the TextEncoder
            Hidden_size (int): hidden_size of RNN
        '''
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,\
             num_layers=num_layers, bias=True, batch_first=False)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, output_dim*2),
            nn.Dropout(0,1),
            nn.ReLU(),
            nn.Linear(output_dim*2, output_dim),
        )

    def forward(self, x_padded, x_length):
        '''
        Input:
            x_padded (torch.Tesnor): [src_len, batch_size, hidden_size]
            x_length (list[int]): (b, )
        Output:
            h (torch.Tensor): [b, hidden_size]
        '''
        src_len, batch_size, hidden_size = x_padded.shape
        x_packed = pack_padded_sequence(x_padded, x_length, enforce_sorted=False)
        enc_hiddens, last_hidden = self.rnn(x_packed)
        last_hidden = last_hidden[self.num_layers-1,...]
        y = self.projection(last_hidden)
        return y

if __name__ == "__main__":
    x_padded = torch.zeros((10, 4, 64)) # [src_len, batch_size, hidden_size]
    x_length = [9, 7, 6, 1]             # x_length
    text_encoder = RNN()                
    y = text_encoder(x_padded, x_length)
    print(y.shape)

