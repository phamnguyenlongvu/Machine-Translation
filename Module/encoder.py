import torch
from . import other_module

class EncoderLayer(torch.nn.Module):

    def __init__(self, d_model, ffb_hidden, n_head, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = other_module.MultiHeadAttention(d_model, n_head)
        self.norm1 = other_module.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)

        self.ffn = other_module.PositionwiseFeedForward(d_model, ffb_hidden)
        self.norm2 = other_module.LayerNorm(d_model)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, x, src_mask):
        _x = x 
        x = self.attention(x, x, x, src_mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)

        return x
    
class Encoder(torch.nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, n_layers, dropout):
        super(Encoder, self).__init__()
        self.layers = torch.nn.ModuleList([EncoderLayer(d_model, ffn_hidden, n_head, dropout)
                                           for _ in range(n_layers)])
        
    def forward(self, x , src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)

        return x