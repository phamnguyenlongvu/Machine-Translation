import torch
from . import other_module

class DecoderLayer(torch.nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, dropout):
        super(DecoderLayer, self).__init__()
        self.attention = other_module.MultiHeadAttention(d_model, n_head)
        self.norm1 = other_module.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)

        self.encoder_decoder_attention = other_module.MultiHeadAttention(d_model, n_head)
        self.norm2 = other_module.LayerNorm(d_model)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.ffn = other_module.PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm3 = other_module.LayerNorm(d_model)
        self.dropout3 = torch.nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        print("In decoder")
        # self attention - sublayer1
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=tgt_mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # sublayer2
        if memory is not None:
            _x = x
            x = self.encoder_decoder_attention(q=x, k=memory, v=memory, mask=src_mask)
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # sublayer3
        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x
    
class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, max_size, d_model, ffn_hidden, n_head, n_layers, dropout, device):
        super(Decoder, self).__init__()
        self.emb = other_module.TransformerEmbedding(vocab_size, d_model, max_size, device)

        self.layers = torch.nn.ModuleList([DecoderLayer(d_model, ffn_hidden, n_head, dropout)
                                           for _ in range(n_layers)])
        self.linear = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        out = self.linear(x)
        return out



