import torch

from Module.encoder import Encoder
from Module.decoder import Decoder
from Module.other_module import Generator

class Transformer(torch.nn.Module):
    def __init__(self,src_voc_size, tgt_vob_size, 
                 d_model, n_head, max_size, ffn_hidden, n_layers, dropout, device):
        super(Transformer, self).__init__()
        self.src_pad_idx = 1
        self.tgt_pad_idx = 1
        self.device = device
        self.generator = Generator(d_model, tgt_vob_size)

        self.encoder = Encoder(src_voc_size, max_size, d_model,
                               ffn_hidden, n_head, n_layers, 
                               dropout, device)
        
        self.decoder = Decoder(tgt_vob_size, max_size, d_model, 
                              ffn_hidden, n_head, n_layers, 
                              dropout, device)
        
    
    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, memory, src_mask, tgt_mask):
        return self.decoder(tgt, memory, src_mask, tgt_mask)
    
    def forward(self, src, tgt, src_mask, src_tgt_mask, tgt_mask):
        return self.decode(tgt, self.encode(src, src_mask), src_tgt_mask, tgt_mask)
        
    def Generator(self, x):
        return self.generator(x)
    