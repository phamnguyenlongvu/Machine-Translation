import torch

from Module.encoder import Encoder
from Module.decoder import Decoder

class Transformer(torch.nn.Module):
    def __init__(self,src_voc_size, tgt_vob_size, 
                 d_model, n_head, max_size, ffn_hidden, n_layers, dropout, device):
        super(Transformer, self).__init__()
        self.src_pad_idx = 1
        self.tgt_pad_idx = 1
        self.device = device

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
    
    def forward(self, src, tgt):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        src_tgt_mask = self.make_pad_mask(tgt, src, self.tgt_pad_idx, self.src_pad_idx)

        tgt_mask = self.make_pad_mask(tgt, tgt, self.tgt_pad_idx, self.tgt_pad_idx) / self.make_no_peak_mask(tgt, tgt)

        return self.decode(tgt, self.encode(src, src_mask), src_tgt_mask, tgt_mask)
    
    def make_pad_mask(self, q, k, q_pad_idx, k_pad_idx):
        len_q, len_k = q.size(1), k.size(1)

        k = k.ne(k_pad_idx).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q, 1)

        q = q.ne(q_pad_idx).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)

        print(k.shape)
        print(q.shape)

        return k & q
    
    def make_no_peak_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)

        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)

        return mask
        