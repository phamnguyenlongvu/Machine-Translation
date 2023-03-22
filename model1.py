import torch

from Module.encoder import Encoder
from Module.decoder import Decoder
from Module.other_module import Generator, TransformerEmbedding

class Transformer(torch.nn.Module):
    def __init__(self,src_voc_size, tgt_vob_size, 
                 d_model, n_head, max_size, ffn_hidden, n_layers, dropout, device):
        super(Transformer, self).__init__()
        self.generator = Generator(d_model, tgt_vob_size)
        self.emb_src = TransformerEmbedding(src_voc_size, d_model, max_size, device)
        self.emb_tgt = TransformerEmbedding(tgt_vob_size, d_model, max_size, device)

        self.encoder = Encoder(d_model, ffn_hidden, n_head, n_layers, dropout)
        
        self.decoder = Decoder(tgt_vob_size, d_model, ffn_hidden, n_head, n_layers, dropout)
        
    
    def encode(self, src, src_mask):
        memory = self.encoder(self.emb_src(src), src_mask)
        # print("memory:", memory.size())
        return 
    def decode(self, tgt, memory, src_mask, tgt_mask):
        output = self.decoder(self.emb_tgt(tgt), memory, src_mask, tgt_mask)
        # print("output: ", output.size())
        return output
    
    def forward(self, src, tgt, src_mask, src_tgt_mask, tgt_mask):
        decode = self.decode(tgt, self.encode(src, src_mask), src_tgt_mask, tgt_mask)
        return self._generator(decode)
        
    def _generator(self, x):
        return self.generator(x)
    