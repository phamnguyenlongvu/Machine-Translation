import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0

def generate_square_subsequent_mask(sz, DEVICE):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, DEVICE):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, DEVICE)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == 1).transpose(0, 1)
    tgt_padding_mask = (tgt == 1).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

class Collation:
    def __init__(self, token_transform, vocab_transform):
        self.token_transform = token_transform
        self.vocab_transform = vocab_transform
        self.text_transform = {}
        for ln in ['de', 'en']:
            self.text_transform[ln] = self.sequential_transforms(token_transform[ln], #Tokenization
                                                                vocab_transform[ln], #Numericalization
                                                                self.tensor_transform) # Add BOS/EOS and create 

    def sequential_transforms(self, *transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func
    
    def tensor_transform(self, token_ids: List[int]):
        return torch.cat((torch.tensor([2]),
                        torch.tensor(token_ids),
                        torch.tensor([3])))
    
    def collate_fn(self, batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(self.text_transform['de'](src_sample.rstrip("\n")))
            tgt_batch.append(self.text_transform['en'](tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=1)
        tgt_batch = pad_sequence(tgt_batch, padding_value=1)
        return src_batch, tgt_batch

if __name__ == '__main__':
    print(subsequent_mask(5))
    print(generate_square_subsequent_mask(5, None))