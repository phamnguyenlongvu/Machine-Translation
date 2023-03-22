import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List
from torch.nn.functional import pad

def subsequent_mask(q, k, DEVICE):
    len_q, len_k = q.size(1), k.size(1)
    mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(DEVICE)
    return mask

def make_pad_mask(q, k, q_pad_idx=1, k_pad_idx=1):
    len_q, len_k = q.size(1), k.size(1)

    k = k.ne(k_pad_idx).unsqueeze(1).unsqueeze(2)
    k = k.repeat(1, 1, len_q, 1)

    q = q.ne(q_pad_idx).unsqueeze(1).unsqueeze(3)
    q = q.repeat(1, 1, 1, len_k)
    return k & q

def greedy_decode(model, src, src_mask, max_len, start_symbol, device):
    src = src.to(device)
    src_mask = src_mask.to(device)
    print(src.shape)
    print(src_mask)
    memory = model.encode(src.transpose(0, 1), src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len - 1):
        memory = memory.to(device)
        tgt_mask = subsequent_mask(ys.transpose(0, 1), ys.transpose(0, 1), device).type_as(src.data)
        out = model.decode(ys, memory, src_mask, tgt_mask)
        print(out.shape)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == 3:
            break

    return ys

def translate(model: torch.nn.Module, src_sentence: str, text_transform, vocab_transform, device):
    model.eval()
    # src = text_transform['de'](src_sentence).view(-1, 1)
    src = text_transform['de'](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=2, device=device).flatten()
    return " ".join(vocab_transform['en'].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

class Collation:
    def __init__(self, token_transform, vocab_transform, max_padding):
        self.token_transform = token_transform
        self.vocab_transform = vocab_transform
        self.text_transform = {}
        for ln in ['de', 'en']:
            self.text_transform[ln] = self.sequential_transforms(token_transform[ln], #Tokenization
                                                                vocab_transform[ln], #Numericalization
                                                                self.tensor_transform) # Add BOS/EOS and create 
        self.max_padding = max_padding

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
            process_src = self.text_transform['de'](src_sample.rstrip("\n"))
            src_batch.append(pad(
                            process_src,
                            (0, self.max_padding - process_src.size(0)),
                            value=1))
            process_tgt = self.text_transform['en'](tgt_sample.rstrip("\n"))
            tgt_batch.append(pad(
                            process_tgt,
                            (0, self.max_padding - process_tgt.size(0)),
                            value=1))


        src_batch = pad_sequence(src_batch, batch_first= True, padding_value=1)
        tgt_batch = pad_sequence(tgt_batch, batch_first= True, padding_value=1)
        return src_batch, tgt_batch
    
    def get_text_transform(self):
        return self.text_transform
