import spacy
import os, sys
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import torch
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data.distributed import DistributedSampler

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_tokenizers():
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en

def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]

def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        print(from_to_tuple[index])
        yield tokenizer(from_to_tuple[index])

def build_vocabulary(spacy_de, spacy_en):
    datasets.multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
    datasets.multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print("Building German Vocabulary ...")
    train = datasets.Multi30k(split = 'train', language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train, tokenize_de, index=0),
        min_freq=1,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    train = datasets.Multi30k(split = 'train', language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train, tokenize_en, index=1),
        min_freq=1,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt

def load_vocab(spacy_de, spacy_en):
    if not os.path.exists("vocab.pt"):
        # print("Save vocab!")
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        print("Load vocab!")
        vocab_src, vocab_tgt = torch.load("vocab.pt")

    # print("Finished. \nVocab sizes:")
    # print(len(vocab_src), len(vocab_tgt))
    return vocab_src, vocab_tgt

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    # print(transforms)
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: list[int]):
    # print(token_ids)
    return torch.cat((torch.tensor([0]), 
                      torch.tensor(token_ids), 
                      torch.tensor([1])))


    
spacy_de, spacy_en = load_tokenizers()
vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)

def collate_fn(batch):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en) 
    
    text_transform_de = sequential_transforms(tokenize_de, #Tokenization
                                            vocab_src, #Numericalization
                                            tensor_transform) # Add BOS/EOS and create tensor

    text_transform_en = sequential_transforms(tokenize_en, #Tokenization
                                            vocab_tgt, #Numericalization
                                            tensor_transform) # Add BOS/EOS and create tensor
    
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform_de(src_sample.rstrip("\n")))
        tgt_batch.append(text_transform_en(tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=2)
    tgt_batch = pad_sequence(tgt_batch, padding_value=2)
    return src_batch, tgt_batch

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == 2).transpose(0, 1)
    tgt_padding_mask = (tgt == 2).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
    


if __name__ == "__main__":

    train_iter = datasets.Multi30k(split='train', language_pair=('de', 'en'))
    print(train_iter)
    train_dataloader = DataLoader(train_iter, batch_size=4, collate_fn=collate_fn)
    print(train_dataloader)
    for a, b in train_dataloader:
        print(b)
    print("Completed")