from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
import json

file = open('config.json')
config = json.load(file)

class PreprocessingData:
    def __init__(self):
        multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
        multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

        self.SRC_LN = config['SRC_LN']
        self.TGT_LN = config['TGT_LN']

        self.token_transform = {}
        self.vocab_transform = {}

        self.token_transform[self.SRC_LN] = get_tokenizer('spacy', language='de_core_news_sm')
        self.token_transform[self.TGT_LN] = get_tokenizer('spacy', language='en_core_web_sm')


    def yield_tokens(self, data_iter: Iterable, language: str) -> List[str]:
        language_index = {self.SRC_LN: 0, self.TGT_LN: 1}

        for data_sample in data_iter:
            yield self.token_transform[language](data_sample[language_index[language]])


    def get_vocab(self):
        UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
        special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        
        for ln in [self.SRC_LN, self.TGT_LN]:
            # Training data Iterator
            train_iter = Multi30k(split='train', language_pair=(self.SRC_LN, self.TGT_LN))
            # Create torchtext's Vocab object
            self.vocab_transform[ln] = build_vocab_from_iterator(self.yield_tokens(train_iter, ln),
                                                            min_freq=1,
                                                            specials=special_symbols,
                                                            special_first=True)
            
        for ln in [self.SRC_LN, self.TGT_LN]:
            self.vocab_transform[ln].set_default_index(UNK_IDX)

        return self.vocab_transform
    
    def get_token(self):
        return self.token_transform