from torch.utils.data import DataLoader
import torch
import torchtext.datasets as datasets
from LoadData import *
from model import *
import torch.nn as nn
from utils import *


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, loss_fn, optimizer, BATCH_SIZE, collate_fn):
    model.train()
    losses = 0
    train_iter = datasets.Multi30k(split='train', language_pair=('de', 'en'))
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    count = 0
    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        count += 1

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, DEVICE)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / count

def evaluate(model, loss_fn, BATCH_SIZE, collate_fn):
    model.eval()
    losses = 0
    count = 0
    val_iter = datasets.Multi30k(split='valid', language_pair=('de', 'en'))
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        count += 1
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, DEVICE)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / count

if __name__ == "__main__":
    torch.manual_seed(0)

    token_transform = PreprocessingData().get_token()
    vocab_transform = PreprocessingData().get_vocab()
    SRC_VOCAB_SIZE = len(vocab_transform['de'])
    TGT_VOCAB_SIZE = len(vocab_transform['en'])
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    BATCH_SIZE = 32
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, 
                                    NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=2)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    from timeit import default_timer as timer
    NUM_EPOCHS = 18

    collate = Collation(token_transform, vocab_transform)
    print("Training .... ")
    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(transformer, loss_fn, optimizer, BATCH_SIZE, collate.collate_fn)
        end_time = timer()
        val_loss = evaluate(transformer, loss_fn, BATCH_SIZE, collate.collate_fn)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))