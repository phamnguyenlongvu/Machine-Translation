from torch.utils.data import DataLoader
import torch
import torchtext.datasets as datasets
from LoadData import *
from model import *
import torch.nn as nn
from utils import *
from model1 import *
import argparse
import matplotlib.pyplot as plt


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, loss_fn, optimizer, BATCH_SIZE, collate_fn):
    model.train()
    losses = 0
    train_iter = datasets.Multi30k(split='train', language_pair=('de', 'en'))
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:, :-1]

        src_mask = make_pad_mask(src, src)
        src_tgt_mask = make_pad_mask(tgt_input, src)
        tgt_mask = make_pad_mask(tgt_input, tgt_input) * subsequent_mask(tgt_input, tgt_input, DEVICE)

        logits = model(src, tgt_input, src_mask, src_tgt_mask, tgt_mask)

        optimizer.zero_grad()

        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        print(logits.reshape(-1, logits.shape[-1]).shape)
        print(tgt_out.reshape(-1).shape)
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))

def evaluate(model, loss_fn, BATCH_SIZE, collate_fn):
    model.eval()
    losses = 0
    val_iter = datasets.Multi30k(split='valid', language_pair=('de', 'en'))
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:, :-1]

        src_mask = make_pad_mask(src, src)
        src_tgt_mask = make_pad_mask(tgt_input, src)
        tgt_mask = make_pad_mask(tgt_input, tgt_input) * subsequent_mask(tgt_input, tgt_input, DEVICE)

        logits = model(src, tgt_input, src_mask, src_tgt_mask, tgt_mask)
        
        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))

def curve(loss):
    plt.figure(figsize=(8,8))
    plt.plot(loss['train'])
    plt.plot(loss['val'])
    plt.title('Loss curve')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def train():
    token_transform = PreprocessingData().get_token()
    vocab_transform = PreprocessingData().get_vocab()
    SRC_VOCAB_SIZE = len(vocab_transform['de'])
    TGT_VOCAB_SIZE = len(vocab_transform['en'])
    EMB_SIZE = 512
    NHEAD = 4
    FFN_HID_DIM = 512
    BATCH_SIZE = 32
    NUM_LAYERS = 3
    DROP_PROB = 0.1
    MAX_SIZE = 100

    transformer = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, EMB_SIZE, NHEAD, MAX_SIZE, FFN_HID_DIM, NUM_LAYERS, DROP_PROB, device=DEVICE)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=1)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    from timeit import default_timer as timer
    NUM_EPOCHS = 10

    collate = Collation(token_transform, vocab_transform, MAX_SIZE)
    loss = {
        'train': [],
        'val': []
    }
    print("Training .... ")
    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(transformer, loss_fn, optimizer, BATCH_SIZE, collate.collate_fn)
        loss['train'].append(train_loss)
        end_time = timer()
        val_loss = evaluate(transformer, loss_fn, BATCH_SIZE, collate.collate_fn)
        loss['val'].append(val_loss)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

    curve(loss)

    torch.save(transformer.state_dict(), 'Saved/first.pt')

def test():
    token_transform = PreprocessingData().get_token()
    vocab_transform = PreprocessingData().get_vocab()
    SRC_VOCAB_SIZE = len(vocab_transform['de'])
    TGT_VOCAB_SIZE = len(vocab_transform['en'])
    EMB_SIZE = 512
    NHEAD = 4
    FFN_HID_DIM = 512
    BATCH_SIZE = 32
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    collation = Collation(token_transform, vocab_transform)
    text_transform = collation.get_text_transform()
    model = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, EMB_SIZE, NHEAD, 500, FFN_HID_DIM, 3, 0.1, device=DEVICE).to(device=DEVICE)
    
    print("Load model completed ....")
    src_sentence = 'Eine Gruppe von Menschen steht vor einem Iglu .'
    print(translate(model, src_sentence, text_transform, vocab_transform, DEVICE))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train')
    arg = parser.parse_args()
    if arg.mode == 'train':
        train()
    elif arg.mode == 'test':
        test()
    # print(arg.mode)