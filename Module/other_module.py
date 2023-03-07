import torch
import math

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_size, device):
        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_size, d_model, device=device)
        self.encoding.requires_grad = False # Dont compute gradient

        pos = torch.arange(0, max_size, device=device)
        pos = pos.float().unsqueeze(dim=1) 

        _2i = torch.arange(0, d_model, step = 2, device=device)

        div_term = torch.exp(_2i * -(math.log(10000.0) / d_model))

        self.encoding[:, 0::2] = torch.sin(pos * div_term)
        self.encoding[:, 1::2] = torch.cos(pos * div_term)

    def forward(self, x):
        _, seq_len = x.size()
        return self.encoding[:seq_len, :]
    

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.attention = Attention()
        self.w_q = torch.nn.Linear(d_model, d_model)
        self.w_k = torch.nn.Linear(d_model, d_model)
        self.w_v = torch.nn.Linear(d_model, d_model)
        self.w_concat = torch.nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # Create q, k, v through Linear 
        print("+++")
        print(q.shape)
        print(k.shape)
        print(v.shape)
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        # Split
        q, k, v = self.split(q), self.split(k), self.split(v)


        # Compute attention
        out, _ = self.attention(q, k, v, mask)
        out = self.concat(out)
        out = self.w_concat(out)
        return out

    def split(self, tensor):
        """
        input: [batch_size, length, d_model]
        return: [batch_size, head, length, d_tensor]
        """
        batch_size, _, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, -1, self.n_head, d_tensor).transpose(1, 2)

        return tensor
    
    def concat(self, tensor):
        """
        input: [batch_size, head, length, d_tensor]
        return: [batch_size, length, d_model]
        """
        batch_size,_ ,length, _ = tensor.size()
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, self.d_model)

        return tensor



class Attention(torch.nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, dropout=None):
        # if mask is not None:
        #     mask = mask.unsqueeze(1)
        d_tensor = k.size(-1)
        print("+++++++++++++")
        print(q.shape)
        print(k.shape)
        print(k.transpose(-2, -1).shape)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_tensor)
        print(scores.size())
        print(mask.shape)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e10)

        scores = self.softmax(scores)

        if dropout is not None:
            scores = dropout(scores)
        return torch.matmul(scores, v), scores
    
class LayerNorm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(d_model))
        self.beta = torch.nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)

        out = (x-mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
    
class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = torch.nn.Linear(d_model, hidden)
        self.linear2 = torch.nn.Linear(hidden, d_model)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class TransformerEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, d_model, max_size, device, dropout=0.1):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_size, device)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.dropout(tok_emb + pos_emb)

