import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import random
import time
import yaml
from Data import *
from tqdm import tqdm

# For plots
# import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_config(path):
    '''读取config文件'''

    return AttrDict(yaml.load(open(path, 'r')))


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class BERT(nn.Module):
    def __init__(self, head_num, d_model, dropout, d_ff, src_vocab, n_layers, max_senlen, word_emb=None):
        super(BERT, self).__init__()
        c = copy.deepcopy
        self.d_model = d_model
        attn = MultiHeadedAttention(head_num, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.position_emb = nn.Embedding(max_senlen, d_model)
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n_layers)
        if word_emb is not None:
            self.word_emb = word_emb
        else:
            self.word_emb = Embeddings(d_model, src_vocab, word_emb)
        self.generator = Generator(d_model, self.word_emb)
        self.init()

    def init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def masked_language_model(self, src_word, src_type, src_mask, tgt_position, tgt_result, criterion):
        # TODO: add concept cluster
        # get the final layer of hidden states
        hidden = self.forward(src_word, src_type, src_mask)
        # time step first instead of batch first
        # get the corresponding representation of targets
        batch_size = src_word.size(0)
        batch_nums = torch.arange(0, batch_size)
        batch_nums = torch.unsqueeze(batch_nums, 1)
        # hidden = torch.index_select(hidden, dim=1, index=torch.unsqueeze(tgt_position, 1))
        # hidden = torch.gather(hidden, dim=1, index=torch.unsqueeze(tgt_position, 1))
        hidden = torch.squeeze(hidden[(batch_nums, torch.unsqueeze(tgt_position, 1))], 1)
        '''
        one_hot = torch.zeros(hidden.size())
        one_hot = one_hot.to(hidden.device)
        one_hot.scatter_(1, tgt_position, 1)
        '''
        prob = self.generator(hidden)
        loss = criterion(prob, tgt_result)
        pred = torch.argmax(prob, -1)
        return pred, loss

    def forward(self, src_word, src_mask):
        return self.encoder(self.word_emb(src_word), src_mask)[0]

    def encode(self, src_hidden, src_mask):
        return self.encoder(src_hidden, src_mask)

    def position_encode(self, src_word_emb, src_mask):
        batch_size, seq_len, _ = src_word_emb.size()
        pos_indices = torch.arange(seq_len).repeat(batch_size, 1).to(src_word_emb.device)
        pos_emb = self.position_emb(pos_indices)
        return self.encoder(src_word_emb + pos_emb, src_mask)[0]


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        if mask.dim() == 2:
            seq_len = mask.size(1)
            mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
            assert mask.size() == (x.size(0), seq_len, seq_len)
        for layer in self.layers:
            x, attn = layer(x, mask)
        return self.norm(x), attn


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity we apply the norm first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer function that maintains the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of two sublayers, self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        attn = self.self_attn.get_attn(x, x, x, mask)
        return self.sublayer[1](x, self.feed_forward), attn


def attention(query, key, value, mask=None, dropout=0.0):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask.eq(0), -1e9)
    p_attn = F.softmax(scores, dim=-1)
    # (Dropout described below)
    p_attn = F.dropout(p_attn, p=dropout)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.p = dropout
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # clone linear for 4 times, query, key, value, output
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
            assert mask.dim() == 4  # batch, head, seq_len, seq_len
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => head * d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.p)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def get_attn(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
            assert mask.dim() == 4  # batch, head, seq_len, seq_len
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => head * d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.p)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        _, self.attn = attention(x, x, x, mask=torch.squeeze(mask, 1), dropout=self.p)
        return self.attn


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # Torch linears have a `b` by default.
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, emb=None):
        super(Embeddings, self).__init__()
        if emb is not None:
            self.lut = nn.Embedding.from_pretrained(emb)
        else:
            self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        self.criterion = nn.CosineEmbeddingLoss(margin=0.5)

    def compare(self, x1, x2, label):
        return self.criterion(self.forward(x1), self.forward(x2), label)

    def reverse_mul(self, x):
        return F.linear(x, self.lut.weight)

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    # TODO: use learnable position encoding
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Generator(nn.Module):
    "Standard generation step. (Not described in the paper.)"

    def __init__(self, d_model, word_emb):
        super(Generator, self).__init__()
        self.ff = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())
        self.emb = word_emb

    def forward(self, x):
        return self.emb.reverse_mul(self.ff(x))


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def train_epoch(train_data, model, criterion, optim, args):
    model.train()
    total_loss = 0.
    for batch in tqdm(train_data, disable=not args.verbose):
        optim.optimizer.zero_grad()
        masked_keyword, masked_keyword_mask, masked_keyword_pos, masked_keyword_result = batch.masked_keyword_id, batch.masked_keyword_mask, batch.masked_keyword_pos, batch.masked_keyword_result
        memory_type = batch.memory_type
        if use_cuda:
            masked_keyword, masked_keyword_mask, masked_keyword_pos, masked_keyword_result = masked_keyword.cuda(), masked_keyword_mask.cuda(), masked_keyword_pos.cuda(), masked_keyword_result.cuda()
            memory_type = memory_type.cuda()
        _, loss = model.masked_language_model(masked_keyword, memory_type, masked_keyword_mask, masked_keyword_pos,
                                              masked_keyword_result, criterion)
        loss /= torch.sum(masked_keyword_mask).float()
        loss.backward()
        optim.step()
        total_loss += loss.data.item()

    print('total loss %.2f' % total_loss)


def valid_epoch(valid_data, model, criterion, args):
    model.eval()
    total_num = 0
    total_right = 0
    total_loss = 0.
    for batch in valid_data:
        masked_keyword, masked_keyword_mask, masked_keyword_pos, masked_keyword_result = batch.masked_keyword_id, batch.masked_keyword_mask, batch.masked_keyword_pos, batch.masked_keyword_result
        memory_type = batch.memory_type
        if use_cuda:
            masked_keyword, masked_keyword_mask, masked_keyword_pos, masked_keyword_result = masked_keyword.cuda(), masked_keyword_mask.cuda(), masked_keyword_pos.cuda(), masked_keyword_result.cuda()
            memory_type = memory_type.cuda()
        pred, loss = model.masked_language_model(masked_keyword, memory_type, masked_keyword_mask, masked_keyword_pos,
                                                 masked_keyword_result, criterion)
        total_loss += loss.data.item()
        right_num = torch.sum(
            pred.eq(masked_keyword_result))  # for each case, there is only one target, so do not need mask
        total_right += right_num
        total_num += masked_keyword.size(0)
    accuracy = total_right.float() / float(total_num)
    return accuracy.data.item(), total_loss


def prepare_data(dataloader, prepare_train, prepare_dev, train_emb):
    if prepare_train:
        for batch in dataloader.train_batches:
            batch.make_mask_batch()
            if prepare_dev:  # indicates that this is the first epoch, just a trick!
                batch.to_tensor()
            batch.masked_to_tensor()
    if prepare_dev:
        for batch in dataloader.dev_batches:
            batch.make_mask_batch()
            batch.to_tensor()
            batch.masked_to_tensor()
    if train_emb:
        batches = []
        data = dataloader.build_ag_data()
        for i in range(0, len(data), dataloader.batch_size):
            batch = data[i:i + dataloader.batch_size]
            x1 = torch.from_numpy(np.array([d[0] for d in batch], dtype=np.long))
            x2 = torch.from_numpy(np.array([d[1] for d in batch], dtype=np.long))
            y = torch.from_numpy(np.array([d[2] for d in batch], dtype=np.float32))
            batches.append((x1, x2, y))
        return batches


def train_entity_emb(batches, model, optim, args):
    model.train()
    total_loss = 0.
    for batch in tqdm(batches, disable=not args.verbose):
        x1, x2, y = batch
        if use_cuda:
            x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()
        optim.optimizer.zero_grad()
        loss = model.word_emb.compare(x1, x2, y)
        total_loss += loss.data.item()
        loss.backward()
        optim.step()
    print('total loss %.2f' % total_loss)


def train_iters(epoch_num, dataloader, model, criterion, args):
    optim = get_std_opt(model)
    best_acc = 0.
    emb_data = None
    for i in range(1, epoch_num + 1):
        print('epoch %d' % i)
        if i == 1:
            emb_data = prepare_data(dataloader, True, True, True)
        else:
            prepare_data(dataloader, True, False, False)
        train_entity_emb(emb_data, model, optim, args)
        train_epoch(dataloader.train_batches, model, criterion, optim, args)
        acc, loss = valid_epoch(dataloader.dev_batches, model, criterion, args)
        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), str(best_acc) + '_bert_model.ckpt')
        print('accuracy: %.3f' % acc * 100, flush=True)
        print('eval loss: %.3f' % loss, flush=True)
    print('best accuracy: %.3f' % best_acc * 100, flush=True)


def pretrain_bert(args):
    # data
    config = read_config('config.yaml')
    print('loading data...\n')
    start_time = time.time()
    keyword_vocab = Vocab(config.keyword_vocab, 90000)
    vocab = Vocab(config.vocab)
    dataloader = DataLoader(config.big_data, config.batch_size, vocab, keyword_vocab, use_ag=args.use_ag, use_oov=False,
                            is_bert=True)
    print('loading time cost: %.3f' % (time.time() - start_time))
    model = BERT(config.head_num, config.decoder_hidden_size, config.dropout, config.decoder_hidden_size,
                 keyword_vocab.voc_size, config.type_num, config.bert_num_layers)
    if use_cuda:
        model.cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=3, reduction='sum')
    train_iters(config.pretrain_epoch, dataloader, model, criterion, args)


if __name__ == '__main__':
    args = None
    pretrain_bert(args)
