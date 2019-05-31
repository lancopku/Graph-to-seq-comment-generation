import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from collections import OrderedDict
import numpy as np
import sys
import copy
import os


class Memory_Network(nn.Module):
    def __init__(self, config, vocab_size, word_level_model, graph_model, memory_layer=2, embedding=None):
        super(Memory_Network, self).__init__()
        assert word_level_model in ['bert', 'memory', 'word']
        assert graph_model in ['GCN', 'GNN', 'none']
        self.word_level_model = word_level_model

        # network configure
        self.memory_layer = memory_layer

        # word2vec input layer
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)

        # graph transformation component
        self.graph_model = graph_model
        if word_level_model == 'bert':
            self.bert = models.bert.BERT(config.head_num, config.emb_size, config.dropout, config.emb_size, vocab_size,
                                         memory_layer, config.max_sentence_len + 1, self.embedding)
        # plus 1 here in case for the extra keyword
        elif word_level_model == 'memory':
            self.memory_attn = models.memory_attention(config.emb_size, config.emb_size, config.decoder_hidden_size)
        if self.graph_model == 'GCN' or self.graph_model == 'GNN':
            self.gcn = models.GraphConvolution(config.emb_size, config.emb_size)
        self.out = nn.Linear(config.emb_size, config.decoder_hidden_size)
        self.dropout = config.dropout
        self.tanh = nn.Tanh()

    def forward(self, text, text_mask, keyword, adj):
        text_emb = self.embedding(text)  # concept, sentence, emb
        keyword_emb = self.embedding(keyword)
        if self.word_level_model == 'bert':
            x = torch.cat([keyword_emb.unsqueeze(1), text_emb], 1)
            mask = torch.cat([torch.ones_like(keyword, dtype=torch.int32).unsqueeze(1), text_mask], 1)
            x = self.bert.position_encode(x, mask)[:, 0]
        elif self.word_level_model == 'memory':
            last_query = keyword_emb
            # TODO: use different memory cells for different layers
            for l in range(self.memory_layer):
                query, _ = self.memory_attn(last_query, text_emb)
                query = query + last_query  # use highway in memory network
                last_query = query
            x = query
        else:
            x = keyword_emb
        if self.graph_model == 'GCN' or self.graph_model == 'GNN':
            x = self.gcn(x, adj) + x  # maybe use highway
        x = self.tanh(self.out(x))
        return x
