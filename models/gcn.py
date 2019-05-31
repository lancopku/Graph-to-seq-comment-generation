# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import sys
import os


class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.

    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """

    def forward(self, matrix1, matrix2):
        self.save_for_backward(matrix1, matrix2)
        return torch.mm(matrix1, matrix2)

    def backward(self, grad_output):
        matrix1, matrix2 = self.saved_tensors
        grad_matrix1 = grad_matrix2 = None

        if self.needs_input_grad[0]:
            grad_matrix1 = torch.mm(grad_output, matrix2.t())

        if self.needs_input_grad[1]:
            grad_matrix2 = torch.mm(matrix1.t(), grad_output)

        return grad_matrix1, grad_matrix2


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.Tanh()

    def forward(self, input, adj):
        support = self.linear(input)
        output = self.activation(torch.sparse.mm(adj, support))
        return output


class GCN_Encoder(nn.Module):
    def __init__(self, config, vocab_size, embedding=None):
        super(GCN_Encoder, self).__init__()

        # network configure
        self.num_filter = config.filter_size  # number of conv1d filters
        self.window_size = config.window_size  # conv1d kernel window size
        self.step_size = config.max_sentence_len  # sentence length
        self.nfeat_trans = len(self.window_size) * self.num_filter  # features use concatination of mul and sub

        # word2vec input layer
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=self.num_filter,
                      kernel_size=(self.window_size[0], config.emb_size)),
            nn.ReLU())
        # nn.MaxPool2d(
        #    kernel_size=(self.step_size - self.window_size[0] + 1, 1)))
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=self.num_filter,
                      kernel_size=(self.window_size[1], config.emb_size)),
            nn.ReLU())
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=self.num_filter,
                      kernel_size=(self.window_size[2], config.emb_size)),
            nn.ReLU())

        # graph transformation component
        self.gc1 = GraphConvolution(self.nfeat_trans, config.decoder_hidden_size)
        self.gc2 = GraphConvolution(config.decoder_hidden_size, config.decoder_hidden_size)
        self.out = nn.Linear(self.nfeat_trans, config.decoder_hidden_size)
        self.dropout = config.dropout
        self.tanh = nn.Tanh()

    def forward(self, text, adj):
        # index to w2v
        text_emb = self.embedding(text)

        batch, seq, embed = text_emb.size()
        # input to conv2d: (N, c_in, h_in, w_in), output: (N, c_out, h_out, w_out)
        x = text_emb.contiguous().view(batch, 1, seq, embed)
        x_1 = torch.max(self.encoder_1(x), dim=2)[0]
        x_1 = x_1.view(-1, self.num_filter)
        x_2 = torch.max(self.encoder_2(x), dim=2)[0]
        x_2 = x_2.view(-1, self.num_filter)
        x_3 = torch.max(self.encoder_3(x), dim=2)[0]
        x_3 = x_3.view(-1, self.num_filter)

        # batch, hidden
        x = torch.cat([x_1, x_2, x_3], 1)

        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc2(x, adj))
        # x = self.tanh(self.out(x))

        return x
