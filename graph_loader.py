# coding=utf-8
import numpy as np
import scipy.sparse as sp
import json
import torch
from sklearn.preprocessing import OneHotEncoder
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
current_dir = os.getcwd()
sys.path.insert(0, parent_dir)
from util.nlp_utils import *
from sklearn.metrics import f1_score


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize(adj, use_gnn):
    """Row-normalize sparse matrix"""
    if use_gnn:
        adj = adj - np.identity(adj.shape[0])
        adj = adj.astype(int) > 0
        rowsum = np.array(adj.sum(1))
        rowsum[rowsum == 0] = 1.
        adj = adj / rowsum
        return sp.coo_matrix(adj)
    else:
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def bc_accuracy(output, labels):
    preds = output >= 0.5
    preds = preds.float().view(-1)
    correct = preds.eq(labels.float()).double()
    correct = correct.sum()
    return correct / len(labels)


def f1score(output, label):
    preds = output >= 0.5
    preds = preds.float().view(-1)
    result = f1_score(label.numpy(), preds.numpy(), pos_label=1, average="binary")
    return result


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_graph_data(path, word_to_ix, max_len):
    """
    :param path: graph data file path
    :param word_to_ix: transform the word sentence the sequence
    :param max_len: maximum length of text extracted from one document in each vertice
    :return:
    """
    target_list = []
    title_list = []
    g_text_features_list = []
    g_vertices_betweenness_list = []
    g_vertices_pagerank_list = []
    g_vertices_katz_list = []
    adjs_numsent_list = []
    adjs_tfidf_list = []
    num_samples = 0

    fin = open(path, "r")
    for line in fin:
        g = json.loads(line)
        target_list.append(right_pad_zeros_1d([word_to_ix[w] for w in g["label"].split()], max_len))
        title_list.append(right_pad_zeros_1d([word_to_ix[w] for w in g["title"].split()], max_len))
        g_vertices_betweenness_list.append(g["g_vertices_betweenness_vec"])
        g_vertices_pagerank_list.append(g["g_vertices_pagerank_vec"])
        g_vertices_katz_list.append(g["g_vertices_katz_vec"])
        features = g["v_text_features_mat"]
        word_idxs = []
        for j, val in enumerate(features):
            val = val.split()
            sent_idx = right_pad_zeros_1d([word_to_ix[w] for w in val], max_len)
            word_idxs.append(sent_idx)
        word_idxs = torch.LongTensor(word_idxs)
        g_text_features_list.append(word_idxs)

        adj_numsent = g["adj_mat_numsent"]
        adj_numsent = sp.coo_matrix(adj_numsent,
                                    shape=(len(adj_numsent), len(adj_numsent)),
                                    dtype=np.float32)
        adj_numsent = normalize(adj_numsent)
        adj_numsent = sparse_mx_to_torch_sparse_tensor(adj_numsent)
        adjs_numsent_list.append(adj_numsent)
        adj_tfidf = g["adj_mat_tfidf"]
        adj_tfidf = sp.coo_matrix(adj_tfidf,
                                  shape=(len(adj_tfidf), len(adj_tfidf)),
                                  dtype=np.float32)
        adj_tfidf = normalize(adj_tfidf)
        adj_tfidf = sparse_mx_to_torch_sparse_tensor(adj_tfidf)
        adjs_tfidf_list.append(adj_tfidf)

        num_samples = num_samples + 1
    targets = torch.LongTensor(target_list)

    g_vertices_betweenness = [torch.FloatTensor(np.array(x[:-1])) for x in g_vertices_betweenness_list]  # !!!!!!!!
    g_vertices_pagerank = [torch.FloatTensor(np.array(x[:-1])) for x in g_vertices_pagerank_list]
    g_vertices_katz = [torch.FloatTensor(np.array(x[:-1])) for x in g_vertices_katz_list]

    return adjs_numsent_list, adjs_tfidf_list, \
           g_text_features_list, g_vertices_betweenness, g_vertices_pagerank, \
           g_vertices_katz, targets, title_list
