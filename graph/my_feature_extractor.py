# coding=utf-8
import json
import csv
import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial
from graph_tool.all import *
from ccig import *
from sentence_score import *
from sentence_pair_score import *
from util.nlp_utils import split_chinese_sentence


def assign_graph_label(g, label):
    """
    Assign label to a whole graph.
    :param g: aligned concept/sentence interaction graph.
    :param label: graph label value.
    :return: graph with label.
    """
    g.graph_properties["label"] = g.new_graph_property("string")
    g.graph_properties["label"] = label
    return g


def assign_graph_node_features(g, sentences, title=None):
    """
    Calculate features for graph nodes, and assign it as node property.
    :param g: aligned concept/sentence interaction graph.
    :param sentences1: left list of sentences.
    :param sentences2: right list of sentences.
    :param title1: left title.
    :param title2: right title.
    :param feature_type: types of features. "manual" or "w2v" or "hybrid".
    :return: graph with node feature property.
    """
    vprop_features = g.new_vertex_property("string")

    for v in g.vertices():
        idxs = list(set(g.vertex_properties["sentidxs"][v]))
        text = [sentences[i] for i in idxs]
        text = " ".join(text)
        if g.vertex_properties["name"][v] == TITLE_VERTEX_NAME:
            if title is not None:
                text = title
        vprop_features[v] = text

    g.vertex_properties["text"] = vprop_features
    return g


def text2graph_worker(i, df, use_cd=True, print_fig=False):
    """
    """
    # df is pandas.DataFrame, loc[i] is the data of index i
    text = split_chinese_sentence(df.loc[i]['content'])
    url = split_chinese_sentence(df.loc[i]['url'])

    concepts = (str(df.loc[i]['concepts']).split(","))
    concepts = list(set(concepts))
    # concepts1 = list(set(remove_stopwords(text1, STOPWORDS).split()))
    # concepts2 = list(set(remove_stopwords(text2, STOPWORDS).split()))

    title = df.loc[i]['title']
    label = df.loc[i]['label']

    g = construct_ccig(text, concepts, title, use_cd)
    if g is None:
        return None
    g.graph_properties["title"] = g.new_graph_property("string")
    g.graph_properties["title"] = title
    g.graph_properties["text"] = g.new_graph_property("string")
    g.graph_properties["text"] = df.loc[i]['content']
    g.graph_properties["url"] = g.new_graph_property("string")
    g.graph_properties["url"] = url

    g = assign_graph_node_features(g, text, title)
    # g = assign_graph_edge_weights(g, sentences1, sentences2)
    g = assign_graph_label(g, label)
    # g = assign_graph_features(g, category, time1, time2, sentences1, sentences2, title1, title2)
    if print_fig:
        print_ccig(g, sentences)
    print(i)
    return g


def extract_graphs_from_data(infile, use_cd=True, parallel=True, extract_range=None, print_fig=False):
    """
    """
    df = pd.read_csv(infile, sep="|", quotechar='\"', quoting=csv.QUOTE_ALL)  # , nrows=100)
    gs = []
    print("parallel extract graph features")
    partial_worker = partial(text2graph_worker, df=df, use_cd=use_cd, print_fig=print_fig)
    if extract_range is None:
        extract_range = range(df.shape[0])

    if parallel:
        pool = mp.Pool(processes=mp.cpu_count())
        gs = pool.map(partial_worker, extract_range)
    else:
        for i in extract_range:
            gs.append(partial_worker(i))  # NOTICE: non-parallel can help debug

    return gs


def save_graph_features_to_file(gs, outfile, draw_fig=False):
    """
    Save graphs' topology and node features to a file.
    :param gs: list of graphs.
    :param outfile: output file name that store graphs.
    """
    f = open(outfile, "w")
    i = 0
    for g in gs:
        print("graph " + str(i))
        if g is None:
            print("Graph is None")
            continue
        if draw_fig:
            draw_ccig(g, "ccig_" + str(i) + ".png")

        # used to save graph features
        dict_g = {}

        # graph label
        dict_g["label"] = g.graph_properties["label"]
        dict_g["title"] = g.graph_properties["title"]
        dict_g["text"] = g.graph_properties["text"]
        dict_g["url"] = g.graph_properties["url"]
        # graph features
        # dict_g["v_text_features_vec"] = g.vertex_properties["text"].get_array().tolist()
        v_names = []
        v_features_mat = []
        num_v = g.num_vertices()
        for idx in range(num_v):
            # NOTICE: how to convert vector<string> to list ... a big 坑
            v_features_mat.append(g.vertex_properties["text"][g.vertex(idx)])
            v_names.append(g.vertex_properties["name"][g.vertex(idx)])
        dict_g["v_text_features_mat"] = v_features_mat
        dict_g["v_names"] = v_names

        # graph vertices scores matrix
        dict_g["g_vertices_betweenness_vec"] = []
        dict_g["g_vertices_pagerank_vec"] = []
        dict_g["g_vertices_katz_vec"] = []
        for idx in range(num_v):
            dict_g["g_vertices_betweenness_vec"].append(g.vertex_properties["betweenness"][g.vertex(idx)])
            dict_g["g_vertices_pagerank_vec"].append(g.vertex_properties["pagerank"][g.vertex(idx)])
            dict_g["g_vertices_katz_vec"].append(g.vertex_properties["katz"][g.vertex(idx)])

        # graph weighted adjacent matrices
        adj_mat_numsent = np.identity(num_v)
        adj_mat_tfidf = np.identity(num_v)
        for e in g.edges():
            vsource_idx = g.vertex_index[e.source()]
            vtarget_idx = g.vertex_index[e.target()]

            w_numsent = g.edge_properties["weight_numsent"][e]
            adj_mat_numsent[vsource_idx, vtarget_idx] = w_numsent
            adj_mat_numsent[vtarget_idx, vsource_idx] = w_numsent

            w_tfidf = g.edge_properties["weight_tfidf"][e]
            adj_mat_tfidf[vsource_idx, vtarget_idx] = w_tfidf
            adj_mat_tfidf[vtarget_idx, vsource_idx] = w_tfidf

        dict_g["adj_mat_numsent"] = adj_mat_numsent.tolist()
        dict_g["adj_mat_tfidf"] = adj_mat_tfidf.tolist()

        json_out = json.dumps(dict_g, ensure_ascii=False)  # to dump Chinese
        f.write(json_out)
        f.write("\n")
        i = i + 1
    f.close()


def dataset2featurefile(infile, outfile, use_cd=False,
                        parallel=True, extract_range=None, draw_fig=False, print_fig=False):
    """
    """
    gs = extract_graphs_from_data(infile, use_cd, parallel, extract_range, print_fig)
    save_graph_features_to_file(gs, outfile, draw_fig)


if __name__ == "__main__":
    print('building train graph')
    dataset2featurefile("../data/train_graph_data.csv", "../data/train_graph_features.json")
    print('building dev graph')
    dataset2featurefile("../data/dev_graph_data.csv", "../data/dev_graph_features.json")
    print('building test graph')
    dataset2featurefile("../data/test_graph_data.csv", "../data/test_graph_features.json")
