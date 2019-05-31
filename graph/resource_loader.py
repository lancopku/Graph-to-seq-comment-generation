# coding:utf-8
import codecs
import os
import pickle
from config import *
from util.nlp_utils import *
from util.tfidf_utils import *
from util.pd_utils import *


def load_IDF():
    contentfile = "../data/seged_content.txt"
    idffile = "../data/IDF.txt"
    if not os.path.exists(idffile):
        gen_idf(contentfile, idffile)
    IDF = load_idf(idffile)
    print("loaded IDF ...")
    return IDF


def load_W2V_VOCAB(language):
    if language=="Chinese":
        print("load w2v vocabulary ...")
        W2V_VOCAB_PKL_FILE = "../../../../data/raw/Tencent-w2v/w2vgood_20170209.vocab.pkl"
        if not os.path.exists(W2V_VOCAB_PKL_FILE):
            W2V = load_w2v("../../../../data/raw/Tencent-w2v/w2vgood_20170209.model",
                           "Tencent", 200)
            W2V_VOCAB = set(W2V.keys())  # must be a set to accelerate remove_OOV
            pickle.dump(W2V_VOCAB, open(W2V_VOCAB_PKL_FILE, "w"))
        else:
            W2V_VOCAB = pickle.load(open(W2V_VOCAB_PKL_FILE, "r"))
        return W2V_VOCAB
    elif language=="English":
        print("load w2v vocabulary ...")
        W2V_VOCAB_PKL_FILE = "../../../../data/raw/Google-w2v/GoogleNews-vectors-negative300.vocab.pkl"
        if not os.path.exists(W2V_VOCAB_PKL_FILE):
            W2V = load_w2v("../../../../data/raw/Google-w2v/GoogleNews-vectors-negative300.bin",
                           "Google", 300)
            W2V_VOCAB = set(W2V.wv.vocab.keys())  # must be a set to accelerate remove_OOV
            pickle.dump(W2V_VOCAB, open(W2V_VOCAB_PKL_FILE, "w"))
        else:
            W2V_VOCAB = pickle.load(open(W2V_VOCAB_PKL_FILE, "r"))
        return W2V_VOCAB


def load_stopwords(language):
    stopwords = []
    file = ""
    if language == "Chinese":
        file = "../../../../data/raw/event-story-cluster/stopwords-zh.txt"
    elif language == "English":
        file = "../../../../data/raw/event-story-cluster/stopwords-en.txt"
    else:
        "Currently only support Chinese or English"
    with codecs.open(file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                stopwords.append(line.strip())
            except Exception:
                pass
    return set(stopwords)
