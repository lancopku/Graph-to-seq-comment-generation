import os
import csv
import codecs
import yaml
import time
import numpy as np
import nltk
from nltk.translate import bleu_score


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_config(path):
    '''读取config文件'''

    return AttrDict(yaml.load(open(path, 'r')))


def read_datas(filename, trans_to_num=False):
    lines = open(filename, 'r').readlines()
    lines = list(map(lambda x: x.split(), lines))
    if trans_to_num:
        lines = [list(map(int, line)) for line in lines]
    return lines


def save_datas(data, filename, trans_to_str=False):
    if trans_to_str:
        data = [list(map(str, line)) for line in data]
    lines = list(map(lambda x: " ".join(x), data))
    with open(filename, 'w') as f:
        f.write("\n".join(lines))


def logging(file):
    def write_log(s):
        print(s)
        with open(file, 'a') as f:
            f.write(s)

    return write_log


def logging_csv(file):
    def write_csv(s):
        with open(file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(s)

    return write_csv


def format_time(t):
    return time.strftime("%Y-%m-%d-%H:%M:%S", t)


def eval_multi_bleu(references, candidate, log_path):
    ref_1, ref_2, ref_3, ref_4 = [], [], [], []
    for refs, cand in zip(references, candidate):
        ref_1.append(refs[0])
        if len(refs) > 1:
            ref_2.append(refs[1])
        else:
            ref_2.append([])
        if len(refs) > 2:
            ref_3.append(refs[2])
        else:
            ref_3.append([])
        if len(refs) > 3:
            ref_4.append(refs[3])
        else:
            ref_4.append([])
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_path = log_path.strip('/')
    ref_file_1 = log_path + '/reference_1.txt'
    ref_file_2 = log_path + '/reference_2.txt'
    ref_file_3 = log_path + '/reference_3.txt'
    ref_file_4 = log_path + '/reference_4.txt'
    cand_file = log_path + '/candidate.txt'
    with codecs.open(ref_file_1, 'w', 'utf-8') as f:
        for s in ref_1:
            f.write(" ".join(s) + '\n')
    with codecs.open(ref_file_2, 'w', 'utf-8') as f:
        for s in ref_2:
            f.write(" ".join(s) + '\n')
    with codecs.open(ref_file_3, 'w', 'utf-8') as f:
        for s in ref_3:
            f.write(" ".join(s) + '\n')
    with codecs.open(ref_file_4, 'w', 'utf-8') as f:
        for s in ref_4:
            f.write(" ".join(s) + '\n')
    with codecs.open(cand_file, 'w', 'utf-8') as f:
        for s in candidate:
            f.write(" ".join(s).strip() + '\n')

    temp = log_path + "/result.txt"
    command = "perl multi-bleu.perl " + ref_file_1 + " " + ref_file_2 + " " + ref_file_3 + " " + ref_file_4 + "<" + cand_file + "> " + temp
    os.system(command)
    with open(temp) as ft:
        result = ft.read()
    os.remove(temp)
    try:
        bleu = float(result.split(',')[0][7:])
    except ValueError:
        bleu = 0
    return result, bleu


def eval_bleu(reference, candidate, log_path):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_path = log_path.strip('/')
    ref_file = log_path + '/reference.txt'
    cand_file = log_path + '/candidate.txt'
    with codecs.open(ref_file, 'w', 'utf-8') as f:
        for s in reference:
            f.write(" ".join(s) + '\n')
    with codecs.open(cand_file, 'w', 'utf-8') as f:
        for s in candidate:
            f.write(" ".join(s).strip() + '\n')

    temp = log_path + "/result.txt"
    command = "perl multi-bleu.perl " + ref_file + "<" + cand_file + "> " + temp
    os.system(command)
    with open(temp) as ft:
        result = ft.read()
    os.remove(temp)
    try:
        bleu = float(result.split(',')[0][7:])
    except ValueError:
        bleu = 0
    return result, bleu


def write_result_to_file(examples, candidates, log_path):
    assert len(examples) == len(candidates), (len(examples), len(candidates))
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_path = log_path.strip('/')
    log_file = log_path + '/observe_result.tsv'
    with codecs.open(log_file, 'w', 'utf-8') as f:
        for e, cand in zip(examples, candidates):
            f.write("".join(cand).strip() + '\t')
            f.write("".join(e.ori_title).strip() + '\t')
            # f.write(";".join(["".join(sent).strip() for sent in e.ori_content]) + '\t')
            f.write("".join(e.ori_original_content).strip() + '\t')
            # f.write("$$".join(["".join(comment).strip() for comment in e.ori_targets]) + '\t')
            f.write("\n")


def count_entity_num(candidates, tags):
    assert type(candidates) == list and type(tags) == list
    num = 0.
    for c, t in zip(candidates, tags):
        for word in c:
            if word in t:
                num += 1.
    return num / float(len(candidates))


def bow(word_list):
    word_dict = {}
    for word in word_list:
        if word not in word_dict:
            word_dict[word] = 1
        #word_dict[word] += 1
    return word_dict
