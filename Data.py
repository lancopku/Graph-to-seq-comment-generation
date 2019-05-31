import os
import json
import sys
import numpy as np
import torch
import random
import copy
from graph_loader import *
from util.nlp_utils import split_chinese_sentence, remove_stopwords
from util.dict_utils import cosine_sim
from util.utils import bow

PAD = 0
BOS = 1
EOS = 2
UNK = 3
MASK = 4
TITLE = 5
MAX_LENGTH = 100


class Vocab:
    def __init__(self, vocab_file, content_file, vocab_size=50000):
        self._word2id = {'[PADDING]': 0, '[START]': 1, '[END]': 2, '[OOV]': 3, '[MASK]': 4, '_TITLE_': 5}
        self._id2word = ['[PADDING]', '[START]', '[END]', '[OOV]', '[MASK]', '_TITLE_']
        self._wordcount = {'[PADDING]': 1, '[START]': 1, '[END]': 1, '[OOV]': 1, '[MASK]': 1, '_TITLE_': 1}
        if not os.path.exists(vocab_file):
            self.build_vocab(content_file, vocab_file)
        self.load_vocab(vocab_file, vocab_size)
        self.voc_size = len(self._word2id)
        self.UNK_token = 3
        self.PAD_token = 0

    @staticmethod
    def build_vocab(corpus_file, vocab_file):
        word2count = {}
        for line in open(corpus_file):
            words = line.strip().split()
            for word in words:
                if word not in word2count:
                    word2count[word] = 0
                word2count[word] += 1
        word2count = list(word2count.items())
        word2count.sort(key=lambda k: k[1], reverse=True)
        write = open(vocab_file, 'w')
        for word_pair in word2count:
            write.write(word_pair[0] + '\t' + str(word_pair[1]) + '\n')
        write.close()

    def load_vocab(self, vocab_file, vocab_size):
        for line in open(vocab_file):
            term_ = line.strip().split('\t')
            if len(term_) != 2:
                continue
            word, count = term_
            assert word not in self._word2id
            self._word2id[word] = len(self._word2id)
            self._id2word.append(word)
            self._wordcount[word] = int(count)
            if len(self._word2id) >= vocab_size:
                break
        assert len(self._word2id) == len(self._id2word)

    def word2id(self, word):
        if word in self._word2id:
            return self._word2id[word]
        return self._word2id['[OOV]']

    def sent2id(self, sent, add_start=False, add_end=False):
        result = [self.word2id(word) for word in sent]
        if add_start:
            result = [self._word2id['[START]']] + result

        if add_end:
            result = result + [self._word2id['[END]']]
        return result

    def id2word(self, word_id):
        return self._id2word[word_id]

    def id2sent(self, sent_id):
        result = []
        for id in sent_id:
            if id == self._word2id['[END]']:
                break
            elif id == self._word2id['[PADDING]']:
                continue
            result.append(self._id2word[id])
        return result


class Example:
    """
    Each example is one data pair
        src: title (has oov)
        tgt: comment (oov has extend ids if use_oov else has oov)
        memory: tag (oov has extend ids)
    """

    def __init__(self, content, original_content, title, title_index, target, adj, concept, vocab, is_train):
        self.ori_title = title
        self.ori_original_content = original_content
        self.ori_content = content
        if is_train:
            self.ori_target = target
        else:
            self.ori_targets = target
        self.ori_concept = concept
        self.adj = adj
        self.concept = [vocab.word2id(c) for c in concept]
        self.title = vocab.sent2id(title)
        self.title_index = title_index
        if is_train:
            self.target = vocab.sent2id(target, add_start=True, add_end=True)
        self.original_content = vocab.sent2id(self.ori_original_content)
        self.sentence_content = split_chinese_sentence(self.ori_original_content)
        self.sentence_content = [vocab.sent2id(sentence) for sentence in
                                 self.sentence_content]
        self.sentence_content_max_len = min(max([len(c) for c in self.sentence_content]), MAX_LENGTH)
        self.sentence_content, self.sentence_content_mask = Batch.padding(self.sentence_content,
                                                                          self.sentence_content_max_len,
                                                                          limit_length=True)
        self.bow = self.bow(self.original_content)
        self.content = [vocab.sent2id(content) for content in self.ori_content]
        self.content_max_len = min(max([len(c) for c in self.content]), MAX_LENGTH)
        self.content, self.content_mask = Batch.padding(self.content, self.content_max_len, limit_length=True)
        assert len(self.content) == self.adj.size(0)

    def bow(self, content, maxlen=MAX_LENGTH):
        bow = {}
        for word_id in content:
            if word_id not in bow:
                bow[word_id] = 0
            bow[word_id] += 1
        bow = list(bow.items())
        bow.sort(key=lambda k: k[1], reverse=True)
        bow.insert(0, (UNK, 1))
        return [word_id[0] for word_id in bow[:maxlen]]


class Batch:
    """
    Each batch is a mini-batch of data

    """

    def __init__(self, example_list, is_train, model):
        max_len = MAX_LENGTH
        self.model = model
        self.is_train = is_train
        self.examples = example_list
        if model == 'h_attention':
            self.sentence_content = [np.array(e.sentence_content, dtype=np.long) for e in example_list]
            self.sentence_content_mask = [np.array(e.sentence_content_mask, dtype=np.int32) for e in example_list]
            self.sentence_content_len = [len(e.sentence_content) for e in example_list]
            max_sent_num = max(self.sentence_content_len)
            self.sentence_mask, _ = self.padding([[1 for _ in range(d)] for d in self.sentence_content_len],
                                                 max_sent_num, limit_length=False)
        elif model == 'graph2seq':
            self.src_len = [len(e.content) for e in example_list]
            batch_src = [e.content for e in example_list]
            self.src = [np.array(src, dtype=np.long) for src in batch_src]
            self.src_mask = [np.array(e.content_mask, dtype=np.int32) for e in example_list]
            concept_max_len = max([len(e.concept) for e in example_list])
            self.concept_vocab, self.concept_mask = self.padding([e.concept for e in example_list], concept_max_len)
            self.concept = [np.array(e.concept, dtype=np.long) for e in example_list]
            self.title_index = [e.title_index for e in example_list]
            self.adj = [e.adj for e in example_list]
        elif model == 'seq2seq':
            self.title_content_len = self.get_length([e.title + e.original_content for e in example_list], max_len)
            self.title_content, self.title_content_mask = self.padding(
                [e.title + e.original_content for e in example_list],
                max(self.title_content_len))
            self.title_len = self.get_length([e.title for e in example_list], max_len)
            self.title, self.title_mask = self.padding([e.title for e in example_list], max(self.title_len))
        elif model == 'bow2seq':
            self.bow_len = self.get_length([e.bow for e in example_list], max_len)
            self.bow, self.bow_mask = self.padding([e.bow for e in example_list], max(self.bow_len))
        if is_train:
            self.tgt_len = self.get_length([e.target for e in example_list], max_len)
            max_tgt_len = max(self.tgt_len)
            batch_tgt, self.tgt_mask = self.padding([e.target for e in example_list], max_tgt_len)
            self.tgt = np.array(batch_tgt, dtype=np.long)
        self.to_tensor()

    def get_length(self, examples, max_len):
        length = []
        for e in examples:
            if len(e) > max_len:
                length.append(max_len)
            else:
                length.append(len(e))
        assert len(length) == len(examples)
        return length

    def to_tensor(self):
        if self.model == 'graph2seq':
            self.src = [torch.from_numpy(src) for src in self.src]
            self.src_mask = [torch.from_numpy(mask) for mask in self.src_mask]
            self.src_len = torch.from_numpy(np.array(self.src_len, dtype=np.long))
            self.title_index = torch.from_numpy(np.array(self.title_index, dtype=np.long))
            self.concept = [torch.from_numpy(concept) for concept in self.concept]
            self.concept_vocab = torch.from_numpy(np.array(self.concept_vocab, dtype=np.long))
            self.concept_mask = torch.from_numpy(np.array(self.concept_mask, dtype=np.int32))
        elif self.model == 'h_attention':
            self.sentence_content = [torch.from_numpy(src) for src in self.sentence_content]
            self.sentence_content_mask = [torch.from_numpy(mask) for mask in self.sentence_content_mask]
            self.sentence_content_len = torch.from_numpy(np.array(self.sentence_content_len, dtype=np.long))
            self.sentence_mask = torch.from_numpy(np.array(self.sentence_mask, dtype=np.int32))
        elif self.model == 'seq2seq':
            self.title_content = torch.from_numpy(np.array(self.title_content, dtype=np.long))
            self.title_content_len = torch.from_numpy(np.array(self.title_content_len, dtype=np.long))
            self.title_content_mask = torch.from_numpy(np.array(self.title_content_mask, dtype=np.long))
            self.title = torch.from_numpy(np.array(self.title, dtype=np.long))
            self.title_len = torch.from_numpy(np.array(self.title_len, dtype=np.long))
            self.title_mask = torch.from_numpy(np.array(self.title_mask, dtype=np.int32))
        elif self.model == 'bow2seq':
            self.bow = torch.from_numpy(np.array(self.bow, dtype=np.long))
            self.bow_len = torch.from_numpy(np.array(self.bow_len, dtype=np.long))
            self.bow_mask = torch.from_numpy(np.array(self.bow_mask, dtype=np.int32))
        if self.is_train:
            self.tgt = torch.from_numpy(self.tgt)
            self.tgt_len = torch.from_numpy(np.array(self.tgt_len, dtype=np.long))
            self.tgt_mask = torch.from_numpy(np.array(self.tgt_mask, dtype=np.int32))
        # adj 本来就是tensor

    @staticmethod
    def padding(batch, max_len, limit_length=True):
        if limit_length:
            max_len = min(max_len, MAX_LENGTH)
        result = []
        mask_batch = []
        for s in batch:
            l = copy.deepcopy(s)
            m = [1. for _ in range(len(l))]
            l = l[:max_len]
            m = m[:max_len]
            while len(l) < max_len:
                l.append(0)
                m.append(0.)
            result.append(l)
            mask_batch.append(m)
        return result, mask_batch


class DataLoader:
    def __init__(self, config, data_path, batch_size, vocab, adj_type, use_gnn, model, no_train=False, debug=False):
        assert MAX_LENGTH == config.max_sentence_len, (MAX_LENGTH, config.max_sentence_len)
        self.debug = debug
        self.vocab = vocab
        self.batch_size = batch_size
        if not no_train:
            self.train_data = self.read_json(os.path.join(data_path, 'train_graph_features.json'), adj_type,
                                             is_train=True, use_gnn=use_gnn)
            self.train_batches = self.make_batch(self.train_data, batch_size, is_train=True, model=model)
            random.shuffle(self.train_batches)
        self.dev_data = self.read_json(os.path.join(data_path, 'dev_graph_features.json'), adj_type, is_train=False,
                                       use_gnn=use_gnn)
        self.test_data = self.read_json(os.path.join(data_path, 'test_graph_features.json'), adj_type, is_train=False,
                                        use_gnn=use_gnn)
        # self.train_data, self.dev_data, self.test_data = self.split_data(self.data)
        self.dev_batches = self.make_batch(self.dev_data, batch_size, is_train=False, model=model)
        self.test_batches = self.make_batch(self.test_data, batch_size, is_train=False, model=model)

    @staticmethod
    def split_data(data):
        total_num = len(data)
        train = data[:round(0.8 * total_num)]
        dev = data[round(0.8 * total_num):round(0.9 * total_num)]
        test = data[round(0.9 * total_num):]
        return train, dev, test

    def read_json(self, filename, adj_type, is_train=True, use_gnn=False):
        result = []
        for line in open(filename, "r"):
            if len(result) > 100 and self.debug:
                break
            g = json.loads(line)
            if is_train:
                target = g["label"].split()
            else:
                targets = [s.split() for s in g["label"].split("$$")]
            title = g["title"].split()
            original_content = g["text"].split()

            # betweenness = g["g_vertices_betweenness_vec"]
            # pagerank = g["g_vertices_pagerank_vec"]
            # katz = g["g_vertices_katz_vec"]
            concept_names = g["v_names"]
            text_features = g["v_text_features_mat"]
            content = []
            title_index = -1
            for i, val in enumerate(text_features):
                if concept_names[i] == "_TITLE_":
                    title_index = i
                content.append(val.split())
            assert len(concept_names) == len(content), (concept_names, content)

            adj_numsent = g["adj_mat_numsent"]
            # adj_numsent is a list(list)
            adj_numsent = sp.coo_matrix(adj_numsent,
                                        shape=(len(adj_numsent), len(adj_numsent)),
                                        dtype=np.float32)
            adj_numsent = normalize(adj_numsent, use_gnn)
            adj_numsent = sparse_mx_to_torch_sparse_tensor(adj_numsent)
            adj_tfidf = g["adj_mat_tfidf"]
            adj_tfidf = sp.coo_matrix(adj_tfidf,
                                      shape=(len(adj_tfidf), len(adj_tfidf)),
                                      dtype=np.float32)
            adj_tfidf = normalize(adj_tfidf, use_gnn)
            adj_tfidf = sparse_mx_to_torch_sparse_tensor(adj_tfidf)
            if adj_type == 'tfidf':
                adj = adj_tfidf
            elif adj_type == 'numsent':
                adj = adj_numsent
            else:
                print('error!!!')
            assert len(content) == adj.size(0), (len(content), adj.size())
            if is_train:
                e = Example(content, original_content, title, title_index, target, adj, concept_names, self.vocab,
                            is_train)
            else:
                e = Example(content, original_content, title, title_index, targets, adj, concept_names, self.vocab,
                            is_train)
            result.append(e)
        return result

    def make_batch(self, data, batch_size, is_train, model):
        batches = []
        for i in range(0, len(data), batch_size):
            batches.append(Batch(data[i:i + batch_size], is_train, model))
        return batches


def data_stats(fname, is_test):
    content_word_num = []
    content_char_num = []
    title_word_num = []
    title_char_num = []
    comment_word_num = []
    comment_char_num = []
    keyword_num = []
    urls = {}

    for line in open(fname, "r"):
        g = json.loads(line)
        url = g["url"]
        if url not in urls:
            urls[url] = 0
        if is_test:
            targets = [s.split() for s in g["label"].split("$$")]
            urls[url] += len(targets)
            for target in targets:
                comment_word_num.append(len(target))
                comment_char_num.append(len("".join(target)))
        else:
            urls[url] += 1
            target = g["label"].split()
            comment_word_num.append(len(target))
            comment_char_num.append(len("".join(target)))
        title = g["title"].split()
        title_word_num.append(len(title))
        title_char_num.append(len("".join(title)))
        original_content = g["text"].split()
        content_word_num.append(len(original_content))
        content_char_num.append(len("".join(original_content)))

        # betweenness = g["g_vertices_betweenness_vec"]
        # pagerank = g["g_vertices_pagerank_vec"]
        # katz = g["g_vertices_katz_vec"]
        concept_names = g["v_names"]
        keyword_num.append(len(concept_names))
        text_features = g["v_text_features_mat"]
        content = []

        adj_numsent = g["adj_mat_numsent"]
        # adj_numsent is a list(list)
        adj_tfidf = g["adj_mat_tfidf"]
    print('number of documents', len(urls))
    print('number of total comments', sum(list(urls.values())))
    print('average number of comments', np.mean(list(urls.values())))
    content_word_num = np.mean(content_word_num)
    content_char_num = np.mean(content_char_num)
    title_word_num = np.mean(title_word_num)
    title_char_num = np.mean(title_char_num)
    comment_word_num = np.mean(comment_word_num)
    comment_char_num = np.mean(comment_char_num)
    keyword_num = np.mean(keyword_num)
    print(
        'average content word number: %.2f, average content character number: %.2f, average title word number: %.2f, '
        % (content_word_num, content_char_num, title_word_num),
        'average title character numerb: %.2f, average comment word number %.2f, average comment character number %.2f'
        % (title_char_num, comment_word_num, comment_char_num),
        'average keyword number %.2f' % keyword_num)


def eval_bow(feature_file, cand_file):
    stop_words = {word.strip() for word in open('stop_words.txt').readlines()}
    contents = []
    for line in open(feature_file, "r"):
        g = json.loads(line)
        contents.append(remove_stopwords(g["text"].split(), stop_words))
    candidates = []
    for line in open(cand_file):
        words = line.strip().split()
        candidates.append(remove_stopwords(words, stop_words))
    assert len(contents) == len(candidates), (len(contents), len(candidates))
    results = []
    for content, candidate in zip(contents, candidates):
        results.append(cosine_sim(bow(content), bow(candidate)))
    return results, np.mean(results)


def eval_unique_words(cand_file):
    stop_words = {word.strip() for word in open('stop_words.txt').readlines()}
    result = set()
    for line in open(cand_file):
        words = set(line.strip().split())
        result.update(words)
    result = result.difference(stop_words)
    return result


def eval_distinct(cand_file):
    unigram, bigram, trigram = set(), set(), set()
    sentence = set()
    for line in open(cand_file):
        words = line.strip().split()
        sentence.add(line)
        unigram.update(set(words))
        for i in range(len(words) - 1):
            bigram.add((words[i], words[i + 1]))
        for i in range(len(words) - 2):
            trigram.add((words[i], words[i + 1], words[i + 2]))
    return unigram, bigram, trigram, sentence


if __name__ == '__main__':
    '''
    print('entertainment')
    data_stats('./data/train_graph_features.json', False)
    data_stats('./data/dev_graph_features.json', True)
    print('sport')
    data_stats('./sport_data/train_graph_features.json', False)
    data_stats('./sport_data/dev_graph_features.json', True)
    '''
    topic = sys.argv[1]
    cand_log = sys.argv[2]
    # print(eval_bow(os.path.join(topic, 'dev_graph_features.json'), os.path.join(topic, 'log', cand_log, 'candidate.txt'))[1])
    unigram, bigram, trigram, sentence = eval_distinct(os.path.join(topic, 'log', cand_log, 'candidate.txt'))
    print('unigram', len(unigram), 'bigram', len(bigram), 'trigram', len(trigram), 'sentence', len(sentence))
    print(len(eval_unique_words(os.path.join(topic, 'log', cand_log, 'candidate.txt'))))
