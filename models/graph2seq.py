import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import models
from torch.nn.utils.rnn import pad_sequence
from Data import *

import numpy as np


class graph2seq(nn.Module):
    def __init__(self, config, vocab, use_cuda, use_copy, use_bert, word_level_model, graph_model, pretrain=None):
        super(graph2seq, self).__init__()
        self.word_level_model = word_level_model
        self.vocab = vocab
        self.vocab_size = vocab.voc_size
        if pretrain is not None:
            self.embedding = pretrain['emb']
        else:
            self.embedding = nn.Embedding(self.vocab_size, config.emb_size)
        # self.encoder = models.GCN_Encoder(config, self.vocab_size, embedding=self.embedding)
        self.use_copy = use_copy
        self.use_bert = use_bert
        if use_bert:
            self.bert_encoder = models.bert.BERT(config.head_num, config.decoder_hidden_size, config.dropout,
                                                 config.decoder_hidden_size, self.vocab_size, config.num_layers,
                                                 config.max_sentence_len)
        self.encoder = models.Memory_Network(config, self.vocab_size, word_level_model, graph_model,
                                             embedding=self.embedding)
        if use_copy:
            self.decoder = models.pointer_decoder(config, self.vocab_size, embedding=self.embedding)
        else:
            self.decoder = models.rnn_decoder(config, self.vocab_size, embedding=self.embedding)
        self.state_wc = nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size * config.num_layers)
        self.state_wh = nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size * config.num_layers)
        self.tanh = nn.Tanh()
        self.config = config
        self.criterion = models.criterion(self.vocab_size, use_cuda)
        self.log_softmax = nn.LogSoftmax()

    def compute_loss(self, hidden_outputs, targets):
        assert hidden_outputs.size(1) == targets.size(1) and hidden_outputs.size(0) == targets.size(0)
        outputs = hidden_outputs.contiguous().view(-1, hidden_outputs.size(2))
        targets = targets.contiguous().view(-1)
        weight = torch.ones(outputs.size(-1))
        weight[PAD] = 0
        weight[UNK] = 0
        weight = weight.to(outputs.device)
        loss = F.nll_loss(torch.log(outputs), targets, weight=weight, reduction='sum')
        pred = outputs.max(dim=1)[1]
        num_correct = pred.data.eq(targets.data).masked_select(targets.ne(PAD).data).sum()
        num_total = targets.ne(PAD).data.sum()
        loss = loss.div(num_total.float())
        acc = num_correct.float() / num_total.float()
        return loss, acc

    def encode(self, contents, contents_mask, concepts, concept_mask, title_index, adjs):
        contexts = []
        states = []
        for content, content_mask, concept, t_index, adj in zip(contents, contents_mask, concepts, title_index, adjs):
            context = self.encoder(content, content_mask, concept, adj)
            # state = context.max(0)[0]  # max pooling
            state = context[t_index, :]
            contexts.append(context)
            states.append(state)
        contexts = pad_sequence(contexts, batch_first=True)
        if self.use_bert:
            contexts, attn = self.bert_encoder.encode(contexts, concept_mask)
        state = torch.stack(states, 0)
        return contexts, state, attn

    def build_init_state(self, state, num_layers):
        c0 = self.tanh(self.state_wc(state)).contiguous().view(-1, num_layers, self.config.decoder_hidden_size)
        h0 = self.tanh(self.state_wh(state)).contiguous().view(-1, num_layers, self.config.decoder_hidden_size)
        c0 = c0.transpose(1, 0)
        h0 = h0.transpose(1, 0)
        return c0, h0

    def forward(self, batch, use_cuda):
        src, adjs, concept, concept_mask, concept_vocab = batch.src, batch.adj, batch.concept, batch.concept_mask, batch.concept_vocab
        src_mask = batch.src_mask
        title_index = batch.title_index
        tgt, tgt_len, tgt_mask = batch.tgt, batch.tgt_len, batch.tgt_mask
        if use_cuda:
            tgt = tgt.cuda()
            src = [s.cuda() for s in src]
            src_mask = [s.cuda() for s in src_mask]
            adjs = [adj.cuda() for adj in adjs]
            concept = [c.cuda() for c in concept]
            concept_mask = concept_mask.cuda()
            title_index = title_index.cuda()
        contexts, state, attns = self.encode(src, src_mask, concept, concept_mask, title_index, adjs)
        c0, h0 = self.build_init_state(state, self.config.num_layers)
        if self.use_copy:
            outputs, final_state, attns, p_gens = self.decoder(tgt[:, :-1], (c0, h0), contexts, concept_mask,
                                                               title_index,
                                                               max_oov=0, extend_vocab=concept_vocab)
        else:
            outputs, final_state, attns = self.decoder(tgt[:, :-1], (c0, h0), contexts)
            outputs = F.softmax(outputs, -1)
        return outputs

    def sample(self, batch, use_cuda):
        src, adjs, concept, concept_mask = batch.src, batch.adj, batch.concept, batch.concept_mask
        src_mask = batch.src_mask
        title_index = batch.title_index
        concept_vocab = batch.concept_vocab
        if use_cuda:
            src = [s.cuda() for s in src]
            src_mask = [s.cuda() for s in src_mask]
            adjs = [adj.cuda() for adj in adjs]
            concept = [c.cuda() for c in concept]
            concept_mask = concept_mask.cuda()
            title_index = title_index.cuda()
        contexts, state, attns = self.encode(src, src_mask, concept, concept_mask, title_index, adjs)
        bos = torch.ones(len(src)).long().fill_(self.vocab.word2id('[START]'))
        if use_cuda:
            bos = bos.cuda()
        c0, h0 = self.build_init_state(state, self.config.num_layers)
        if self.use_copy:
            sample_ids, final_outputs, p_gens = self.decoder.sample([bos], (c0, h0), contexts, concept_mask,
                                                                    title_index,
                                                                    max_oov=0, extend_vocab=concept_vocab)
        else:
            sample_ids, final_outputs = self.decoder.sample([bos], (c0, h0), contexts)
        probs, attns_weight = final_outputs
        alignments = attns_weight.max(dim=2)[1]

        return sample_ids, attns

    # TODO: fix beam search
    def beam_sample(self, batch, use_cuda, beam_size=1):
        src, adjs, concept, concept_mask = batch.src, batch.adj, batch.concept, batch.concept_mask
        src_mask = batch.src_mask
        concept_vocab = batch.concept_vocab
        title_index = batch.title_index
        if use_cuda:
            src = [s.cuda() for s in src]
            src_mask = [s.cuda() for s in src_mask]
            adjs = [adj.cuda() for adj in adjs]
            concept = [c.cuda() for c in concept]
            concept_mask = concept_mask.cuda()
            title_index = title_index.cuda()
        # beam_size = self.config.beam_size
        batch_size = len(src)

        # (1) Run the encoder on the src. Done!!!!
        contexts, state = self.encode(src, src_mask, concept, concept_mask, title_index, adjs)
        c0, h0 = self.build_init_state(state, self.config.num_layers)

        def rvar(a):
            return a.repeat(1, beam_size, 1)

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # Repeat everything beam_size times.
        contexts = contexts.repeat(beam_size, 1, 1)
        concept_mask = concept_mask.repeat(beam_size, 1)
        concept_vocab = concept_vocab.repeat(beam_size, 1)
        title_index = title_index.repeat(beam_size)
        decState = (c0.repeat(1, beam_size, 1), h0.repeat(1, beam_size, 1))
        beam = [models.Beam(beam_size, n_best=1, cuda=use_cuda)
                for _ in range(batch_size)]

        # (2) run the decoder to generate sentences, using beam search.

        for i in range(self.config.max_tgt_len):

            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = torch.stack([b.getCurrentState() for b in beam]).t().contiguous().view(-1)

            # Run one step.
            if self.use_copy:
                output, decState, attn, p_gen = self.decoder.sample_one(inp, decState, contexts, concept_mask,
                                                                        title_index,
                                                                        max_oov=0, extend_vocab=concept_vocab)
            else:
                output, decState, attn = self.decoder.sample_one(inp, decState, contexts)
                output = F.softmax(output, -1)
            # decOut: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            output = unbottle(torch.log(output))
            attn = unbottle(attn)
            # beam x tgt_vocab

            # (c) Advance each beam.
            # update state
            for j, b in enumerate(beam):
                b.advance(output.data[:, j], attn.data[:, j])
                b.beam_update(decState, j)

        # (3) Package everything up.
        allHyps, allScores, allAttn = [], [], []

        for j in range(batch_size):
            b = beam[j]
            n_best = 1
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                attn.append(att.max(1)[1])
            allAttn.append(attn[0])
            allHyps.append(hyps[0])

        # print(allHyps)
        # print(allAttn)
        return allHyps, allAttn
