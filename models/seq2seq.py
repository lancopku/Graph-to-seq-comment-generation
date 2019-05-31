import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import models
from Data import *

import numpy as np


class seq2seq(nn.Module):

    def __init__(self, config, vocab, use_cuda, use_content=False, pretrain=None):
        super(seq2seq, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.voc_size
        if pretrain is not None:
            self.embedding = pretrain['emb']
        else:
            self.embedding = nn.Embedding(self.vocab_size, config.emb_size)
        self.encoder = models.rnn_encoder(config, self.vocab_size, embedding=self.embedding)
        self.decoder = models.rnn_decoder(config, self.vocab_size, embedding=self.embedding)
        self.config = config
        self.use_content = use_content
        self.criterion = models.criterion(self.vocab_size, use_cuda)
        self.log_softmax = nn.LogSoftmax()
        self.tanh = nn.Tanh()

    def compute_loss(self, hidden_outputs, targets):
        return models.cross_entropy_loss(hidden_outputs, targets, self.criterion)

    def forward(self, batch, use_cuda):
        if self.use_content:
            src, src_len, src_mask = batch.title_content, batch.title_content_len, batch.title_content_mask
        else:
            src, src_len, src_mask = batch.title, batch.title_len, batch.title_mask
        tgt = batch.tgt
        if use_cuda:
            src, src_len, tgt, src_mask = src.cuda(), src_len.cuda(), tgt.cuda(), src_mask.cuda()
        contexts, state = self.encoder(src, src_len)
        outputs, final_state, attns = self.decoder(tgt[:, :-1], state, contexts)
        return outputs

    def sample(self, batch, use_cuda):
        if self.use_content:
            src, src_len, src_mask = batch.title_content, batch.title_content_len, batch.title_content_mask
        else:
            src, src_len, src_mask = batch.title, batch.title_len, batch.title_mask
        if use_cuda:
            src, src_len, src_mask = src.cuda(), src_len.cuda(), src_mask.cuda()
        bos = torch.ones(src.size(0)).long().fill_(self.vocab.word2id('[START]'))
        bos = bos.to(src.device)

        contexts, state = self.encoder(src, src_len)
        sample_ids, final_outputs = self.decoder.sample([bos], state, contexts)

        return sample_ids, final_outputs[1]

    # TODO: fix beam search
    def beam_sample(self, batch, use_cuda, beam_size=1):
        if self.use_title:
            src, src_len, src_mask = batch.title, batch.title_len, batch.title_mask
        else:
            src, src_len, src_mask = batch.ori_content, batch.ori_content_len, batch.ori_content_mask
        if use_cuda:
            src, src_len, src_mask = src.cuda(), src_len.cuda(), src_mask.cuda()
        # beam_size = self.config.beam_size
        batch_size = src.size(0)

        # (1) Run the encoder on the src. Done!!!!
        contexts, encState = self.encoder(src, src_len)

        #  (1b) Initialize for the decoder.
        def rvar(a):
            return a.repeat(1, beam_size, 1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # Repeat everything beam_size times.
        # (batch, seq, nh) -> (beam*batch, seq, nh)
        contexts = contexts.repeat(beam_size, 1, 1)
        # (batch, seq) -> (beam*batch, seq)
        src_mask = src_mask.repeat(beam_size, 1)
        assert contexts.size(0) == src_mask.size(0), (contexts.size(), src_mask.size())
        assert contexts.size(1) == src_mask.size(1), (contexts.size(), src_mask.size())
        decState = (rvar(encState[0]), rvar(encState[1]))  # layer, beam*batch, nh
        # decState.repeat_beam_size_times(beam_size)
        beam = [models.Beam(beam_size, n_best=1, cuda=use_cuda)
                for _ in range(batch_size)]

        # (2) run the decoder to generate sentences, using beam search.

        for i in range(self.config.max_tgt_len):

            if all((b.done() for b in beam)):
                break

            # Construct beam*batch  nxt words.
            # Get all the pending current beam words and arrange for forward.
            # beam is batch_sized, so stack on dimension 1 not 0
            inp = torch.stack([b.getCurrentState() for b in beam], 1).contiguous().view(-1)
            if use_cuda:
                inp = inp.cuda()

            # Run one step.
            output, decState, attn = self.decoder.sample_one(inp, decState, contexts, src_mask)
            # decOut: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            output = unbottle(self.log_softmax(output, -1))
            attn = unbottle(attn)
            # beam x tgt_vocab

            # (c) Advance each beam.
            # update state
            for j, b in enumerate(beam):  # there are batch size beams!!! so here enumerate over batch
                b.advance(output.data[:, j], attn.data[:, j])  # output is beam first
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
            allHyps.append(hyps[0])
            allScores.append(scores[0])
            allAttn.append(attn[0])

        # print(allHyps)
        # print(allAttn)
        return allHyps, allAttn
