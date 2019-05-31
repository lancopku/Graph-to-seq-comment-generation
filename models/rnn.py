import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import models
from Data import *
import copy


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        # layerwise stack
        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)

    def zero_state(self):
        return torch.zeros(self.num_layers, self.hidden_size)


class rnn_encoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None):
        super(rnn_encoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.encoder_hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout, bidirectional=config.bidirec,
                           batch_first=True)
        self.config = config

    def forward(self, input, lengths):
        mask = input.ne(0)
        cal_length = torch.sum(mask, 1)
        assert all(cal_length.eq(lengths)), (input, lengths)
        length, indices = torch.sort(lengths, dim=0, descending=True)
        _, ind = torch.sort(indices, dim=0)
        input_length = list(torch.unbind(length, dim=0))
        embs = pack(self.embedding(torch.index_select(input, dim=0, index=indices)), input_length, batch_first=True)
        outputs, (h, c) = self.rnn(embs)
        outputs = unpack(outputs, batch_first=True)[0]
        outputs = torch.index_select(outputs, dim=0, index=ind)
        h = torch.index_select(h, dim=1, index=ind)
        c = torch.index_select(c, dim=1, index=ind)
        # h (directions*layers, batch, hidden)
        if not self.config.bidirec:
            return outputs, (h, c)
        else:
            batch_size = h.size(1)
            # h (directions*layers, batch, hidden) ---> h (layers, batch, direction*hidden)
            h = h.transpose(0, 1).contiguous().view(batch_size, -1, 2 * self.config.encoder_hidden_size)
            c = c.transpose(0, 1).contiguous().view(batch_size, -1, 2 * self.config.encoder_hidden_size)
            state = (h.transpose(0, 1), c.transpose(0, 1))
            return outputs, state


class gated_rnn_encoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None):
        super(gated_rnn_encoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.encoder_hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout)
        self.gated = nn.Sequential(nn.Linear(config.encoder_hidden_size, 1), nn.Sigmoid())

    def forward(self, input, lengths):
        embs = pack(self.embedding(input), lengths)
        outputs, state = self.rnn(embs)
        outputs = unpack(outputs)[0]
        p = self.gated(outputs)
        outputs = outputs * p
        return outputs, state


class rnn_decoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None):
        super(rnn_decoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        self.rnn = StackedLSTM(input_size=config.emb_size, hidden_size=config.decoder_hidden_size,
                               num_layers=config.num_layers, dropout=config.dropout)

        self.linear = nn.Linear(config.decoder_hidden_size, vocab_size)

        if hasattr(config, 'att_act'):
            activation = config.att_act
            print('use attention activation %s' % activation)
        else:
            activation = None

        self.attention = models.global_attention(config.decoder_hidden_size, activation)
        # self.attention = models.global_attention(config.decoder_hidden_size, activation)
        self.hidden_size = config.decoder_hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.config = config

    def forward(self, inputs, init_state, contexts=None):
        embs = self.embedding(inputs)
        outputs, state, attns = [], init_state, []
        for emb in embs.split(1, dim=1):
            output, state = self.rnn(emb.squeeze(1), state)
            attn_weights = None
            if contexts is not None:
                output, attn_weights = self.attention(output, contexts)
            output = self.dropout(output)
            outputs += [self.linear(output)]
            attns += [attn_weights]
        outputs = torch.stack(outputs, 1)
        if contexts is not None:
            attns = torch.stack(attns, 1)
        return outputs, state, attns

    def decode_ae(self, input_length, init_state, bos):
        outputs, state = [], init_state
        emb = self.embedding(bos)

        for _ in torch.arange(0, input_length):
            output, state = self.rnn(emb, state)
            output = self.dropout(output)
            prob = self.linear(output)
            word = prob.max(-1)[1]
            emb = self.embedding(word)
            outputs += [prob]
        outputs = torch.stack(outputs, 1)
        return outputs, state

    def sample(self, input, init_state, contexts=None):
        # emb = self.embedding(input)
        inputs, outputs, sample_ids, state = [], [], [], init_state
        attns = []
        inputs += input
        max_time_step = self.config.max_tgt_len

        for i in range(max_time_step):
            # output: [batch, tgt_vocab_size]
            output, state, attn_weights = self.sample_one(inputs[i], state,
                                                          contexts)  # inputs is a list we just built, not a big batch
            predicted = output.max(dim=1)[1]  # max returns max_value, max_id
            inputs += [predicted]
            sample_ids += [predicted]
            outputs += [output]
            attns += [attn_weights]

        sample_ids = torch.stack(sample_ids, 1)
        if contexts is None:
            return sample_ids, outputs
        else:
            attns = torch.stack(attns, 1)
            return sample_ids, (outputs, attns)

    def sample_one(self, input, state, contexts):
        emb = self.embedding(input)
        output, state = self.rnn(emb, state)
        attn_weigths = None
        if contexts is not None:
            output, attn_weigths = self.attention(output, contexts)
        output = self.linear(output)

        return output, state, attn_weigths


class pointer_decoder(nn.Module):
    def __init__(self, config, vocab_size, embedding=None):
        super(pointer_decoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        self.rnn = StackedLSTM(input_size=config.emb_size, hidden_size=config.decoder_hidden_size,
                               num_layers=config.num_layers, dropout=config.dropout)

        self.p_gen_weight = nn.Linear(config.emb_size + config.decoder_hidden_size * 3, 1)
        self.linear = nn.Linear(config.decoder_hidden_size, vocab_size)
        self.output_merge = nn.Linear(2 * config.decoder_hidden_size, config.decoder_hidden_size)

        if hasattr(config, 'att_act'):
            activation = config.att_act
            print('use attention activation %s' % activation)
        else:
            activation = None

        self.attention = models.global_attention(config.decoder_hidden_size, activation)
        self.memory_attention = models.global_attention(config.decoder_hidden_size, activation)
        self.softmax = nn.Softmax(-1)
        self.hidden_size = config.decoder_hidden_size
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(config.dropout)
        self.config = config

    def forward(self, inputs, init_state, contexts, enc_mask, title_index, max_oov, extend_vocab):
        embs = self.embedding(inputs)
        outputs, state, attns = [], init_state, []
        p_gens = []
        for emb in embs.split(1, dim=1):
            output, state = self.rnn(emb.squeeze(1), state)
            text_output, attn_weights = self.attention(output, contexts, enc_mask)
            p_gen = torch.sigmoid(self.p_gen_weight(
                torch.cat([state[0][-1], state[1][-1], text_output, emb.squeeze(1)], 1)))
            # state[0]->h, [-1]->last layer, state[1]->c, [-1]->last layer
            p_gens.append(p_gen)
            output = self.softmax(self.linear(self.dropout(text_output)))
            assert not torch.isnan(torch.sum(attn_weights)).item(), attn_weights
            attn_weights_mask = self._get_title_attn_weight_mask(attn_weights, title_index)
            final_output = self._calc_final_dist(output, attn_weights * attn_weights_mask, p_gen, max_oov, extend_vocab)
            outputs += [final_output]
            attns += [attn_weights]
        outputs = torch.stack(outputs, 1)
        attns = torch.stack(attns, 1)
        p_gens = torch.stack(p_gens, 1)
        return outputs, state, attns, p_gens

    def sample(self, input, init_state, text_contexts, text_mask, title_index, max_oov, extend_vocab):
        inputs, outputs, sample_ids, state = [], [], [], init_state
        attns = []
        p_gens = []
        inputs += input
        max_time_step = self.config.max_tgt_len

        for i in range(max_time_step):
            # output: [batch, tgt_vocab_size]
            output, state, attn_weights, p_gen = self.sample_one(inputs[i], state, text_contexts, text_mask,
                                                                 title_index, max_oov, extend_vocab)
            p_gens.append(p_gen)
            # TODO: ERROR! if predicted word is oov, should change to dynamic emb afterwards
            predicted = output.max(dim=1)[1]
            input_predicted = predicted.cpu().clone()
            for j in range(input_predicted.size()[0]):
                if input_predicted[j] >= self.vocab_size:
                    input_predicted[j] = torch.LongTensor([UNK])
            inputs += [input_predicted.to(input[0].device)]
            sample_ids += [predicted]
            outputs += [output]
            attns += [attn_weights]

        sample_ids = torch.stack(sample_ids, 1)
        attns = torch.stack(attns, 1)
        p_gens = torch.stack(p_gens, 1)
        return sample_ids, (outputs, attns), p_gens

    def sample_one(self, input, state, text_contexts, text_mask, title_index, max_oov, extend_vocab):
        assert input.max() < self.vocab_size, input
        emb = self.embedding(input)
        output, state = self.rnn(emb, state)
        text_output, attn_weights = self.attention(output, text_contexts, text_mask)
        p_gen = torch.sigmoid(
            self.p_gen_weight(torch.cat([state[0][-1], state[1][-1], text_output, emb.squeeze(1)], 1)))
        # feed last layer of state.h and state.c
        output = self.linear(text_output)
        vocab_dist = self.softmax(output)
        assert extend_vocab.max() < vocab_dist.size(-1) + max_oov, extend_vocab
        attn_weights_mask = self._get_title_attn_weight_mask(attn_weights, title_index)
        final_output = self._calc_final_dist(vocab_dist, attn_weights * attn_weights_mask, p_gen, max_oov, extend_vocab)

        return final_output, state, attn_weights, p_gen

    @staticmethod
    def _get_title_attn_weight_mask(attn_weights, title_index):
        mask = torch.ones_like(attn_weights, dtype=torch.float32)
        batch_size = attn_weights.size(0)
        batch_nums = torch.arange(0, end=batch_size, device=attn_weights.device)  # shape (batch_size)
        batch_nums = torch.unsqueeze(batch_nums, 1)  # shape (batch_size, 1)
        title_index = title_index.unsqueeze(1)
        mask[(batch_nums, title_index)] = torch.zeros_like(title_index, dtype=torch.float32)
        return mask

    def _calc_final_dist(self, vocab_dist, attn_dist, p_gen, max_oov, enc_batch_extend_vocab):
        """Calculate the final distribution, for the pointer-generator model

        Args:
          vocab_dist: The vocabulary distributions. (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
          attn_dist: The attention distributions. (batch_size, attn_len) arrays

        Returns:
          final_dist: The final distributions. (batch_size, extended_vsize) arrays.
        """
        # TODO: this method should be correct, the biggest problem now is the OOV index in train.py (compute loss) and
        # copynet.py in decoder() batch.memory -> batch.memory_extend
        # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
        vocab_dist = vocab_dist * p_gen
        attn_dist = (1 - p_gen) * attn_dist

        # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
        extended_vsize = self.vocab_size + max_oov  # the maximum (over the batch) size of the extended vocabulary
        batch_size = vocab_dist.size(0)
        extra_zeros = torch.zeros((batch_size, max_oov), device=vocab_dist.device)
        vocab_dist_extended = torch.cat([vocab_dist, extra_zeros], 1)  # (batch_size, extended_vsize)

        # Project the values in the attention distributions onto the appropriate entries in the final distributions
        # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
        # This is done for each decoder timestep.
        batch_nums = torch.arange(0, end=batch_size, device=vocab_dist.device)  # shape (batch_size)
        batch_nums = torch.unsqueeze(batch_nums, 1)  # shape (batch_size, 1)
        attn_len = attn_dist.size(1)
        batch_nums = batch_nums.repeat([1, attn_len])  # shape (batch_size, attn_len)
        assert batch_nums.size() == enc_batch_extend_vocab.size() and batch_nums.size() == attn_dist.size(), (
            batch_nums.size(), enc_batch_extend_vocab.size())
        indices = (batch_nums, enc_batch_extend_vocab)  # shape (batch_size, enc_t, 2)
        # the indices here are actually got from input, batch_nums only provides the first dimension according to batch
        attn_dist = attn_dist + 1e-12
        attn_dist_projected = torch.zeros(batch_size, extended_vsize, device=vocab_dist.device)
        attn_dist_projected.index_put_(indices, attn_dist)
        # TODO: should fix this comment after debugging

        # Add the vocab distributions and the copy distributions together to get the final distributions        assert torch.equal(attn_dist_projected[indices], attn_dist), (attn_dist_projected, attn_dist)
        # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
        # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
        assert vocab_dist_extended.size() == attn_dist_projected.size()
        final_dist = vocab_dist_extended + attn_dist_projected

        return final_dist
