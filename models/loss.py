import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import Data as dict


def criterion(tgt_vocab_size, use_cuda):
    weight = torch.ones(tgt_vocab_size)
    weight[dict.PAD] = 0
    crit = nn.CrossEntropyLoss(weight, ignore_index=dict.UNK, reduction='sum')
    if use_cuda:
        crit.cuda()
    return crit


def memory_efficiency_cross_entropy_loss(hidden_outputs, targets, criterion, config):
    outputs = torch.Tensor(hidden_outputs.data, requires_grad=True, volatile=False)
    num_total, num_correct, loss = 0, 0, 0

    outputs_split = torch.split(outputs, config.max_generator_batches)
    targets_split = torch.split(targets, config.max_generator_batches)
    for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
        out_t = out_t.view(-1, out_t.size(2))
        # scores_t = decoder.compute_score(out_t)
        loss_t = criterion(out_t, targ_t.view(-1))
        pred_t = out_t.max(1)[1]
        num_correct_t = pred_t.data.eq(targ_t.data).masked_select(targ_t.ne(dict.PAD).data).sum()
        num_total_t = targ_t.ne(dict.PAD).data.sum()
        num_correct += num_correct_t
        num_total += num_total_t
        loss += loss_t.data[0]
        loss_t.div(num_total_t).backward()

    grad_output = outputs.grad.data
    hidden_outputs.backward(grad_output)

    return loss, num_total, num_correct, config.tgt_vocab, config.tgt_vocab


def cross_entropy_loss(hidden_outputs, targets, criterion):
    outputs = hidden_outputs.contiguous().view(-1, hidden_outputs.size(2))
    targets = targets.contiguous().view(-1)
    # scores = decoder.compute_score(outputs)
    loss = criterion(outputs, targets)
    pred = outputs.max(dim=1)[1]
    num_correct = pred.data.eq(targets.data).masked_select(targets.ne(dict.PAD).data).sum()
    num_total = targets.ne(dict.PAD).data.sum()
    loss = loss.div(num_total.float())

    return loss, num_correct.float() / num_total.float()
