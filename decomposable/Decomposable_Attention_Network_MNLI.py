
from data_loader_MNLI import *
import os
import sys
sys.path.append('./model')

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import re
import random

#import decomposable as model
import time
import pickle

'''
data_path = "./data/snli_1.0/"
glove_path = "./data/glove.6B.300d.txt"
train_file = "snli_1.0_train.txt"
valid_file = "snli_1.0_dev.txt"
'''
accu_value = 0  # 0.1
parameter_std = 0.01  # 0.01
hidden_size = 200  # 200
label_size = 3
learning_rate = 0.05
weight_decay = 1e-5  # 1e-5
epoch_number = 5000
batch_size = 32
note = "decomposable_hs_200_epoch5000_MNLI"
model_saving_dir = "saved_model/"
embedding_saved_dir = "saved_embedding/"
train_genre = 'telephone'  # "slate"  # None if not training seperate genre
note = note + train_genre
load_and_train = False
evaluation_only = True


class EmbedEncoder(nn.Module):

    def __init__(self, input_size, embedding_dim, hidden_dim, para_init):
        super(EmbedEncoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(input_size, embedding_dim, padding_idx=100)
        self.input_linear = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.para_init = para_init

        '''initialize parameters'''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, self.para_init)

    def forward(self, prem, hypo):
        batch_size = prem.size(0)

        prem_emb = self.embed(prem)
        hypo_emb = self.embed(hypo)

        prem_emb = prem_emb.view(-1, self.embedding_dim)
        hypo_emb = hypo_emb.view(-1, self.embedding_dim)

        prem_emb = self.input_linear(prem_emb).view(batch_size, -1, self.hidden_dim)
        hypo_emb = self.input_linear(hypo_emb).view(batch_size, -1, self.hidden_dim)

        return prem_emb, hypo_emb


# Decomposable Attention
class DecomposableAttention(nn.Module):
    # inheriting from nn.Module!

    def __init__(self, hidden_dim, num_labels, para_init):
        super(DecomposableAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.dropout = nn.Dropout(p=0.2)
        self.para_init = para_init

        # layer F, G, and H are feed forward nn with ReLu
        self.mlp_F = self.mlp(hidden_dim, hidden_dim)
        self.mlp_G = self.mlp(2 * hidden_dim, hidden_dim)
        self.mlp_H = self.mlp(2 * hidden_dim, hidden_dim)

        # final layer will not use dropout, so defining independently
        self.linear_final = nn.Linear(hidden_dim, num_labels, bias=True)

        '''initialize parameters'''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, self.para_init)
                m.bias.data.normal_(0, self.para_init)

    def mlp(self, input_dim, output_dim):
        '''
        Define a feed forward neural network with ReLu activations
        '''
        feed_forward = []
        feed_forward.append(self.dropout)
        feed_forward.append(nn.Linear(input_dim, output_dim, bias=True))
        feed_forward.append(nn.ReLU())
        feed_forward.append(self.dropout)
        feed_forward.append(nn.Linear(output_dim, output_dim, bias=True))
        feed_forward.append(nn.ReLU())
        return nn.Sequential(*feed_forward)

    def forward(self, prem_emb, hypo_emb):
        '''Input layer'''
        len_prem = prem_emb.size(1)
        len_hypo = hypo_emb.size(1)

        '''Attend'''
        f_prem = self.mlp_F(prem_emb.view(-1, self.hidden_dim))
        f_hypo = self.mlp_F(hypo_emb.view(-1, self.hidden_dim))

        f_prem = f_prem.view(-1, len_prem, self.hidden_dim)
        f_hypo = f_hypo.view(-1, len_hypo, self.hidden_dim)

        e_ij = torch.bmm(f_prem, torch.transpose(f_hypo, 1, 2))
        beta_ij = F.softmax(e_ij.view(-1, len_hypo)).view(-1, len_prem, len_hypo)
        beta_i = torch.bmm(beta_ij, hypo_emb)

        e_ji = torch.transpose(e_ij.contiguous(), 1, 2)
        e_ji = e_ji.contiguous()
        alpha_ji = F.softmax(e_ji.view(-1, len_prem)).view(-1, len_hypo, len_prem)
        alpha_j = torch.bmm(alpha_ji, prem_emb)

        '''Compare'''
        concat_1 = torch.cat((prem_emb, beta_i), 2)
        concat_2 = torch.cat((hypo_emb, alpha_j), 2)
        compare_1 = self.mlp_G(concat_1.view(-1, 2 * self.hidden_dim))
        compare_2 = self.mlp_G(concat_2.view(-1, 2 * self.hidden_dim))
        compare_1 = compare_1.view(-1, len_prem, self.hidden_dim)
        compare_2 = compare_2.view(-1, len_hypo, self.hidden_dim)

        '''Aggregate'''
        v_1 = torch.sum(compare_1, 1)
        v_1 = torch.squeeze(v_1, 1)
        v_2 = torch.sum(compare_2, 1)
        v_2 = torch.squeeze(v_2, 1)
        v_concat = torch.cat((v_1, v_2), 1)
        y_pred = self.mlp_H(v_concat)

        '''Final layer'''
        out = F.log_softmax(self.linear_final(y_pred))

        return out


def train(batch_size, use_shrinkage, epoch_number, initial_accumulator_value, learning_rate, model_saving_dir, note, to_lower, hidden_size, load_and_train=None, train_genre=None):
    print("loading glove")
    vocab, word_embeddings, word_to_index, index_to_word = load_embedding_and_build_vocab('./data/glove.6B.300d.txt')
    print("loading data")
    if train_genre != None:
        training_set = process_mnli_genre('./data/multinli_1.0/multinli_1.0_train.jsonl', word_to_index, to_lower, train_genre)
    else:
        training_set = process_mnli('./data/multinli_1.0/multinli_1.0_train.jsonl', word_to_index, to_lower)
    train_iter = batch_iter(dataset=training_set, batch_size=batch_size, shuffle=True)

    if train_genre != None:
        dev_set_match_genre = process_mnli_genre('./data/multinli_1.0/multinli_1.0_dev_matched.jsonl', word_to_index, to_lower, train_genre)
        dev_iter_match_genre = batch_iter(dataset=dev_set_match_genre, batch_size=batch_size, shuffle=False)

    dev_set_match = process_mnli('./data/multinli_1.0/multinli_1.0_dev_matched.jsonl', word_to_index, to_lower)
    dev_iter_match = batch_iter(dataset=dev_set_match, batch_size=batch_size, shuffle=False)

    dev_set_mismatch = process_mnli('./data/multinli_1.0/multinli_1.0_dev_mismatched.jsonl', word_to_index, to_lower)
    dev_iter_mismatch = batch_iter(dataset=dev_set_mismatch, batch_size=batch_size, shuffle=False)

    num_batch_match = len(dev_set_match) // batch_size
    num_batch_mismatch = len(dev_set_mismatch) // batch_size
    num_batch_train = len(training_set) // batch_size

    use_cuda = torch.cuda.is_available()

    # Normalize embedding vector (l2-norm = 1)
    word_embeddings[100, :] = np.ones(300)
    word_embeddings = (word_embeddings.T / np.linalg.norm(word_embeddings, ord=2, axis=1)).T
    word_embeddings[100, :] = np.zeros(300)

    # Encoder and Model
    input_encoder = EmbedEncoder(input_size=word_embeddings.shape[0], embedding_dim=300, hidden_dim=hidden_size, para_init=0.01)
    input_encoder.embed.weight.data.copy_(torch.from_numpy(word_embeddings))
    input_encoder.embed.weight.requires_grad = False
    model = DecomposableAttention(hidden_dim=200, num_labels=3, para_init=0.01)

    if load_and_train == True:
        model.load_state_dict(torch.load(model_saving_dir + 'model' + '_' + note + '.pt'))
        input_encoder.load_state_dict(torch.load(model_saving_dir + 'input_encoder' + '_' + note + '.pt'))

    if use_cuda:
        input_encoder.cuda()
        model.cuda()

    # Optimizer
    para1 = filter(lambda p: p.requires_grad, input_encoder.parameters())
    para2 = model.parameters()
    input_optimizer = torch.optim.Adagrad(para1, lr=learning_rate, weight_decay=0)
    optimizer = torch.optim.Adagrad(para2, lr=learning_rate, weight_decay=0)

    # Initialize the optimizer
    for group in input_optimizer.param_groups:
        for p in group['params']:
            state = input_optimizer.state[p]
            state['sum'] += initial_accumulator_value
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            state['sum'] += initial_accumulator_value

    # Loss
    loss = nn.NLLLoss()

    train_losses = []
    train_accuracies = []
    valid_accuracies_match = []
    valid_accuracies_mismatch = []
    train_times = []
    valid_times = []

    train_statistics = {}

    best_acc = 0  # match dataset

    for epoch in range(epoch_number):
        # for epoch in range(2):
        total = 0
        correct = 0
        loss_data = 0

        epoch_timer = time.time()

        for i in range(num_batch_train):
            # for i in range(2):
            timer = time.time()
            input_encoder.train()
            model.train()

            label, premise, hypothesis, genre = next(train_iter)

            if use_cuda:
                premise_var = Variable(torch.LongTensor(premise).cuda())
                hypothesis_var = Variable(torch.LongTensor(hypothesis).cuda())
                label_var = Variable(torch.LongTensor(label).cuda())
            else:
                premise_var = Variable(torch.LongTensor(premise))
                hypothesis_var = Variable(torch.LongTensor(hypothesis))
                label_var = Variable(torch.LongTensor(label))

            input_encoder.zero_grad()
            model.zero_grad()

            prem_emb, hypo_emb = input_encoder(premise_var, hypothesis_var)
            output = model(prem_emb, hypo_emb)

            lossy = loss(output, label_var)
            lossy.backward()

            # Shrinkage
            if use_shrinkage is True:
                grad_norm = 0.
                for m in input_encoder.modules():
                    if isinstance(m, nn.Linear):
                        grad_norm += m.weight.grad.data.norm() ** 2
                        if m.bias is not None:
                            grad_norm += m.bias.grad.data.norm() ** 2
                for m in model.modules():
                    if isinstance(m, nn.Linear):
                        grad_norm += m.weight.grad.data.norm() ** 2
                        if m.bias is not None:
                            grad_norm += m.bias.grad.data.norm() ** 2
                grad_norm ** 0.5
                shrinkage = 5 / (grad_norm + 1e-6)
                if shrinkage < 1:
                    for m in input_encoder.modules():
                        if isinstance(m, nn.Linear):
                            m.weight.grad.data = m.weight.grad.data * shrinkage
                    for m in model.modules():
                        if isinstance(m, nn.Linear):
                            m.weight.grad.data = m.weight.grad.data * shrinkage
                            m.bias.grad.data = m.bias.grad.data * shrinkage

            input_optimizer.step()
            optimizer.step()
            ###

            #print("after opt.step()")
            # for para in para1:
            #    print(para.data.cpu())
            # print(para1)

            _, predicted = torch.max(output.data, 1)

            total += batch_size
            correct += torch.sum(predicted == label_var.data)
            loss_data += (lossy.data[0] * batch_size)

            # if i == 5:
            #    break

            if i % 100 == 0:
                print("predicted")
                print(predicted.cpu().numpy())
                print("actual label")
                print(label_var.data.cpu().numpy())

                print('epoch: {}, batches: {}|{}, train-acc: {}, loss: {}, time : {}s '.format
                      (epoch, i + 1, num_batch_train, correct / float(total),
                       loss_data / float(total), time.time() - timer))

        epoch_time = time.time() - epoch_timer
        print('epoch: {}, train-acc: {}, loss: {}, time : {}s '.format
              (epoch, correct / float(total),
               loss_data / float(total), epoch_time))

        valid_timer = time.time()
        if train_genre:
            valid_accuracy_match = evaluate(model, input_encoder, dev_iter_match_genre, num_batch_match, use_cuda)
            valid_accuracy_mismatch = evaluate(model, input_encoder, dev_iter_mismatch_genre, num_batch_mismatch, use_cuda)
        else:
            valid_accuracy_match = evaluate(model, input_encoder, dev_iter_match, num_batch_match, use_cuda)
            valid_accuracy_mismatch = evaluate(model, input_encoder, dev_iter_mismatch, num_batch_mismatch, use_cuda)
        valid_time = time.time() - valid_timer

        train_losses.append(loss_data / total)
        train_accuracies.append(correct / total)
        train_times.append(epoch_time)

        valid_accuracies_match.append(valid_accuracy_match)
        valid_accuracies_mismatch.append(valid_accuracy_mismatch)
        valid_times.append(valid_time)

        train_statistics['train_loss_history'] = train_losses
        train_statistics['train_accuracy_history'] = train_accuracies
        train_statistics['train_time'] = train_times

        train_statistics['valid_accuracies_match'] = valid_accuracies_match
        train_statistics['valid_accuracy_mismatch'] = valid_accuracy_mismatch
        train_statistics['valid_time'] = valid_times

        acc_match, prediction_match = evaluate_by_genre(model, input_encoder, dev_iter_match, num_batch_match, use_cuda)
        print('Accuracy by genres in match dataset: {}'.format(acc_match))
        acc_mismatch, prediction_mismatch = evaluate_by_genre(model, input_encoder, dev_iter_mismatch, num_batch_mismatch, use_cuda)
        print('Accuracy by genres in mismatch dataset: {}'.format(acc_mismatch))

        if valid_accuracy_match > best_acc:
            torch.save(input_encoder.state_dict(), model_saving_dir + 'input_encoder' + '_' + note + '.pt')
            torch.save(model.state_dict(), model_saving_dir + 'model' + '_' + note + '.pt')
            best_acc = valid_accuracy_match
            pickle.dump(prediction_match, open(model_saving_dir + 'match_prediction' + '_' + note + '.pk', 'wb'))
            pickle.dump(prediction_mismatch, open(model_saving_dir + 'mismatch_prediction' + '_' + note + '.pk', 'wb'))

        print('epoch: {}, valid-acc_match : {}, valid-acc_mismatch: {}, best_valid_acc_match: {}, valid_time : {}s '.format
              (epoch, valid_accuracy_match, valid_accuracy_mismatch, best_acc, valid_time))

        pickle.dump(train_statistics, open(model_saving_dir + 'training_history' + '_' + note + '.pk', 'wb'))


def evaluate(inter_atten, input_encoder, data_iter, num_batch, use_cuda):
    input_encoder.eval()
    inter_atten.eval()
    correct = 0
    total = 0
    #step = 0
    loss_data = 0
    print("valuating the model")
    for i in range(num_batch):
        label, sentence1, sentence2, genre = next(data_iter)
        if use_cuda:
            sentence1_var = Variable(torch.LongTensor(sentence1).cuda())
            sentence2_var = Variable(torch.LongTensor(sentence2).cuda())
            label_var = Variable(torch.LongTensor(label).cuda())
        else:
            sentence1_var = Variable(torch.LongTensor(sentence1))
            sentence2_var = Variable(torch.LongTensor(sentence2))
            label_var = Variable(torch.LongTensor(label))

        embed_1, embed_2 = input_encoder(sentence1_var, sentence2_var)  # batch_size * length * embedding_dim
        prob = inter_atten(embed_1, embed_2)

        if use_cuda:
            prob.cpu()

        _, predicted = torch.max(prob.data, 1)

        total += len(label)
        correct += (predicted == label_var.data).sum()

        # if i == 5:
        #    break

    input_encoder.train()
    inter_atten.train()

    return correct / float(total)


def evaluate_by_genre(inter_atten, input_encoder, data_iter, num_batch, use_cuda):

    genres = []
    input_encoder.eval()
    inter_atten.eval()
    correct = {}
    total = {}
    #step = 0
    prediction = []

    for i in range(num_batch):
        label, sentence1, sentence2, genre = next(data_iter)
        total.update(dict([(i, genre.count(i)) for i in set(genre)]))
        correct_list = []
        if use_cuda:
            sentence1_var = Variable(torch.LongTensor(sentence1).cuda())
            sentence2_var = Variable(torch.LongTensor(sentence2).cuda())
            label_var = Variable(torch.LongTensor(label).cuda())
        else:
            sentence1_var = Variable(torch.LongTensor(sentence1))
            sentence2_var = Variable(torch.LongTensor(sentence2))
            label_var = Variable(torch.LongTensor(label))

        embed_1, embed_2 = input_encoder(sentence1_var, sentence2_var)  # batch_size * length * embedding_dim
        prob = inter_atten(embed_1, embed_2)

        if use_cuda:
            prob.cpu()

        _, predicted = torch.max(prob.data, 1)
        prediction.extend(predicted.cpu().numpy())

        if len(genre) > 1:
            for j in range(len(genre)):
                print("{}, {}, {}".format(genre[j], label[j], predicted.cpu().numpy()[j]))
                if predicted.cpu().numpy()[j] == label[j]:
                    correct_list.append(genre[j])
        else:
            if predicted.cpu().numpy() == label:
                correct_list.append(genre)

        correct.update([(i, correct_list.count(i)) for i in set(correct_list)])

        # if i == 5:
        #    break

    input_encoder.train()
    inter_atten.train()

    accuracy = {k: correct[k] / float(total[k]) for k in correct}

    return accuracy, prediction


def evaluation_load_model(batch_size, use_shrinkage, epoch_number, initial_accumulator_value, learning_rate, model_saving_dir, note, to_lower, hidden_size, load_and_train=None, train_genre=None):
    print("loading glove")
    vocab, word_embeddings, word_to_index, index_to_word = load_embedding_and_build_vocab('./data/glove.6B.300d.txt')
    print("loading data")
    
    if train_genre != None:
        dev_set_match_genre = process_mnli_genre('./data/multinli_1.0/multinli_1.0_dev_matched.jsonl', word_to_index, to_lower, train_genre)
        dev_iter_match_genre = batch_iter(dataset=dev_set_match_genre, batch_size=batch_size, shuffle=False)
    
    dev_set_match = process_mnli('./data/multinli_1.0/multinli_1.0_dev_matched.jsonl', word_to_index, to_lower)
    dev_iter_match = batch_iter(dataset=dev_set_match, batch_size=batch_size, shuffle=False)

    dev_set_mismatch = process_mnli('./data/multinli_1.0/multinli_1.0_dev_mismatched.jsonl', word_to_index, to_lower)
    dev_iter_mismatch = batch_iter(dataset=dev_set_mismatch, batch_size=batch_size, shuffle=False)

    num_batch_match = len(dev_set_match) // batch_size
    num_batch_mismatch = len(dev_set_mismatch) // batch_size

    use_cuda = torch.cuda.is_available()

    # Encoder and Model
    input_encoder = EmbedEncoder(input_size=word_embeddings.shape[0], embedding_dim=300, hidden_dim=hidden_size, para_init=0.01)
    model = DecomposableAttention(hidden_dim=200, num_labels=3, para_init=0.01)

    if use_cuda:
        input_encoder.cuda()
        model.cuda()

    model.load_state_dict(torch.load(model_saving_dir + 'model' + '_' + note + '.pt'))
    input_encoder.load_state_dict(torch.load(model_saving_dir + 'input_encoder' + '_' + note + '.pt'))

    accuracy_match, _ = evaluate_by_genre(model, input_encoder, dev_iter_match, num_batch_match, use_cuda)
    accuracy_mismatch, _ = evaluate_by_genre(model, input_encoder, dev_iter_mismatch, num_batch_mismatch, use_cuda)
    print('Accuracy by genres in match dataset: {}'.format(accuracy_match))
    print('Accuracy by genres in mismatch dataset: {}'.format(accuracy_mismatch))


if __name__ == '__main__':
    '''
    begin_preprocess = time.time()
    print("Loading Glove")
    glove_dic = pp.loadGloveData(glove_path)

    print("Reading data set")
    train_set = pp.read_data_set(os.path.join(data_path, train_file))
    valid_set = pp.read_data_set(os.path.join(data_path, valid_file))

    print("Loading word embedding")
    idx2word, word2idx, embedding = pp.build_vocabulary_with_glove(train_set, glove_dic)
    #embedding = pickle.load(open('embedding' + '_' + note + '.pk', 'rb'))
    #word2idx = pickle.load(open('word2idx' + '_' + note + '.pk', 'rb'))
    pickle.dump(embedding, open(embedding_saved_dir + 'embedding' + '_' + note + '.pk', 'wb'))
    pickle.dump(word2idx, open(embedding_saved_dir + 'word2idx' + '_' + note + '.pk', 'wb'))

    print("batchfying both training and valid data")
    train_data_batch = pp.batch_iter(train_set, batch_size, word2idx)
    valid_data_batch = pp.batch_iter(valid_set, batch_size, word2idx)

    print("Time takes to process data: {}s".format(time.time() - begin_preprocess))

    use_cuda = torch.cuda.is_available()

    print("Use cuda? : {}".format(use_cuda))

    train(embedding, train_data_batch, valid_data_batch, use_cuda)
    '''
    if __name__ == '__main__':
        if evaluation_only == True:
            evaluation_load_model(batch_size=batch_size, use_shrinkage=False, epoch_number=epoch_number, initial_accumulator_value=accu_value,
                                  learning_rate=learning_rate, model_saving_dir=model_saving_dir, note=note, to_lower=True, hidden_size=hidden_size, load_and_train=False, train_genre=train_genre)

        else:
            train(batch_size=batch_size, use_shrinkage=False, epoch_number=epoch_number, initial_accumulator_value=accu_value,
                  learning_rate=learning_rate, model_saving_dir=model_saving_dir, note=note, to_lower=True, hidden_size=hidden_size, load_and_train=False, train_genre=train_genre)
