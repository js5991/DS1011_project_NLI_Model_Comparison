
import preprocess.snli_preprocessing as pp
import os
import sys
sys.path.append('./model')

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import decomposable as model
import time
import pickle


data_path = "./data/snli_1.0/"
glove_path = "./data/glove.6B.300d.txt"
train_file = "snli_1.0_train.txt"
valid_file = "snli_1.0_dev.txt"

accu_value = 0.1  # 0.1
parameter_std = 0.01  # 0.01
hidden_size = 200  # 200
label_size = 3
learning_rate = 1e-1
weight_decay = 1e-5  # 1e-5
epoch_number = 100
batch_size = 32
display_minibatch_batch = 100
note = "decomposable_wd_1e-5_std_1e-2_av_0.1_hs_200_epoch100"
model_saving_dir = "saved_model/"
embedding_saved_dir = "saved_embedding/"


def train(embedding, train_data_batch, valid_data_batch, use_cuda):

    input_encoder = model.encoder(embedding.shape[0], embedding_size=300, hidden_size=hidden_size, para_init=parameter_std)
    input_encoder.embedding.weight.data.copy_(torch.from_numpy(embedding))
    input_encoder.embedding.weight.requires_grad = False

    inter_atten = model.atten(hidden_size=hidden_size, label_size=label_size, para_init=parameter_std)

    if use_cuda:
        input_encoder.cuda()
        inter_atten.cuda()

    para1 = list(filter(lambda p: p.requires_grad, input_encoder.parameters()))
    para2 = list(inter_atten.parameters())

    input_optimizer = optim.Adagrad(para1, lr=learning_rate, weight_decay=weight_decay)
    inter_atten_optimizer = optim.Adagrad(para2, lr=learning_rate, weight_decay=weight_decay)

    criterion = nn.NLLLoss(size_average=True)

    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    train_times = []
    valid_times = []

    train_statistics = {}

    best_acc = 0

    for epoch in range(epoch_number):
        # for epoch in range(2):
        total = 0
        correct = 0
        loss_data = 0

        step_size_per_epoch = int(len(train_set) / batch_size)
        epoch_timer = time.time()

        for i in range(step_size_per_epoch):
            # for i in range(2):
            timer = time.time()
            sentence1, sentence2, label = next(train_data_batch)
            input_encoder.train()
            inter_atten.train()

            if use_cuda:
                sentence1_var = Variable(torch.LongTensor(sentence1).cuda())
                sentence2_var = Variable(torch.LongTensor(sentence2).cuda())
                label_var = Variable(torch.LongTensor(label).cuda())
            else:
                sentence1_var = Variable(torch.LongTensor(sentence1))
                sentence2_var = Variable(torch.LongTensor(sentence2))
                label_var = Variable(torch.LongTensor(label))

            # print(label_var)

            input_encoder.zero_grad()
            inter_atten.zero_grad()

            for group in input_optimizer.param_groups:
                for p in group['params']:
                    state = input_optimizer.state[p]
                    state['sum'] += accu_value
            for group in inter_atten_optimizer.param_groups:
                for p in group['params']:
                    state = inter_atten_optimizer.state[p]
                    state['sum'] += accu_value

            embed_1, embed_2 = input_encoder(sentence1_var, sentence2_var)  # batch_size * length * embedding_dim
            prob = inter_atten(embed_1, embed_2)

            # print(prob)
            loss = criterion(prob, label_var)
            loss.backward()

            grad_norm = 0.
            para_norm = 0.

            for m in input_encoder.modules():
                if isinstance(m, nn.Linear):
                    # print(m.weight.grad.data.cpu())
                    grad_norm += m.weight.grad.data.norm() ** 2
                    para_norm += m.weight.data.norm() ** 2
                    if m.bias:
                        grad_norm += m.bias.grad.data.norm() ** 2
                        para_norm += m.bias.data.norm() ** 2

            for m in inter_atten.modules():
                if isinstance(m, nn.Linear):
                    grad_norm += m.weight.grad.data.norm() ** 2
                    para_norm += m.weight.data.norm() ** 2
                    # if m.bias:
                    grad_norm += m.bias.grad.data.norm() ** 2
                    para_norm += m.bias.data.norm() ** 2

            grad_norm ** 0.5
            para_norm ** 0.5

            max_grad_norm = 5.0
            shrinkage = max_grad_norm / grad_norm
            if shrinkage < 1:
                for m in input_encoder.modules():

                    if isinstance(m, nn.Linear):
                        m.weight.grad.data = m.weight.grad.data * shrinkage
                for m in inter_atten.modules():
                    if isinstance(m, nn.Linear):
                        m.weight.grad.data = m.weight.grad.data * shrinkage
                        m.bias.grad.data = m.bias.grad.data * shrinkage

            #print("before opt.step()")
            #print(para1)

            input_optimizer.step()
            inter_atten_optimizer.step()
            ###

            #print("after opt.step()")
            #for para in para1:
            #    print(para.data.cpu())
            # print(para1)

            _, predict = prob.data.max(dim=1)

            total += batch_size
            correct += torch.sum(predict == label_var.data)
            loss_data += (loss.data[0] * batch_size)

            if i % display_minibatch_batch == 0:
                print("predicted")
                print(predict.cpu().numpy())
                print("actual label")
                print(label_var.data.cpu().numpy())

                print('epoch: {}, batches: {}|{}, train-acc: {}, loss: {}, para-norm: {}, grad-norm: {}, time : {}s '.format
                      (epoch, i + 1, step_size_per_epoch, correct / float(total),
                       loss_data / float(total), para_norm, grad_norm, time.time() - timer))

        epoch_time = time.time() - epoch_timer
        print('epoch: {}, train-acc: {}, loss: {}, para-norm: {}, grad-norm: {}, time : {}s '.format
              (epoch, correct / float(total),
               loss_data / float(total), para_norm, grad_norm, epoch_time))

        valid_timer = time.time()
        valid_accuracy, valid_loss = evaluate(inter_atten, input_encoder, valid_data_batch, use_cuda)
        valid_time = time.time() - valid_timer
        print('epoch: {}, valid-acc: {}, loss: {}, valid_time : {}s '.format
              (epoch, valid_accuracy, valid_loss, valid_time))

        train_losses.append(loss_data / total)
        train_accuracies.append(correct / total)
        train_times.append(epoch_time)

        valid_accuracies.append(valid_accuracy)
        valid_losses.append(valid_loss)
        valid_times.append(valid_time)

        train_statistics['train_loss_history'] = train_losses
        train_statistics['train_accuracy_history'] = train_accuracies
        train_statistics['train_time'] = train_times

        train_statistics['valid_loss_history'] = valid_losses
        train_statistics['valid_accuracy_history'] = valid_accuracies
        train_statistics['valid_time'] = valid_times

        if valid_accuracy > best_acc:
            torch.save(input_encoder.state_dict(), model_saving_dir + 'input_encoder' + '_' + note + '.pt')
            torch.save(inter_atten.state_dict(), model_saving_dir + 'inter_atten' + '_' + note + '.pt')
            best_acc = valid_accuracy

        pickle.dump(train_statistics, open(model_saving_dir + 'training_history' + '_' + note + '.pk', 'wb'))


def evaluate(inter_atten, input_encoder, data_iter, use_cuda):
    input_encoder.eval()
    inter_atten.eval()
    correct = 0
    total = 0
    #step = 0
    loss_data = 0
    print("valuating the model")
    for i in range(int(len(valid_set) / batch_size)):
        sentence1, sentence2, label = next(data_iter)
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

        _, predicted = prob.data.max(dim=1)

        total += label_var.size(0)
        correct += (predicted == label_var.data).sum()
        criterion = nn.NLLLoss(size_average=True)
        loss = criterion(prob, label_var)
        loss_data += (loss.data[0] * label_var.data.shape[0])

        # print(total)
        #step += 1
        # if step > 5:
        #    break

    input_encoder.train()
    inter_atten.train()

    return correct / float(total), loss_data / float(total)


if __name__ == '__main__':
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
