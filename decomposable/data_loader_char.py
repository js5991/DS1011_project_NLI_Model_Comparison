import numpy as np
import random
import json
from operator import add
import random

UNKNOWN = '<UNK>'  # 0
PADDING = '<PAD>'  # 1


def process_snli(file_path, word_to_index, to_lower, n=5):
    label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            example = {}
            line = json.loads(line)
            if line['gold_label'] != '-':
                example['label'] = label_dict[line['gold_label']]
                if to_lower is True:
                    tmp1 = line['sentence1_binary_parse'].replace('(', '').replace(')', '').lower().split()
                    tmp2 = line['sentence2_binary_parse'].replace('(', '').replace(')', '').lower().split()
                else:
                    tmp1 = line['sentence1_binary_parse'].replace('(', '').replace(')', '').split()
                    tmp2 = line['sentence2_binary_parse'].replace('(', '').replace(')', '').split()
                tmp1.insert(0, '<NULL>')
                tmp2.insert(0, '<NULL>')
                example['premise'] = ' '.join(tmp1)
                example['hypothesis'] = ' '.join(tmp2)
                example['premise_to_words'] = [word for word in example['premise'].split(' ')]
                example['hypothesis_to_words'] = [word for word in example['hypothesis'].split(' ')]
                example['premise_to_tokens'] = [word_to_index[word] if word in word_to_index.keys() else (hash(word) % 100) for word in example['premise_to_words']]
                example['hypothesis_to_tokens'] = [word_to_index[word] if word in word_to_index.keys() else (hash(word) % 100) for word in example['hypothesis_to_words']]
                example['max_length'] = max(len(example['premise_to_tokens']), len(example['hypothesis_to_tokens']))

                if n > 0:
                    example['question_1_ngrams'] = [word_to_char_ngrams(word, n)
                                                    for word in example['premise_to_words']]
                    example['question_2_ngrams'] = [word_to_char_ngrams(word, n)
                                                    for word in example['hypothesis_to_words']]
                data.append(example)
    return data


def add_char_ngrams(data, build_ngram_map=True, ngram_to_index_map=None):
    if build_ngram_map:
        ngram_to_index_map = {UNKNOWN: 0, PADDING: 1}
        index_to_ngram_map = {0: UNKNOWN, 1: PADDING}
        cur_index = 2
        # build ngram map
        for example in data:
            for ngrams in example['question_1_ngrams'] + example['question_2_ngrams']:
                for ngram in ngrams:
                    if ngram not in ngram_to_index_map.keys():
                        ngram_to_index_map[ngram] = cur_index
                        index_to_ngram_map[cur_index] = ngram
                        cur_index += 1

    # add index for ngram
    for example in data:
        example['question_1_ngrams'] = [[ngram_to_index_map[ngram] if ngram in ngram_to_index_map.keys() else (hash(ngram) % 100) for ngram in ngrams]
                                        for ngrams in example['question_1_ngrams']]
        example['question_2_ngrams'] = [[ngram_to_index_map[ngram] if ngram in ngram_to_index_map.keys() else (hash(ngram) % 100)for ngram in ngrams]
                                        for ngrams in example['question_2_ngrams']]

    if build_ngram_map:
        return data, ngram_to_index_map, index_to_ngram_map
    else:
        return data


def word_to_char_ngrams(word, n=5, max_len=15):
    tmp = '#' + word + '#'
    if len(tmp) < n:
        return [tmp]
    else:
        return [tmp[3 * i: 3 * i + n] for i in range(min(len(tmp) - n + 1, max_len - n + 1) // 3)]


def load_ngram_vocab(path):
    ngram_to_index_map = {UNKNOWN: 0, PADDING: 1}
    index_to_ngram_map = {0: UNKNOWN, 1: PADDING}
    vocabulary = [UNKNOWN, PADDING]
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            ngram = line.strip()
            vocabulary.append(ngram)
            ngram_to_index_map[ngram] = i + 2
            index_to_ngram_map[i + 2] = ngram
    return vocabulary, ngram_to_index_map, index_to_ngram_map


def load_embedding_and_build_vocab(file_path):
    vocab = []
    word_embeddings = []
    for i in range(0, 100):
        oov_word = '<OOV' + str(i) + '>'
        vocab.append(oov_word)
        word_embeddings.append(list(np.random.normal(scale=1, size=300)))
    vocab.append('<PAD>')
    word_embeddings.append(list(np.zeros(300)))
    index_to_word = dict(enumerate(vocab))
    word_to_index = dict([(index_to_word[index], index) for index in index_to_word])

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.split()
            if len(line) < 301:
                continue
            word = ' '.join(line[:-300])
            vector = [float(x) for x in line[-300:]]
            vocab.append(word)
            word_embeddings.append(vector)
            word_to_index[word] = i + 101
            index_to_word[i + 101] = word

    return vocab, np.array(word_embeddings), word_to_index, index_to_word


def load_embedding_and_build_vocab_ngram(file_path, n=5):
    vocab = []
    word_embeddings = []
    for i in range(0, 100):
        oov_word = '<OOV' + str(i) + '>'
        vocab.append(oov_word)
        word_embeddings.append(list(np.random.normal(scale=1, size=300)))
    vocab.append('<PAD>')
    word_embeddings.append(list(np.zeros(300)))
    index_to_word = dict(enumerate(vocab))
    word_to_index = dict([(index_to_word[index], index) for index in index_to_word])
    ngram_number = 100
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.split()
            if len(line) < 301:
                continue
            word = ' '.join(line[:-300])
            vector = [float(x) for x in line[-300:]]
            n_gram_word = word_to_char_ngrams(word, n)
            vocab = list(set(vocab) | set(n_gram_word))
            for n_gram in n_gram_word:
                if n_gram in word_to_index:
                    # print(n_gram)
                    # print(word_to_index[n_gram])
                    word_embeddings[word_to_index[n_gram]] = [sum(i) for i in zip(word_embeddings[word_to_index[n_gram]], vector)]
                else:
                    ngram_number += 1
                    word_embeddings.append(vector)
                    word_to_index[n_gram] = ngram_number
                    index_to_word[ngram_number] = n_gram

    return vocab, np.array(word_embeddings), word_to_index, index_to_word


def semi_sort_data(data):
    return [example for example in data if example['max_length'] < 20] + \
           [example for example in data if 20 <= example['max_length'] < 50] +\
           [example for example in data if example['max_length'] >= 50]


def batch_iter(dataset, batch_size, use_ngram=False, shuffle=False):
    start = -1 * batch_size
    dataset_size = len(dataset)

    if shuffle:
        semi_sort_data(dataset)
        random.shuffle(dataset)

    index_list = list(range(len(dataset)))

    while True:
        start += batch_size
        label = []
        premise = []
        hypothesis = []
        max_length_ngram = 0
        if start > dataset_size - batch_size:
            start = 0
            if shuffle:
                semi_sort_data(dataset)
                random.shuffle(dataset)
        batch_indices = index_list[start:start + batch_size]
        batch = [dataset[index] for index in batch_indices]
        for k in batch:
            label.append(k['label'])
            if use_ngram:
                max_length_ngram = max([max_length_ngram,
                                        max([len(ngrams) for ngrams in k['question_1_ngrams']]),
                                        max([len(ngrams) for ngrams in k['question_2_ngrams']])])

            premise.append(k['premise_to_tokens'])
            hypothesis.append(k['hypothesis_to_tokens'])
        max_length_prem = max([len(item) for item in premise])
        max_length_hypo = max([len(item) for item in hypothesis])

        if use_ngram:
            for question in premise:
                question.extend([100] * (max_length_prem - len(question)))
                for ngrams in question:
                    ngrams.extend([100] * (max_length_ngram - len(ngrams)))
            for question in hypothesis:
                question.extend([100] * (max_length_hypo - len(question)))
                for ngrams in question:
                    ngrams.extend([100] * (max_length_ngram - len(ngrams)))
        else:
            for item in premise:
                item.extend([100] * (max_length_prem - len(item)))
            for item in hypothesis:
                item.extend([100] * (max_length_hypo - len(item)))

        yield [label, premise, hypothesis]
