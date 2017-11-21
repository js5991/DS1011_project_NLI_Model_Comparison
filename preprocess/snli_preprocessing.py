import pandas as pd
import os
import nltk
import random
import collections
import numpy as np
import pickle

### read data ###
def read_data_set(data_path):
    """
    read dataset from data_path
    return a dataset with 5 columns of tokens of sentence 1 and 2, sizes of tokens, and one hot vector as label
    data with gold_label of "-" are removed
    """
    
    def tokenize_sentence(parse):
        """
        Read the given parse string
        return a list of token with NULL prepended
        """
        parse = parse.lower()
        tree = nltk.Tree.fromstring(parse)
        token_list = tree.leaves()
        token_list.insert(0,"NULL")
        return token_list
    
    def tokenize_sentence_from_binary_parse(parse):
        """
        @param parse: a binary parse of the sentence
        @return: token of the parsed sentence
        """
        token_list = ["<NULL>"]
        if parse:
            token_list  = token_list + parse.lower().replace("(", "").replace(")", "").strip().split()
        return token_list
        
    
    def label_function (label):
        """
        Read a label of entailment, contradiction, and neutral
        Return a onehot vector of [1,0,0] for entailment, [0,1,0] for contradiction, and  [0,0,1] for neutral
        """
        labels = {'entailment':0, 'contradiction':1, 'neutral':2}
        try:
            onehot = [0,0,0]
            onehot[labels[label]] = 1
            return onehot
        except:
            return None
    
    data = pd.read_csv(data_path,delimiter="\t")
    data = data[data["gold_label"] != "-"]
    
    data['sentence1_token'] = data.apply(lambda row: tokenize_sentence_from_binary_parse(row['sentence1_binary_parse']), axis=1)
    data['sentence2_token'] = data.apply(lambda row: tokenize_sentence_from_binary_parse(row['sentence2_binary_parse']), axis=1)
    
    data['sentence1_size'] =  data.apply(lambda row: len(row['sentence1_token']), axis=1)
    data['sentence2_size'] = data.apply(lambda row: len(row['sentence2_token']), axis=1)
    data['max_size'] = data.apply(lambda row: max(row['sentence1_size'], row['sentence2_size']), axis=1)
    data['min_size'] = data.apply(lambda row: min(row['sentence1_size'], row['sentence2_size']), axis=1)
    
    data = data[data["min_size"] > 1]
    
    data['onehot_label'] = data.apply(lambda row: label_function(row['gold_label']), axis=1)
    return data[['sentence1_token', 'sentence2_token', 'sentence1_size', 'sentence2_size', 'onehot_label', 'max_size']]


###Embeddings ####
def loadGloveData(gloveFile):
    '''
    @para gloveFile: directory of glove.6B.300d.txt
    @return: glove embedding dictionary
    '''
    f = open(gloveFile,'r')
    dic = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        dic[word] = embedding
    return dic

def build_vocabulary_with_glove(dataframe, glove_dic, vocabulary_size = 50000):
    '''
    build the vocabulary
    @para dataframe: training data set
    @return: index_to_word_map, word_to_index_map
    '''
    word_counter = {}
    for tokens in dataframe['sentence1_token']:
        for token in tokens:
            if token in glove_dic:
                if token in word_counter:
                    word_counter[token]+= 1
                else:
                    word_counter[token] = 1
        
    for tokens in dataframe['sentence2_token']:
        for token in tokens:
            if token in glove_dic:
                if token in word_counter:
                    word_counter[token]+= 1
                else:
                    word_counter[token] = 1 

    vocabulary = sorted(word_counter, key=lambda key: word_counter[key], reverse=True)[0:vocabulary_size - 100]
    for i in range(1,101): 
        oov_word = '<oov'+ str(i) + '>'
        vocabulary.append(oov_word)
        
    index_to_word_map = dict(enumerate(vocabulary))
    word_to_index_map = dict([(index_to_word_map[index], index) for index in index_to_word_map])
    
    #load glove embedding and initialize oov embedding
    word2vec_embedding = np.random.normal(size = (len(index_to_word_map), 300))
    for i in range(len(index_to_word_map)):
        if index_to_word_map[i] in glove_dic:
            word2vec_embedding[i] = glove_dic[index_to_word_map[i]]
            
    for i in range(len(word2vec_embedding)):
        word2vec_embedding[i] = word2vec_embedding[i]/np.linalg.norm(word2vec_embedding[i])
    
    return index_to_word_map, word_to_index_map, word2vec_embedding

def save_embedding(embedding, file_name):
    '''
    save_embedding
    @param embedding: embedding matrix
    @para file_name: file name to save
    '''
    with open(file_name, 'wb') as handle:
        pickle.dump(embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)
   
    
### Batch ###
def batch_iter(dataframe, batch_size):
    
    def pad_sentence(token_list, pad_length):
        """
        PAD in front of the token_list
        @param token_list:
        @param pad_length: 
        @return: padded_list
        """
        padding = ["<PAD>"] * (pad_length - len(token_list))
        padded_list = padding + token_list
        return padded_list
    
    def shuffle_based_on_length (dataframe):
        '''
        It shuffles the index based on the length of the sentences
        @param data_frame: a data frame
        @return: list of shuffled index
        '''
        index_of_len_less_20 = dataframe.index[dataframe['max_size'] < 20].tolist()
        index_other = dataframe.index[dataframe['max_size'] > 50].tolist()
        index_of_len_less_50 = list(set(dataframe.index.tolist()) - set(index_of_len_less_20) - set(index_other))
        

        random.shuffle(index_of_len_less_20)
        random.shuffle(index_of_len_less_50)
        random.shuffle(index_other)

        index = index_of_len_less_20 + index_of_len_less_50 + index_other
        return index

    start = - 1 * batch_size
    index_list = shuffle_based_on_length (dataframe)
    dataset_size = len(index_list)
    
    while True:
        start += batch_size
        
        if start > dataset_size - batch_size:
            # Start another epoch.
            start = 0
            index_list = shuffle_based_on_length (dataframe)
        batch_indices = index_list[start:start + batch_size]
        batch = dataframe.loc[batch_indices]
        max_length = batch['max_size'].max()
        batch['padded_sentence1'] = batch.apply(lambda row: pad_sentence(row['sentence1_token'], max_length), axis=1)
        batch['padded_sentence2'] = batch.apply(lambda row: pad_sentence(row['sentence2_token'], max_length), axis=1)
            
        yield [batch['padded_sentence1'].tolist(), batch['padded_sentence2'].tolist(), batch['onehot_label'].tolist()]
    


    