import pandas as pd
import os
import nltk

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
    
    data['sentence1_token'] = data.apply(lambda row: tokenize_sentence(row['sentence1_parse']), axis=1)
    data['sentence2_token'] = data.apply(lambda row: tokenize_sentence(row['sentence2_parse']), axis=1)
    
    data['sentence1_size'] =  data.apply(lambda row: len(row['sentence1_token']), axis=1)
    data['sentence2_size'] = data.apply(lambda row: len(row['sentence2_token']), axis=1)
    data['max_size'] = data.apply(lambda row: max(row['sentence1_size'], row['sentence2_size']), axis=1)
    data['onehot_label'] = data.apply(lambda row: label_function(row['gold_label']), axis=1)
    data = data.sort_values(by = 'max_size')
    return data[['sentence1_token', 'sentence2_token', 'sentence1_size', 'sentence2_size', 'onehot_label']]

def pad_sentence(token_list, pad_length):
    """
    PAD 0 in front of the token_list
    return padded_list
    """

    padding = [0] * (pad_length - len(token_list))
    padded_list = padding + token_list
    return padded_list

