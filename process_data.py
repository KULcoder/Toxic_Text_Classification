import pandas as pd
import numpy as np
import langid
from sklearn.model_selection import train_test_split

def is_english(text):
    try:
        lang, _ = langid.classify(text)
        return lang == 'en'
    except:
        return False
# identifier = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def load_data():
    # only load the balanced train data, split into .9 : .1 for validation
    # redecuced the computation expanse

    train_data = pd.read_csv('data/train.csv')
    train_data = train_data[train_data['comment_text'].apply(is_english)]

    positive_data = train_data[train_data['toxic'] == 1]
    negative_data = train_data[train_data['toxic'] == 0]

    min_length = min(len(positive_data), len(negative_data))

    positive_data = positive_data.sample(min_length)
    negative_data = negative_data.sample(min_length)

    output = pd.concat([positive_data, negative_data]).sample(frac=1)
    X, y = output['comment_text'].to_numpy(), output['toxic'].to_numpy()

    return train_test_split(X, y, test_size=0.1)

def load_test_data():
    test_data = pd.read_csv('data/train.csv')
    test_data['label'] = test_data['toxic']

    test_data = test_data[test_data['comment_text'].apply(is_english)]
    
    # try to balance the data base on test positive fraction
    positive_data = test_data[test_data['label'] == 1]
    negative_data = test_data[test_data['label'] == 0]

    min_length = min(len(positive_data), len(negative_data))

    positive_data = positive_data.sample(min_length)
    negative_data = negative_data.sample(min_length)

    output = pd.concat([positive_data, negative_data]).sample(frac=1)
    return output['comment_text'].to_numpy(), output['label'].to_numpy()
    
    
def load_train_data():
    train_data = pd.read_csv('data/test.csv')
    train_data['label'] = pd.read_csv('data/test_labels.csv')['toxic']

    train_data = train_data[train_data['comment_text'].apply(is_english)]
    
    # try to balance the data base on test positive fraction
    positive_data = train_data[train_data['label'] == -1]
    positive_data['label'] = (positive_data['label'] == -1)*1 # transform from -1 to 1
    negative_data = train_data[train_data['label'] == 0]

    min_length = min(len(positive_data), len(negative_data))

    positive_data = positive_data.sample(min_length)
    negative_data = negative_data.sample(min_length)

    output = pd.concat([positive_data, negative_data]).sample(frac=1)
    return output['comment_text'].to_numpy(), output['label'].to_numpy()