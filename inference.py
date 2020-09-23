#!/usr/bin/env python
# coding: utf-8

# In[156]:


import sys
import glob
import json
import torch
import pandas as pd
sys.path.append('/home/long8v/BERT-NER')
from bert import Ner


path = '/home/long8v/ICDAR-2019-SROIE/task3/data/test_dict.pth'
data_dict = torch.load(path)


import re
import pickle
from transformers import *


model = Ner('/home/long8v/sroie_data/raw_bert_replace_special_token')
tokenizer = model.tokenizer
re_int = re.compile('\d+')
re_float = re.compile('(\d+\.\d+)')
re_percent = re.compile('(\d+.?\d+%)')
re_date = re.compile('(\d{2}[/-]\d{2}[/-]\d{2,4})')



re_dict = {re_float:'float', re_percent:'percent', re_date:'date', re_int:'int'}


# In[170]:


def re_text(text):
    for key, value in re_dict.items():
        text = key.sub(value, text)
    return text


# In[171]:


def get_tokenized_word(text):
    token_word = tokenizer.tokenize(text)
    return token_word


# In[172]:


def preprocess(text):
    text = re_text(text)
    # text = get_tokenized_word(text)
    return text


# In[173]:


preprocess_data = {}
for key, value in data_dict.items():
    preprocess_data[key] = preprocess(value)




result_data = {}
for key, value in preprocess_data.items():
    try:
        result_data[key] = model.predict(value[:512])
        result_data[key].extend(model.predict(value[512:1024]))
    except Exception as e :
        print('fail {}'.format(key))
        print(e)


from collections import defaultdict


# In[152]:
def get_result_json(result_list):
    result_json = defaultdict(list)
    for sample in result_list:
        tag = sample['tag'].lower()
        word = sample['word']
        word = word.replace('##', '')
        if tag != 'o':
            try:
                result_json[tag.split('-')[1]] += [word]
            except:
                print(tag)
    result_json = {key:' '.join(value)
                    for key, value in result_json.items()}
    return dict(result_json)


json_data = {}
for key, value in result_data.items():
    json_data[key] = get_result_json(value)


path = '/home/long8v/docrv2_sroie/submission/SROIE_example_t3'
for key, value in json_data.items():
    with open('{}/{}.txt'.format(path, key), 'w') as f:
        json.dump(value, f)
        print('{} saved.'.format(key))
