# coding: utf-8

import sys
import glob
import json
import torch
import pandas as pd
import re
from transformers import *
sys.path.append('/home/long8v/BERT-NER')
from bert import Ner


## Data Loading
path = '/home/long8v/ICDAR-2019-SROIE/task3/data/data_dict.pth'
data_dict = torch.load(path)
data_dict = {key:value[0] for key, value in data_dict.items()}

## inference

## model loading
model = Ner('/home/long8v/sroie_bert/experiment/{}'.format(input('your experiment name is ...')))
pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

re_int = re.compile('\d+')
re_float = re.compile('(\d+\.\d+)')
re_percent = re.compile('(\d+.?\d+%)')
re_date = re.compile('(\d{2}[/-]\d{2}[/-]\d{2,4})')
re_row = re.compile('\n')
re_dict = {re_float:' float ', re_percent:' percent ', re_date:' date ', re_int:' int ', re_row:' row '}

def re_text(text):
    for key, value in re_dict.items():
        text = key.sub(value, text)
    return text


def re_find_pattern(text):
    pattern = re.compile('float|percent|date|int|row')
    if pattern.findall(text):
        return True
    else:
        return False


def re_find_which_pattern(text):
    pattern = re.compile('float|percent|date|int|row')
    try:
        return pattern.findall(text)[0]
    except:
        return text


def get_re_text(text):
    patterns = {}
    for key, value in re_dict.items():
        patterns[value.strip()] = re.findall(key, text) 
        text = key.sub(value, text)
    return patterns


def get_tokenized_word(text):
    token_word = tokenizer.tokenize(text)
    return token_word


def preprocess(text):
    text = re_text(text)
    return text



preprocess_data = {}
re_data = {}
for key, value in data_dict.items():
    preprocess_data[key] = preprocess(value)
    re_data[key] = get_re_text(value)


result_data = {}
for key, value in preprocess_data.items():
    cut_index = 512
    try:
        while 1:
            if value[cut_index] != ' ':
                cut_index -= 1
            else:
                break
    except:
        pass
    result_data[key] = model.predict(value[:cut_index])
    result_data[key].extend(model.predict(value[cut_index:1024]))


import nltk
from nltk import pos_tag
from nltk.tree import Tree
from nltk.chunk import conlltags2tree


# In[124]:


def get_result_json(result_list):
    tokens = [result['word'] for result in result_list]
    tags = [result['tag'] for result in result_list]

    re_dict_json = defaultdict(int)
    result_json = defaultdict(list)
    re_dict_json = defaultdict(int)

        
    pos_tags = [pos for token, pos in pos_tag(tokens)]
    conlltags = [(token, pos, tg) for token, pos, tg in zip(tokens, pos_tags, tags)]
    ne_tree = conlltags2tree(conlltags)
    original_text = defaultdict(list)
    for subtree in ne_tree:
        original_string = []
        if type(subtree) == Tree:
            original_label = subtree.label()
            leaves = subtree.leaves()
        else:
            leaves = [subtree]
        for token, pos in leaves:
            token = token.replace('##', '')
            re_dict_json[re_find_which_pattern(token)] +=1 
            original_string.extend([(token, int(re_dict_json[re_find_which_pattern(token)]))])
        if original_string:
            try:
                original_text[original_label.lower()].append(original_string)
            except:
                pass
    return original_text



from collections import defaultdict

json_data = {}
for key, value in result_data.items():
    json_data[key] = get_result_json(value)
    new_json_data = {}
    for k, v in json_data[key].items():
        v = sorted(v, key=lambda e: -len(e))[0]
        words = [re_data[key][re_find_which_pattern(word)][count-1] 
                 if re_find_pattern(word) else word 
                 for word, count in v]
        if words:
            words = [re.escape(word) for word in words if word.strip()]
            pattern = '\s*'.join(words)
            try:
                v_with_space = list(filter(lambda e: e, re.findall(pattern, data_dict[key])))
                new_json_data[k] = v_with_space
                if not v_with_space:
                    print(key, k, words)
            except Exception as e:
                pass
    json_data[key] = new_json_data


new_json_data = {}
for key, value in json_data.items():
    new_dict = defaultdict(str)
    for k, v in value.items():
        if v:
            if k == 'total':
                try:
                    v = max(list(map(lambda e: float(e), v)))
                except ValueError:
                    pass
            else:
                v = v[0]
        else:
            v = ''
        new_dict[k] = str(v).replace('\n', ' ').replace('\t', ' ')
        if not new_dict['total']:
            new_dict['total'] = max(list(map(lambda e: float(e),re_data[key]['float'])))
    new_json_data[key] = new_dict


path = '/home/long8v/docrv2_sroie/submission/SROIE_example_t3'
for key, value in new_json_data.items():
    with open('{}/{}.txt'.format(path, key), 'w') as f:
        f.write('{\n')
        f.write('    "company": "{}",\n'.format(value['company']))
        f.write('    "date": "{}",\n'.format(value['date']))
        f.write('    "address": "{}",\n'.format(value['address']))
        f.write('    "total": "{}"\n'.format(value['total']))
        f.write('}')

## zip file saving

in_path = '/home/long8v/docrv2_sroie/submission/SROIE_example_t3'
out_path = '/home/long8v/docrv2_sroie/submission/SROIE_example_t3'
out_zip_file = '/home/long8v/docrv2_sroie/evaluation/task3/submit.zip'

import os
import zipfile


submission_zip = zipfile.ZipFile(out_zip_file, 'w')
for folder, subfolders, files in os.walk(in_path): 
    for file in files:
        submission_zip.write(os.path.join(folder, file), 
                             os.path.relpath(os.path.join(folder,file), out_path), 
                             compress_type = zipfile.ZIP_DEFLATED)
print('zip saved!')
submission_zip.close()
