#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from scipy.stats import ks_2samp
from scipy.special import rel_entr
from scipy.stats import chisquare
import scipy.stats as st
from statistics import mean
import argparse


# In[ ]:


def stage2(model_name, tokenizer, seq, decoding_type, hp, device):
    
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id).cuda()
    
    seq_length = len(tokenizer(seq, return_tensors='pt')['input_ids'][0])
    input_ids = tokenizer(seq, return_tensors='pt').to(device)
    
    
    init_text = seq
    type_pred = "Greedy Search"
    for j in range(1,21):
        if hp == "None":
            temp_text = model.generate(**input_ids, max_length=seq_length+j)
            temp_text = tokenizer.decode(temp_text[0], skip_special_tokens=True)
        else:
            #print("Fuck")
            temp_text = model.generate(**input_ids, max_length=seq_length+j, num_beams = hp)
            temp_text = tokenizer.decode(temp_text[0], skip_special_tokens=True)

        if init_text in temp_text:
            init_text = temp_text
        else:
            type_pred = "Beam Search"
            break
            
    return type_pred


# In[ ]:


def stage2_beam_size_helper(model, tokenizer, seq, beam_size, device):
    
    
    seq_length = len(tokenizer(seq, return_tensors='pt')['input_ids'][0])
    input_ids = tokenizer(seq, return_tensors='pt').to(device)
    
    all_seq = []
    
    for j in range(1,25):

        temp_text = model.generate(**input_ids, max_length=seq_length+j, num_beams = beam_size)
        all_seq.append(tokenizer.decode(temp_text[0][seq_length], skip_special_tokens=True))
        

    unique_tokens = list(set(all_seq))   
    
    return unique_tokens


# In[ ]:


def stage2_beam_size(model_name, tokenizer, beam_size, device):
    
    tokenizer_1 = GPT2Tokenizer.from_pretrained(model_name)
    model_1 = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer_1.eos_token_id).cuda()

    max_val = 0
    sample = ""
    num_of_sample = 0

    for i in range(30,60):
        temp = test_df['Stories'][i].split()[2:10]
        temp = ' '.join(temp)
        seq_length = len(tokenizer_1(temp, return_tensors='pt')['input_ids'][0])
        input_ids = tokenizer_1(temp, return_tensors='pt').to(device)

        reference_tokens = []
        for j in range(20000):
            temp_text = model_1.generate(**input_ids, max_length=seq_length+1, do_sample=True, top_k = 0)
            temp_text = tokenizer_1.decode(temp_text[0][seq_length], skip_special_tokens=True)
            reference_tokens.append(temp_text)

        reference_tokens_df = count_tokens(reference_tokens)
        reference_tokens_df = reference_tokens_df.reset_index()   
        reference_tokens_df.drop(['index'], axis=1)    

        unique_tokens = stage2_beam_size_helper(model_1, tokenizer_1, temp, beam_size)


        for j in range(len(unique_tokens)):
            for k in range(len(reference_tokens_df)):
                if reference_tokens_df['words'][k] == unique_tokens[j]:
                    if k+1 > max_val:
                        max_val = k+1
                    break

        if max_val == b:
            sample = test_df['Stories'][i]
            num_of_sample = i
            break
    
    return max_value

