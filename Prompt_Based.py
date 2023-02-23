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


def gpt2_prompt_eng_temp(model_name, seq, prompt, temperature, num_of_queries, num_of_repetition):
    
    tokenizer_1 = GPT2Tokenizer.from_pretrained(model_name)
    model_1 = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer_1.eos_token_id).cuda()
    
    temp = seq.split()[:808]
    temp = ' '.join(temp)

    seq_length = len(tokenizer_1(temp, return_tensors='pt')['input_ids'][0])
    input_ids = tokenizer_1(temp, return_tensors='pt').to(device)


    reference_tokens = []
    for j in range(num_of_queries):

        temp_text = model_1.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = 0)
        temp_text = tokenizer_1.decode(temp_text[0][seq_length], skip_special_tokens=True)
        reference_tokens.append(temp_text)

    reference_tokens_df = count_tokens(reference_tokens)
    reference_tokens_df = reference_tokens_df.reset_index()   
    reference_tokens_df.drop(['index'], axis=1)
    
    
    
    temp = new_text.split()[:808]
    temp = ' '.join(temp)
    temp = prompt + temp
    seq_length = len(tokenizer_1(temp, return_tensors='pt')['input_ids'][0])
    input_ids = tokenizer_1(temp, return_tensors='pt').to(device)


    temp_values = []
    for k in range(num_of_repetition):
        victim_tokens = []
        for j in range(num_of_queries):

            temp_text = model_1.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = 0, temperature = temperature)
            temp_text = tokenizer_1.decode(temp_text[0][seq_length], skip_special_tokens=True)
            victim_tokens.append(temp_text)

        victim_tokens_df = count_tokens(victim_tokens)
        victim_tokens_df = victim_tokens_df.reset_index()   
        victim_tokens_df.drop(['index'], axis=1)

        temp_values.append((np.log(reference_tokens_df['freq'][0]/reference_tokens_df['freq'][1])/np.log(victim_tokens_df['freq'][0]/victim_tokens_df['freq'][1])))
        
     
    conf_interval = st.t.interval(alpha=0.95, df=len(temp_values)-1, loc=np.mean(temp_values), scale=st.sem(temp_values))
    estimated_value = mean(temp_values)
    
    return conf_interval, estimated_value


# In[ ]:


def gpt2_prompt_eng_topp(model_name, seq, prompt, p, num_of_queries, num_of_repetition):
    
    
    tokenizer_1 = GPT2Tokenizer.from_pretrained(model_name)
    model_1 = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer_1.eos_token_id).cuda()
    
    temp = seq.split()[:808]
    temp = ' '.join(temp)

    seq_length = len(tokenizer_1(temp, return_tensors='pt')['input_ids'][0])
    input_ids = tokenizer_1(temp, return_tensors='pt').to(device)


    reference_tokens = []
    for j in range(num_of_queries):

        temp_text = model_1.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = 0)
        temp_text = tokenizer_1.decode(temp_text[0][seq_length], skip_special_tokens=True)
        reference_tokens.append(temp_text)

    reference_tokens_df = count_tokens(reference_tokens)
    reference_tokens_df = reference_tokens_df.reset_index()   
    reference_tokens_df.drop(['index'], axis=1)
    
    
    
    temp = new_text.split()[:808]
    temp = ' '.join(temp)
    temp = prompt + temp
    seq_length = len(tokenizer_1(temp, return_tensors='pt')['input_ids'][0])
    input_ids = tokenizer_1(temp, return_tensors='pt').to(device)


    p_values = []
    for k in range(num_of_repetition):
        victim_tokens = []
        for j in range(num_of_queries):

            temp_text = model_1.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = 0, temperature = temperature)
            temp_text = tokenizer_1.decode(temp_text[0][seq_length], skip_special_tokens=True)
            victim_tokens.append(temp_text)

        victim_tokens_df = count_tokens(victim_tokens)
        victim_tokens_df = victim_tokens_df.reset_index()   
        victim_tokens_df.drop(['index'], axis=1)

        p_values.append(reference_tokens_df['freq'][0]/victim_tokens_df['freq'][0])
        
     
    conf_interval = st.t.interval(alpha=0.95, df=len(p_values)-1, loc=np.mean(p_values), scale=st.sem(p_values))
    estimated_value = mean(p_values)
    
    return conf_interval, estimated_value


# In[ ]:


def gpt3_prompt_eng_topp(model_name, seq, prompt, p, num_of_queries):
    
    idx_list = [186, 189, 221, 229, 231, 240, 252, 275]
    
    tokenizer_1 = GPT2Tokenizer.from_pretrained('gpt2')
    
    for idx in idx_list:
    
        temp_prompt1 = seq.split()[:idx]
        temp_prompt1 = ' '.join(temp_prompt1)

        all_tokens = []

        for i in range(num_of_queries):

            response = openai.Completion.create(
              model=model_name,
              prompt=temp_prompt1,
              temperature=1,
              max_tokens=3,
              top_p=1,
              frequency_penalty=0,
              presence_penalty=0
            )

            new_text = response.choices[0]['text']
            input_ids = tokenizer_1(new_text, return_tensors='pt')['input_ids'].to(device)
            all_tokens.append(tokenizer_1.decode(input_ids[0][0]))

        all_tokens_df = count_tokens(all_tokens)
        all_tokens_df = all_tokens_df.reset_index()   
        all_tokens_df.drop(['index'], axis=1)




        temp_prompt2 = text.split()[:idx]
        temp_prompt2 = ' '.join(temp_prompt2)
        temp_prompt2 = prompt + temp_prompt2

        all_tokens2 = []

        for i in range(num_of_queries):
            response = openai.Completion.create(
              model=model_name,
              prompt=temp_prompt2,
              temperature=1,
              max_tokens=3,
              top_p=p,
              frequency_penalty=0,
              presence_penalty=0
            )

            new_text = response.choices[0]['text']
            input_ids = tokenizer_1(new_text, return_tensors='pt')['input_ids'].to(device)
            all_tokens2.append(tokenizer_1.decode(input_ids[0][0]))

        all_tokens_df2 = count_tokens(all_tokens2)
        all_tokens_df2 = all_tokens_df2.reset_index()   
        all_tokens_df2.drop(['index'], axis=1) 


        sum = 0
        for i in range(len(all_tokens_df2)):

            for j in range(len(all_tokens_df)):
                if all_tokens_df['words'][j] == all_tokens_df2['words'][i]:
                    #iter += 1
                    sum += all_tokens_df['freq'][j]# + all_tokens_df['freq'][j-1]

        p_values.append(sum/10000)
    
    conf_interval = st.t.interval(alpha=0.95, df=len(p_values)-1, loc=np.mean(p_values), scale=st.sem(p_values))
    estimated_value = mean(p_values)
    
    return conf_interval, estimated_value

