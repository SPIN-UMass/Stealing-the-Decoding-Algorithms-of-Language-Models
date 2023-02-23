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
from Sampling_Methods import gpt2_stage3_temp, gpt2_stage3_temp_topk, gpt2_stage3_temp_NS, gpt2_stage3_temp_topk_NS, gpt3_stage3_temp, stage4, gpt2_stage5, gpt3_stage5
from Prompt_Based import gpt2_prompt_eng_temp, gpt2_prompt_eng_topp, gpt3_prompt_eng_topp
from utils import count_tokens, new_dist_metric, create_hist 
from collections import Counter


# In[ ]:


def parse_option():
    parser = argparse.ArgumentParser('Decoding Algorithm Stealing')
    
    
    parser.add_argument('--algorithm', type=str, default='Greedy Search',
                        help='The decoding algorithm used in the API')
    
    parser.add_argument('--targeted_model', type=str, default='gpt2',
                        help='The type of the targeted model')
    
    parser.add_argument('--original_k', type=int, default=0,
                        help='The hyperparameter k in the original distribution')
    
    parser.add_argument('--original_p', type=float, default=1,
                        help='The hyperparameter p in the original distribution')
    
    parser.add_argument('--original_temperature', type=float, default=1,
                        help='The hyperparameter temperature in the original distribution')
    
    parser.add_argument('--estimated_k', type=int, default=0,
                        help='The estimated k')
    
    parser.add_argument('--estimated_p', type=float, default=1,
                        help='The estimated p')
    
    parser.add_argument('--estimated_temperature', type=float, default=1,
                        help='The estimated temperature')
    
    
    parser.add_argument('--number_of_queries', type=int, default=20000,
                        help='Number of queries used to provide distributions')
    
    

    args = parser.parse_args()
    
    return args


# In[ ]:


device=torch.device("cuda")

df = pd.read_csv('6_genre_eval_data.txt', header=None, delimiter = "\t")
df.columns = ['Stories']

# To make sure all samples have a permitted length

long_tokens = []
long_tokens_ids = []
tokenizer = GPT2Tokenizer.from_pretrained('gpt2',    
                            bos_token='<BOS>',
                            eos_token='<EOS>',
                            pad_token='<|pad|>')
for i in range(len(df)):
    text = df.iloc[i,0]
    encoded = tokenizer(text)
    if len(encoded['input_ids']) >= 1022:
        long_tokens.append(text)
        long_tokens_ids.append(i)
        
for i in long_tokens:
    df.drop(df.index[df['Stories'] == i], inplace=True)
    

df = df.reset_index()
df.drop(['index'], axis=1)


#openai.api_key = "The Public Key"


# In[ ]:


def KL_evaluation(model_name, test_dataset, k1, p1, temperature1, k2, p2, temperature2, num_of_queries):
    
    
    torch.manual_seed(42)
   
    tokenizer_1 = GPT2Tokenizer.from_pretrained(model_name)
    model_1 = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id).cuda()
    
    
    print("Correct")
    
    KL_scores = []
    chi_scores = []
    for i in [9, 20]:


        for j in [0, 10, 25]:
            temp = test_dataset['Stories'][i].split()[2:10+j]
            temp = ' '.join(temp)
            seq_length = len(tokenizer_1(temp, return_tensors='pt')['input_ids'][0])
            input_ids = tokenizer_1(temp, return_tensors='pt').to(device)
            original_tokens_1 = []
            for k in range(num_of_queries):

                temp_text = model_1.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = k1, top_p = p1, temperature = temperature1)
                temp_text = tokenizer_1.decode(temp_text[0][seq_length], skip_special_tokens=True)
                original_tokens_1.append(temp_text)

            original_tokens_1_df = count_tokens(original_tokens_1)
            original_tokens_1_df = original_tokens_1_df.reset_index()   
            original_tokens_1_df.drop(['index'], axis=1)


            predicted_tokens_1 = []
            for k in range(num_of_queries):

                temp_text = model_1.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = k2, top_p = p2, temperature = temperature2)
                temp_text = tokenizer_1.decode(temp_text[0][seq_length], skip_special_tokens=True)
                predicted_tokens_1.append(temp_text)

            predicted_tokens_1_df = count_tokens(predicted_tokens_1)
            predicted_tokens_1_df = predicted_tokens_1_df.reset_index()   
            predicted_tokens_1_df.drop(['index'], axis=1)

            ls_new1, ls_new2 = new_dist_metric(original_tokens_1_df, predicted_tokens_1_df)

            ls_new1 = [(element / num_of_queries) for element in ls_new1]
            ls_new2 = [(element / num_of_queries) for element in ls_new2]

            print(sum(rel_entr(ls_new2, ls_new1)))

            KL_scores.append(sum(rel_entr(ls_new2, ls_new1)))
    
    conf_interval = st.t.interval(alpha=0.95, df=len(KL_scores)-1, loc=np.mean(KL_scores), scale=st.sem(KL_scores))
    estimated_value = mean(KL_scores)
    
    return conf_interval, estimated_value
    


# In[ ]:


def KS_evaluation(model_name, test_dataset, k1, p1, temperature1, k2, p2, temperature2, num_of_queries):
    
    
    torch.manual_seed(42)
   
    tokenizer_1 = GPT2Tokenizer.from_pretrained(model_name)
    model_1 = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id).cuda()
    
    
    
    
    KL_scores = []
    chi_scores = []
    for i in [9, 20]:


        for j in [0, 10, 25]:
            temp = test_dataset['Stories'][i].split()[2:10+j]
            temp = ' '.join(temp)
            seq_length = len(tokenizer_1(temp, return_tensors='pt')['input_ids'][0])
            input_ids = tokenizer_1(temp, return_tensors='pt').to(device)
            original_tokens_1 = []
            for k in range(num_of_queries):

                temp_text = model_1.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = k1, top_p = p1, temperature = temperature1)
                temp_text = tokenizer_1.decode(temp_text[0][seq_length], skip_special_tokens=True)
                original_tokens_1.append(temp_text)

            original_tokens_1_df = count_tokens(original_tokens_1)
            original_tokens_1_df = original_tokens_1_df.reset_index()   
            original_tokens_1_df.drop(['index'], axis=1)


            predicted_tokens_1 = []
            for k in range(num_of_queries):

                temp_text = model_1.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = k2, top_p = p2, temperature = temperature2)
                temp_text = tokenizer_1.decode(temp_text[0][seq_length], skip_special_tokens=True)
                predicted_tokens_1.append(temp_text)

            predicted_tokens_1_df = count_tokens(predicted_tokens_1)
            predicted_tokens_1_df = predicted_tokens_1_df.reset_index()   
            predicted_tokens_1_df.drop(['index'], axis=1)

            l1 = create_hist(original_tokens_1)
            l2 = create_hist(predicted_tokens_1)

            statistic, p_value = ks_2samp(l1, l2)
            print(p_value)
            KS_scores.append(p_value)
    
    conf_interval = st.t.interval(alpha=0.95, df=len(KS_scores)-1, loc=np.mean(KS_scores), scale=st.sem(KS_scores))
    estimated_value = mean(KS_scores)
    
    return conf_interval, estimated_value
    


# In[ ]:

args = parse_option()


KL_confidence_interval, KL_score = KL_evaluation(args.targeted_model, df, args.original_k, args.original_p, args.original_temperature, args.estimated_k, args.estimated_p, args.estimated_temperature, args.number_of_queries)

KS_confidence_interval, KS_score = KS_evaluation(args.targeted_model, df, args.original_k, args.original_p, args.original_temperature, args.estimated_k, args.estimated_p, args.estimated_temperature, args.number_of_queries)

print("The confidence interval of KL_score: " + str(KL_confidence_interval))
print("The KL_score: " + str(KL_score))
print("The confidence interval of KS_score: " + str(KS_confidence_interval))
print("The KS_score: " + str(KS_score))

