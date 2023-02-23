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
from collections import Counter


# In[ ]:


def count_tokens(my_list):
    
    counts = Counter(my_list)

    unique_my_list = list(set(my_list))

    random_words_dict = {}
    for i in unique_my_list:
        random_words_dict[i] = counts[i]

    df = pd.DataFrame(list(random_words_dict.items()),columns = ['words','freq'])
    df = df.sort_values(by ='freq', ascending=False )

    df['rank'] = df['freq'].rank(ascending=False)
    
    return df


# In[ ]:


def new_dist_metric(df1, df2):
    
    new_df1 = df1.sort_values(by ='words', ascending=False)
    new_df2 = df2.sort_values(by ='words', ascending=False)
    
    ls1 = []
    ls2 = []
    
    if len(new_df1) >= len(new_df2):
        length = len(new_df1)
        df_max = new_df1
        df_min = new_df2
    else:
        length = len(new_df2)
        df_max = new_df2
        df_min = new_df1

    for i in range(length):
        
        ls1.append(df_max['freq'][i])
        if i >= len(df_min):
            ls2.append(0)
        else:
            ls2.append(df_min['freq'][i])
        
    return ls1, ls2


# In[ ]:


def create_hist(ls):
    
    all_vocabs = []
    for i in range(50257):
        all_vocabs.append(0)
        
    for i in range(len(ls)):
        
        all_vocabs[ls[i]] += 1
        
    return all_vocabs


# In[ ]:




