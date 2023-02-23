#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


def gpt2_stage3_temp(model_name, seq, temperature, num_of_queries, num_of_repetition, device):
    
    torch.manual_seed(42)
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id).cuda()
    
    
    
    seq_length = len(tokenizer(seq, return_tensors='pt')['input_ids'][0])
    input_ids = tokenizer(seq, return_tensors='pt').to(device)


    temp_values = []
    for k in range(num_of_repetition):
        
        
        reference_tokens = []
        for j in range(num_of_queries):


            temp_text = model.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = 0)
            temp_text = tokenizer.decode(temp_text[0][seq_length], skip_special_tokens=True)
            reference_tokens.append(temp_text)

        reference_tokens_df = count_tokens(reference_tokens)
        reference_tokens_df = reference_tokens_df.reset_index()   
        reference_tokens_df.drop(['index'], axis=1)    


        victim_tokens = []
        for j in range(num_of_queries):



            temp_text = model.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = 0, temperature = temperature)
            temp_text = tokenizer.decode(temp_text[0][seq_length], skip_special_tokens=True)
            victim_tokens.append(temp_text)




        victim_tokens_df = count_tokens(victim_tokens)

        victim_tokens_df = victim_tokens_df.reset_index()

        victim_tokens_df.drop(['index'], axis=1)


        temp_values.append(np.log(reference_tokens_df['freq'][0]/reference_tokens_df['freq'][1])/np.log(victim_tokens_df['freq'][0]/victim_tokens_df['freq'][1]))
        
    
    conf_interval = st.t.interval(alpha=0.95, df=len(temp_values)-1, loc=np.mean(temp_values), scale=st.sem(temp_values))
    estimated_value = mean(temp_values)
    
    return conf_interval, estimated_value


# In[ ]:


def gpt2_stage3_temp_topk(model_name, seq, temperature, k, num_of_queries, num_of_repetition, device):
    
    torch.manual_seed(42)
   
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id).cuda()
    
    
    
    seq_length = len(tokenizer(seq, return_tensors='pt')['input_ids'][0])
    input_ids = tokenizer(seq, return_tensors='pt').to(device)


    temp_values = []
    for k in range(num_of_repetition):

        reference_tokens = []
        for j in range(num_of_queries):


            temp_text = model.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = 0)
            temp_text = tokenizer.decode(temp_text[0][seq_length], skip_special_tokens=True)
            reference_tokens.append(temp_text)

        reference_tokens_df = count_tokens(reference_tokens)
        reference_tokens_df = reference_tokens_df.reset_index()   
        reference_tokens_df.drop(['index'], axis=1)    


        victim_tokens = []
        for j in range(num_of_queries):



            temp_text = model.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = k, temperature = temperature)
            temp_text = tokenizer.decode(temp_text[0][seq_length], skip_special_tokens=True)
            victim_tokens.append(temp_text)




        victim_tokens_df = count_tokens(victim_tokens)

        victim_tokens_df = victim_tokens_df.reset_index()

        victim_tokens_df.drop(['index'], axis=1)


        temp_values.append(np.log(reference_tokens_df['freq'][0]/reference_tokens_df['freq'][1])/np.log(victim_tokens_df['freq'][0]/victim_tokens_df['freq'][1]))
        
    
    conf_interval = st.t.interval(alpha=0.95, df=len(temp_values)-1, loc=np.mean(temp_values), scale=st.sem(temp_values))
    estimated_value = mean(temp_values)
    
    return conf_interval, estimated_value


# In[ ]:


def gpt2_stage3_temp_NS(model_name, seq, temperature, p, num_of_queries, num_of_repetition, device):
    
    torch.manual_seed(42)
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id).cuda()
    
    
    
    seq_length = len(tokenizer(seq, return_tensors='pt')['input_ids'][0])
    input_ids = tokenizer(seq, return_tensors='pt').to(device)


    temp_values = []
    for k in range(num_of_repetition):

        reference_tokens = []
        for j in range(num_of_queries):


            temp_text = model.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = 0)
            temp_text = tokenizer.decode(temp_text[0][seq_length], skip_special_tokens=True)
            reference_tokens.append(temp_text)

        reference_tokens_df = count_tokens(reference_tokens)
        reference_tokens_df = reference_tokens_df.reset_index()   
        reference_tokens_df.drop(['index'], axis=1)    


        victim_tokens = []
        for j in range(num_of_queries):



            temp_text = model.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = 0, top_p = p, temperature = temperature)
            temp_text = tokenizer.decode(temp_text[0][seq_length], skip_special_tokens=True)
            victim_tokens.append(temp_text)




        victim_tokens_df = count_tokens(victim_tokens)

        victim_tokens_df = victim_tokens_df.reset_index()

        victim_tokens_df.drop(['index'], axis=1)


        temp_values.append(np.log(reference_tokens_df['freq'][0]/reference_tokens_df['freq'][1])/np.log(victim_tokens_df['freq'][0]/victim_tokens_df['freq'][1]))
        
    
    conf_interval = st.t.interval(alpha=0.95, df=len(temp_values)-1, loc=np.mean(temp_values), scale=st.sem(temp_values))
    estimated_value = mean(temp_values)
    
    return conf_interval, estimated_value


# In[ ]:


def gpt2_stage3_temp_topk_NS(model_name, seq, temperature, k, p, num_of_queries, num_of_repetition, device):
    
    torch.manual_seed(42)
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id).cuda()
    
    
    
    seq_length = len(tokenizer(seq, return_tensors='pt')['input_ids'][0])
    input_ids = tokenizer(seq, return_tensors='pt').to(device)


    temp_values = []
    for k in range(num_of_repetition):

        reference_tokens = []
        for j in range(num_of_queries):


            temp_text = model.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = 0)
            temp_text = tokenizer.decode(temp_text[0][seq_length], skip_special_tokens=True)
            reference_tokens.append(temp_text)

        reference_tokens_df = count_tokens(reference_tokens)
        reference_tokens_df = reference_tokens_df.reset_index()   
        reference_tokens_df.drop(['index'], axis=1)    


        victim_tokens = []
        for j in range(num_of_queries):



            temp_text = model.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = k, top_p = p, temperature = temperature)
            temp_text = tokenizer.decode(temp_text[0][seq_length], skip_special_tokens=True)
            victim_tokens.append(temp_text)




        victim_tokens_df = count_tokens(victim_tokens)

        victim_tokens_df = victim_tokens_df.reset_index()

        victim_tokens_df.drop(['index'], axis=1)


        temp_values.append(np.log(reference_tokens_df['freq'][0]/reference_tokens_df['freq'][1])/np.log(victim_tokens_df['freq'][0]/victim_tokens_df['freq'][1]))
        
    
    conf_interval = st.t.interval(alpha=0.95, df=len(temp_values)-1, loc=np.mean(temp_values), scale=st.sem(temp_values))
    estimated_value = mean(temp_values)
    
    return conf_interval, estimated_value


# In[ ]:


def gpt3_stage3_temp(model_name, seq, ref_prob1, ref_prob2, temperature, num_of_queries, num_of_repetition, device):
    
    torch.manual_seed(42)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    temp_values = []
    for j in range(num_of_repetition):

        victim_tokens = []
        for i in range(num_of_queries):

            response = openai.Completion.create(
              model=model_name,
              prompt=seq,
              temperature=temperature,
              max_tokens=2,
              top_p=1,
              frequency_penalty=0,
              presence_penalty=0
            )

            new_text = response.choices[0]['text']
            input_ids = tokenizer(new_text, return_tensors='pt')['input_ids'].to(device)
            victim_tokens.append(tokenizer.decode(input_ids[0][0]))

        victim_tokens_df = count_tokens(victim_tokens)
        victim_tokens_df = victim_tokens_df.reset_index()   
        victim_tokens_df.drop(['index'], axis=1) 

        temp_values.append(np.log(ref_prob1/ref_prob2)/np.log(victim_tokens_df['freq'][0]/victim_tokens_df['freq'][1]))
        
    conf_interval = st.t.interval(alpha=0.95, df=len(temp_values)-1, loc=np.mean(temp_values), scale=st.sem(temp_values))
    estimated_value = mean(temp_values)
    
    return conf_interval, estimated_value


# In[ ]:


def stage4(model_name, seq_examples, k, temperature, num_of_queries, num_of_repetition, device):

    
    torch.manual_seed(42)
   
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id).cuda()

    if temperature == None:
        
        temp_ks = []

        for text in seq_examples:

            seq_length = len(tokenizer(text, return_tensors='pt')['input_ids'][0])
            input_ids = tokenizer(text, return_tensors='pt').to(device)


            victim_tokens = []
            for j in range(num_of_queries):


                temp_text = model.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = k)
                temp_text = tokenizer.decode(temp_text[0][seq_length], skip_special_tokens=True)
                victim_tokens.append(temp_text)




            victim_tokens_df = count_tokens(victim_tokens)

            victim_tokens_df = victim_tokens_df.reset_index()


            victim_tokens_df.drop(['index'], axis=1)


            temp_ks.append(len(victim_tokens_df))



        if len(list(set(temp_ks))) != 1:
            res = "It is not Top_k"
        else:       
            res = "It is Top_k"
            k_pred = list(set(temp_ks))[0]
        
    
    
    else:
        
        temp_ks = []

        for text in seq_examples:

            seq_length = len(tokenizer(text, return_tensors='pt')['input_ids'][0])
            input_ids = tokenizer(text, return_tensors='pt').to(device)


            victim_tokens = []
            for j in range(num_of_queries):


                temp_text = model.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = k, temperature = temperature)
                temp_text = tokenizer.decode(temp_text[0][seq_length], skip_special_tokens=True)
                victim_tokens.append(temp_text)




            victim_tokens_df = count_tokens(victim_tokens)

            victim_tokens_df = victim_tokens_df.reset_index()


            victim_tokens_df.drop(['index'], axis=1)


            temp_ks.append(len(victim_tokens_df))



        if len(list(set(temp_ks))) != 1:
            res = "It is not Top_k and Temperature"
        else:       
            res = "It is Top_k and Temperature"
            k_pred = list(set(temp_ks))[0]
    
    
    return res, k_pred


# In[ ]:


def gpt2_stage5(model_name, seq, seq2, k, p, temperature, num_of_queries, num_of_repetition, device):
    
    
        
    torch.manual_seed(42)
   
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id).cuda()
    
    if temperature == None and k == 0:
        
        seq_length = len(tokenizer(seq, return_tensors='pt')['input_ids'][0])
        input_ids = tokenizer(seq, return_tensors='pt').to(device)


        p_values = []

        reference_tokens = []
        for j in range(num_of_queries):

            if j in flags:
                print(j)

            temp_text = model.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = 0)
            temp_text = tokenizer.decode(temp_text[0][seq_length], skip_special_tokens=True)
            reference_tokens.append(temp_text)

        reference_tokens_df = count_tokens(reference_tokens)
        reference_tokens_df = reference_tokens_df.reset_index()   
        reference_tokens_df.drop(['index'], axis=1)    



        for k in range(8):

            print(k)

            victim_tokens = []
            for j in range(num_of_queries):


                temp_text = model.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = 0, top_p = p)
                temp_text = tokenizer.decode(temp_text[0][seq_length], skip_special_tokens=True)
                victim_tokens.append(temp_text)




            victim_tokens_df = count_tokens(victim_tokens)

            victim_tokens_df = victim_tokens_df.reset_index()


            victim_tokens_df.drop(['index'], axis=1)



            p_values.append(reference_tokens_df['freq'][0]/victim_tokens_df['freq'][0])
    
        return mean(p_values)
    
    elif (k == 0) and (temperature < 1):
        
        
        seq_length = len(tokenizer(seq, return_tensors='pt')['input_ids'][0])
        input_ids = tokenizer(seq, return_tensors='pt').to(device)


        p_values = []

        reference_tokens = []
        for j in range(num_of_queries):

            temp_text = model.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = 0)
            temp_text = tokenizer.decode(temp_text[0][seq_length], skip_special_tokens=True)
            reference_tokens.append(temp_text)

        reference_tokens_df = count_tokens(reference_tokens)
        reference_tokens_df = reference_tokens_df.reset_index()   
        reference_tokens_df.drop(['index'], axis=1)    



        for k in range(8):


            victim_tokens = []
            for j in range(num_of_queries):


                temp_text = model.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = 0, temperature = temperature, top_p = p)
                temp_text = tokenizer.decode(temp_text[0][seq_length], skip_special_tokens=True)
                victim_tokens.append(temp_text)




            victim_tokens_df = count_tokens(victim_tokens)

            victim_tokens_df = victim_tokens_df.reset_index()


            victim_tokens_df.drop(['index'], axis=1)


            temperature_pred = np.log(reference_tokens_df['freq'][0]/reference_tokens_df['freq'][1])/np.log(victim_tokens_df['freq'][0]/victim_tokens_df['freq'][1])



            new_reference_tokens = []
            for j in range(num_of_queries):
                temp_text = model.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = 0, temperature = temperature_pred)
                temp_text = tokenizer.decode(temp_text[0][seq_length], skip_special_tokens=True)
                new_reference_tokens.append(temp_text)

            new_reference_tokens_df = count_tokens(new_reference_tokens)
            new_reference_tokens_df = new_reference_tokens_df.reset_index()   
            new_reference_tokens_df.drop(['index'], axis=1) 

            p_values.append(new_reference_tokens_df['freq'][0]/victim_tokens_df['freq'][0])
         
        return mean(p_values)
            
    elif (k > 0) and (temperature == None):
        
        
        seq_length = len(tokenizer(seq, return_tensors='pt')['input_ids'][0])
        input_ids = tokenizer(seq, return_tensors='pt').to(device)

        reference_tokens = []
        for j in range(100000):
            temp_text = model.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = 0)
            temp_text = tokenizer.decode(temp_text[0][seq_length], skip_special_tokens=True)
            reference_tokens.append(temp_text)

        reference_tokens_df = count_tokens(reference_tokens)
        reference_tokens_df = reference_tokens_df.reset_index()   
        reference_tokens_df.drop(['index'], axis=1) 
        
        
        seq_length = len(tokenizer(seq2, return_tensors='pt')['input_ids'][0])
        input_ids = tokenizer(seq2, return_tensors='pt').to(device)

        reference_tokens2 = []
        for j in range(100000):

            temp_text = model.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = 0)
            temp_text = tokenizer.decode(temp_text[0][seq_length], skip_special_tokens=True)
            reference_tokens2.append(temp_text)

        reference_tokens_df2 = count_tokens(reference_tokens2)
        reference_tokens_df2 = reference_tokens_df2.reset_index()   
        reference_tokens_df2.drop(['index'], axis=1) 
        
        
        
        seq_length = len(tokenizer(seq, return_tensors='pt')['input_ids'][0])
        input_ids = tokenizer(seq, return_tensors='pt').to(device)
        victim_tokens = []
        for j in range(100000):



            temp_text = model.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = k, top_p = p)
            temp_text = tokenizer.decode(temp_text[0][seq_length], skip_special_tokens=True)
            victim_tokens.append(temp_text)




        victim_tokens_df = count_tokens(victim_tokens)
        victim_tokens_df = victim_tokens_df.reset_index()
        victim_tokens_df.drop(['index'], axis=1)
        
        
        
        
        seq_length = len(tokenizer(seq2, return_tensors='pt')['input_ids'][0])
        input_ids = tokenizer(seq2, return_tensors='pt').to(device)
        victim_tokens2 = []
        for j in range(100000):

            if j in flags:
                print(j)

            temp_text = model.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = 30, top_p = 0.90)
            temp_text = tokenizer.decode(temp_text[0][seq_length], skip_special_tokens=True)
            victim_tokens2.append(temp_text)


        victim_tokens_df2 = count_tokens(victim_tokens2)
        victim_tokens_df2 = victim_tokens_df2.reset_index()
        victim_tokens_df2.drop(['index'], axis=1)
        
        
        
        
        num = 0
        sum_candidate = 0
        min_min = 100000
        candidates = []
        for i in range(10, 60):
            pred = 0
            sum = 0 
            for j in range(i):
                sum += reference_tokens_df['freq'][j]
            sum /= 100000

            S_k_1 = sum
            S_k_2 = (S_k_1*victim_tokens_df['freq'][0]*reference_tokens_df2['freq'][0])/(victim_tokens_df2['freq'][0]*reference_tokens_df['freq'][0])

            sum = 0
            min = 100000
            for k in range(100):

                sum += reference_tokens_df2['freq'][k]/100000
                if np.absolute(sum-S_k_2) < min:
                    min = np.absolute(sum-S_k_2)
                    pred = k

            if np.absolute(pred-i) <= min_min:
                min_min = np.absolute(pred-i)
                candidates.append(i)

        final_pred = 0
        for i in range(1,len(candidates)):
            final_pred += candidates[i]
        final_k_pred = int(final_pred/(len(candidates)-1))
        
        
        
        
        
        seq_length = len(tokenizer(seq2, return_tensors='pt')['input_ids'][0])
        input_ids = tokenizer(seq2, return_tensors='pt').to(device)
        victim_tokens_temp = []
        for j in range(100000):


            temp_text = model.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = 30, top_p = 0.90)
            temp_text = tokenizer.decode(temp_text[0][seq_length], skip_special_tokens=True)
            victim_tokens_temp.append(temp_text)




        victim_tokens_temp_df = count_tokens(victim_tokens_temp)
        victim_tokens_temp_df = victim_tokens_temp_df.reset_index()
        victim_tokens_temp_df.drop(['index'], axis=1)


        new_reference_tokens = []
        for j in range(100000):
            temp_text = model.generate(**input_ids, max_length=seq_length+2, do_sample=True, top_k = final_k_pred)
            temp_text = tokenizer.decode(temp_text[0][seq_length], skip_special_tokens=True)
            new_reference_tokens.append(temp_text)

        new_reference_tokens_df = count_tokens(new_reference_tokens)
        new_reference_tokens_df = new_reference_tokens_df.reset_index()   
        new_reference_tokens_df.drop(['index'], axis=1) 

        NS_parameter_pred = new_reference_tokens_df['freq'][0]/victim_tokens_temp_df['freq'][0]
        
        
        return final_k_pred, NS_parameter_pred
        
        


# In[ ]:


def gpt3_stage5(model_name, seq, ref_prob1, p, num_of_queries, num_of_repetition, device):
    
    
    torch.manual_seed(42)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


    p_values = []
    for j in range(8):

        victim_tokens = []
        for i in range(20000):
            if i in flags:
                print(i)
            #print(i)
            response = openai.Completion.create(
              model=model_name,
              prompt=seq,
              temperature=1,
              max_tokens=2,
              top_p=p,
              frequency_penalty=0,
              presence_penalty=0
            )

            new_text = response.choices[0]['text']
            input_ids = tokenizer(new_text, return_tensors='pt')['input_ids'].to(device)
            victim_tokens.append(tokenizer.decode(input_ids[0][0]))

        victim_tokens_df = count_tokens(victim_tokens)
        victim_tokens_df = victim_tokens_df.reset_index()   
        victim_tokens_df.drop(['index'], axis=1) 

        p_values.append(ref_prob1/victim_tokens_df['freq'][0])
    
    
    
    for i in range(len(p_values)):
        p_values[i] = p_values[i] * 200
    conf_interval = st.t.interval(alpha=0.95, df=len(p_values)-1, loc=np.mean(p_values), scale=st.sem(p_values))
    estimated_value = mean(p_values)
    
    return conf_interval, estimated_value

