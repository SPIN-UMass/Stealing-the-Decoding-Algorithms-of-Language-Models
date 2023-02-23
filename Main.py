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
    
    parser.add_argument('--stage', type=int, default=0,
                        help='The index of the stage')
    
    parser.add_argument('--algorithm', type=str, default='Greedy Search',
                        help='The decoding algorithm used in the API')
    
    parser.add_argument('--targeted_hyperparameter', type=str, default='temperature',
                        help='The type of the targeted hyperparameter')
    
    parser.add_argument('--targeted_model', type=str, default='gpt2',
                        help='The type of the targeted model')
    
    parser.add_argument('--phase', type=str, default='base_model',
                        help='That is how the model is used by the API')
    
    parser.add_argument('--beam_size', type=int, default=1,
                        help='The beam size if beam search is used')
    
    parser.add_argument('--temperature', type=float, default=1,
                        help='The temperature if sampling with temperature is used')
    
    parser.add_argument('--top_k', type=int, default=0,
                        help='The hyperparameter k if top_k sampling is used')
    
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='The hyperparameter p if Nucleus Sampling is used')
    
    parser.add_argument('--number_of_queries', type=int, default=20000,
                        help='Number of queries used to estimate the hyperparameter')
    
    parser.add_argument('--number_of_repetitions', type=int, default=6,
                        help='Number of times the whole experiment repeated to estimate an confidence interval for the targeted hyperparameter')
    
    parser.add_argument('--ref_prob1', type=float, default=1,
                        help='The first reference probability when gpt-3 is used in the API')
    
    parser.add_argument('--ref_prob2', type=float, default=1,
                        help='The second reference probability when gpt-3 is used in the API')

    args = parser.parse_args()
    
    return args


# In[ ]:


device = torch.device("cuda")

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


# In[ ]:


#openai.api_key = "The Public Key"


# In[ ]:


args = parse_option()

if args.phase == "base_model":
    
    if args.stage == 2:

        seq = "Students opened their"

        estimated_type = stage2(args.targeted_model, args.targeted_model, seq, decoding_type, hp, device)

        if model_type == "Beam Search":

            estimated_beam_size = stage2_beam_size(args.targeted_model, args.targeted_model, args.beam_size, device)

            print("The estimated beam size: " + str(estimated_beam_size))
            
    elif args.stage == 3:

        if args.targeted_model == "gpt2":

            if args.algorithm == "temperature":
                
                
                seq = "Students opened their"

                estimated_confidence_interval, estimated_temperature = gpt2_stage3_temp(args.targeted_model, seq, args.temperature, args.number_of_queries, args.number_of_repetitions, device)

                print("The confidence interval: " + str(estimated_confidence_interval))
                print("The estimated temperature: " + str(estimated_temperature))
                
                
            elif args.algorithm == "temperature_top_k":

                seq = "Students opened their"

                estimated_confidence_interval, estimated_temperature = gpt2_stage3_temp_topk(args.targeted_model, seq, args.temperature, args.top_k, args.number_of_queries, args.number_of_repetitions, device)

                print("The confidence interval: " + str(estimated_confidence_interval))
                print("The estimated temperature: " + str(estimated_temperature))
                
            elif args.algorithm == "temperature_NS":

                seq = "Students opened their"

                estimated_confidence_interval, estimated_temperature = gpt2_stage3_temp_NS(args.targeted_model, seq, args.temperature, args.top_p, args.number_of_queries, args.number_of_repetitions, device)

                print("The confidence interval: " + str(estimated_confidence_interval))
                print("The estimated temperature: " + str(estimated_temperature))
                
            elif args.algorithm == "temperature_top_k_NS":

                seq = "Students opened their"

                estimated_confidence_interval, estimated_temperature = gpt2_stage3_temp_topk_NS(args.targeted_model, seq, args.temperature, args.top_k, args.top_p, args.number_of_queries, args.number_of_repetitions, device)

                print("The confidence interval: " + str(estimated_confidence_interval))
                print("The estimated temperature: " + str(estimated_temperature))
                

        elif args.targeted_model == "gpt-3":


            seq = "Students opened their"

            estimated_confidence_interval, estimated_temperature = gpt3_stage3_temp(args.targeted_model, seq, args.ref_prob1, args.ref_prob2, args.temperature, args.number_of_queries, args.number_of_repetitions, device)

            print("The confidence interval: " + str(estimated_confidence_interval))
            print("The estimated temperature: " + str(estimated_temperature))
            

    elif args.stage == 4:

        seq_examples = ['My', 'I love', 'It is a', 'I want to']

        estimated_k = stage4(args.targeted_model, seq_examples, args.top_k, args.temperature, args.number_of_queries, args.number_of_repetitions, device)
        
        print("The estimated k: " + str(estimated_k))

    elif args.stage == 5 or args.stage == 6:

        if args.targeted_model == 'gpt-2':

            seq = "Students opened their"

            seq = "My school is close to"

            estimated_k, estimated_p = gpt2_stage5(args.targeted_model, seq, seq2, args.top_k, args.top_p, args.temperature, args.number_of_queries, args.number_of_repetitions, device)

            print("The estimated k: " + str(estimated_k))
            print("The estimated p: " + str(estimated_p))

        elif args.targeted_model == "gpt-3":
            
            if args.algorithm == "Nucleus_Sampling":
                
                seq = "Students opened their"

                estimated_confidence_interval, estimated_value = gpt3_stage5(args.targeted_model, seq, args.ref_prob1, args.top_p, args.number_of_queries, args.number_of_repetitions, device)

                print("The confidence interval: " + str(estimated_confidence_interval))
                print("The estimated p: " + str(estimated_value))
            
elif args.phase == "prompt_based":
    
    if args.targeted_model == "gpt-2":
    
        if args.algorithm == "temperature":

            seq = "As a young girl, Sarah always felt like she was different from everyone else. She didn't quite fit in with the other kids her age, and she always had an active imagination. Little did she know, her imagination would soon become a reality. One day, while out exploring in the woods near her home, Sarah stumbled upon an old, dilapidated cottage. Curiosity getting the best of her, she decided to investigate. As she made her way inside, she was immediately struck by a feeling of power and magic. As she explored the cottage, she came across an ancient book hidden behind a loose stone in the wall. Sarah couldn't resist the urge to open it, and as she did, a bright light burst from the pages, enveloping her in its glow. When the light faded, Sarah felt a surge of energy coursing through her veins. She soon realized that she had been imbued with magical powers, and she knew that she had to use them for good. Sarah set out to learn how to control and harness her powers, and with each passing day, she grew stronger and more skilled. But she knew that she couldn't keep her powers a secret for long. One day, Sarah received a summons from the king, who had learned of her powers and needed her help. A dark sorceress was threatening to take over the kingdom, and Sarah was the only one who could stop her.  Sarah was nervous, but she knew that she couldn't let her fear get in the way of her duty. She accepted the mission and set out to confront the sorceress. As Sarah faced off against the dark sorceress, she knew that this was the moment that would define her. She summoned all of her magical energy and focused it into a single, powerful blast. The sorceress was no match for Sarah's strength, and she was defeated. The kingdom rejoiced at the news of the sorceress' defeat, and Sarah was hailed as a hero. She had saved the kingdom and proved that she was more than just a young woman with magical powers - she was a true warrior. Sarah returned home a changed person. She no longer felt like an outcast, but rather, a hero and a beacon of hope for others. And as she looked to the future, she knew that her adventures were far from over. She would continue to use her powers for good, and she would always be ready to defend her kingdom whenever it was in danger. Sarah's journey had only just begun, and she knew that there would be many challenges ahead. But she was ready to face them head on, armed with her magical powers and her unwavering determination to do good in the world. As she continued to hone her skills and learn more about her powers, Sarah became a powerful force for good in her kingdom. She used her magic to protect the innocent and bring justice to those who threatened the peace of the land. Despite the many dangers she faced, Sarah remained brave and dedicated to her cause. And as she traveled across the kingdom, she gained many loyal friends and allies who stood by her side and supported her on her journey. Through her bravery and selflessness, Sarah became a true hero and an inspiration to all those around her. She had discovered her true purpose in life and was determined to use her gifts for the greater good. And so, Sarah's adventures continued, as she worked to rid the kingdom of evil and bring about a brighter, more peaceful future for all. Sarah's adventures took her to far-off lands and through treacherous terrains, but she never lost sight of her purpose. She encountered all sorts of magical creatures and encountered powerful sorcerers and witches who sought to challenge her. But Sarah was not one to be underestimated. She had become a master of her powers, and she used them with great precision and skill. She battled fierce monsters and defeated powerful enemies, always emerging victorious. As she traveled, Sarah met many people who were in need of her help. She used her powers to heal the sick, protect the helpless, and bring hope to those who had lost it. She became known as a guardian of the innocent and a defender of justice, and her reputation grew with each passing day. Eventually, Sarah returned home to her kingdom, where she was greeted with great celebration. The people hailed her as a hero and thanked her for her bravery and selflessness. Sarah basked in the adoration of her people, but she knew that her work was far from over. She vowed to continue using her powers for good, and to always stand up for what was right, no matter the cost. And so, Sarah's adventures continued, as she traveled across the land, always ready to face whatever challenges came her way. She knew that with her magical powers and her strong sense of purpose, she could overcome anything."

            prompt = "Complete the story about a young woman who discovers she has magical powers and must learn how to use them to save her kingdom from a dark sorceress."

            estimated_confidence_interval, estimated_value = gpt2_prompt_eng_temp(args.targeted_model, seq, prompt, args.temperature, args.number_of_queries, args.number_of_repetitions, device)
            
            print("The confidence interval: " + str(estimated_confidence_interval))
            print("The estimated temperature: " + str(estimated_value))
            

        elif args.algorithm == "Nucleus_Sampling":

            seq = "As a young girl, Sarah always felt like she was different from everyone else. She didn't quite fit in with the other kids her age, and she always had an active imagination. Little did she know, her imagination would soon become a reality. One day, while out exploring in the woods near her home, Sarah stumbled upon an old, dilapidated cottage. Curiosity getting the best of her, she decided to investigate. As she made her way inside, she was immediately struck by a feeling of power and magic. As she explored the cottage, she came across an ancient book hidden behind a loose stone in the wall. Sarah couldn't resist the urge to open it, and as she did, a bright light burst from the pages, enveloping her in its glow. When the light faded, Sarah felt a surge of energy coursing through her veins. She soon realized that she had been imbued with magical powers, and she knew that she had to use them for good. Sarah set out to learn how to control and harness her powers, and with each passing day, she grew stronger and more skilled. But she knew that she couldn't keep her powers a secret for long. One day, Sarah received a summons from the king, who had learned of her powers and needed her help. A dark sorceress was threatening to take over the kingdom, and Sarah was the only one who could stop her.  Sarah was nervous, but she knew that she couldn't let her fear get in the way of her duty. She accepted the mission and set out to confront the sorceress. As Sarah faced off against the dark sorceress, she knew that this was the moment that would define her. She summoned all of her magical energy and focused it into a single, powerful blast. The sorceress was no match for Sarah's strength, and she was defeated. The kingdom rejoiced at the news of the sorceress' defeat, and Sarah was hailed as a hero. She had saved the kingdom and proved that she was more than just a young woman with magical powers - she was a true warrior. Sarah returned home a changed person. She no longer felt like an outcast, but rather, a hero and a beacon of hope for others. And as she looked to the future, she knew that her adventures were far from over. She would continue to use her powers for good, and she would always be ready to defend her kingdom whenever it was in danger. Sarah's journey had only just begun, and she knew that there would be many challenges ahead. But she was ready to face them head on, armed with her magical powers and her unwavering determination to do good in the world. As she continued to hone her skills and learn more about her powers, Sarah became a powerful force for good in her kingdom. She used her magic to protect the innocent and bring justice to those who threatened the peace of the land. Despite the many dangers she faced, Sarah remained brave and dedicated to her cause. And as she traveled across the kingdom, she gained many loyal friends and allies who stood by her side and supported her on her journey. Through her bravery and selflessness, Sarah became a true hero and an inspiration to all those around her. She had discovered her true purpose in life and was determined to use her gifts for the greater good. And so, Sarah's adventures continued, as she worked to rid the kingdom of evil and bring about a brighter, more peaceful future for all. Sarah's adventures took her to far-off lands and through treacherous terrains, but she never lost sight of her purpose. She encountered all sorts of magical creatures and encountered powerful sorcerers and witches who sought to challenge her. But Sarah was not one to be underestimated. She had become a master of her powers, and she used them with great precision and skill. She battled fierce monsters and defeated powerful enemies, always emerging victorious. As she traveled, Sarah met many people who were in need of her help. She used her powers to heal the sick, protect the helpless, and bring hope to those who had lost it. She became known as a guardian of the innocent and a defender of justice, and her reputation grew with each passing day. Eventually, Sarah returned home to her kingdom, where she was greeted with great celebration. The people hailed her as a hero and thanked her for her bravery and selflessness. Sarah basked in the adoration of her people, but she knew that her work was far from over. She vowed to continue using her powers for good, and to always stand up for what was right, no matter the cost. And so, Sarah's adventures continued, as she traveled across the land, always ready to face whatever challenges came her way. She knew that with her magical powers and her strong sense of purpose, she could overcome anything."

            prompt = "Complete the story about a young woman who discovers she has magical powers and must learn how to use them to save her kingdom from a dark sorceress."

            estimated_confidence_interval, estimated_value = gpt2_prompt_eng_topp(args.targeted_model, seq, prompt, args.top_p, args.number_of_queries, args.number_of_repetitions, device)
            
            print("The confidence interval: " + str(estimated_confidence_interval))
            print("The estimated p: " + str(estimated_value))
     
    elif args.targeted_model == "gpt-3":
        
        
        if args.algorithm == "Nucleus_Sampling":
            
            seq = "As a young girl, Sarah always felt like she was different from everyone else. She didn't quite fit in with the other kids her age, and she always had an active imagination. Little did she know, her imagination would soon become a reality. One day, while out exploring in the woods near her home, Sarah stumbled upon an old, dilapidated cottage. Curiosity getting the best of her, she decided to investigate. As she made her way inside, she was immediately struck by a feeling of power and magic. As she explored the cottage, she came across an ancient book hidden behind a loose stone in the wall. Sarah couldn't resist the urge to open it, and as she did, a bright light burst from the pages, enveloping her in its glow. When the light faded, Sarah felt a surge of energy coursing through her veins. She soon realized that she had been imbued with magical powers, and she knew that she had to use them for good. Sarah set out to learn how to control and harness her powers, and with each passing day, she grew stronger and more skilled. But she knew that she couldn't keep her powers a secret for long. One day, Sarah received a summons from the king, who had learned of her powers and needed her help. A dark sorceress was threatening to take over the kingdom, and Sarah was the only one who could stop her.  Sarah was nervous, but she knew that she couldn't let her fear get in the way of her duty. She accepted the mission and set out to confront the sorceress. As Sarah faced off against the dark sorceress, she knew that this was the moment that would define her. She summoned all of her magical energy and focused it into a single, powerful blast. The sorceress was no match for Sarah's strength, and she was defeated. The kingdom rejoiced at the news of the sorceress' defeat, and Sarah was hailed as a hero. She had saved the kingdom and proved that she was more than just a young woman with magical powers - she was a true warrior. Sarah returned home a changed person. She no longer felt like an outcast, but rather, a hero and a beacon of hope for others. And as she looked to the future, she knew that her adventures were far from over. She would continue to use her powers for good, and she would always be ready to defend her kingdom whenever it was in danger. Sarah's journey had only just begun, and she knew that there would be many challenges ahead. But she was ready to face them head on, armed with her magical powers and her unwavering determination to do good in the world. As she continued to hone her skills and learn more about her powers, Sarah became a powerful force for good in her kingdom. She used her magic to protect the innocent and bring justice to those who threatened the peace of the land. Despite the many dangers she faced, Sarah remained brave and dedicated to her cause. And as she traveled across the kingdom, she gained many loyal friends and allies who stood by her side and supported her on her journey. Through her bravery and selflessness, Sarah became a true hero and an inspiration to all those around her. She had discovered her true purpose in life and was determined to use her gifts for the greater good. And so, Sarah's adventures continued, as she worked to rid the kingdom of evil and bring about a brighter, more peaceful future for all. Sarah's adventures took her to far-off lands and through treacherous terrains, but she never lost sight of her purpose. She encountered all sorts of magical creatures and encountered powerful sorcerers and witches who sought to challenge her. But Sarah was not one to be underestimated. She had become a master of her powers, and she used them with great precision and skill. She battled fierce monsters and defeated powerful enemies, always emerging victorious. As she traveled, Sarah met many people who were in need of her help. She used her powers to heal the sick, protect the helpless, and bring hope to those who had lost it. She became known as a guardian of the innocent and a defender of justice, and her reputation grew with each passing day. Eventually, Sarah returned home to her kingdom, where she was greeted with great celebration. The people hailed her as a hero and thanked her for her bravery and selflessness. Sarah basked in the adoration of her people, but she knew that her work was far from over. She vowed to continue using her powers for good, and to always stand up for what was right, no matter the cost. And so, Sarah's adventures continued, as she traveled across the land, always ready to face whatever challenges came her way. She knew that with her magical powers and her strong sense of purpose, she could overcome anything."

            prompt = "Complete the story about a young woman who discovers she has magical powers and must learn how to use them to save her kingdom from a dark sorceress."

            
            estimated_confidence_interval, estimated_value = gpt3_prompt_eng_topp(args.targeted_model, seq, prompt, args.top_p, args.number_of_queries, device)
            
            print("The confidence interval: " + str(estimated_confidence_interval))
            print("The estimated p: " + str(estimated_value))
            


# In[ ]:




