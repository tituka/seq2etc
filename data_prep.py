# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:24:08 2018

@author: tkalliom
"""
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

st = SnowballStemmer("english")

stop_words = set(stopwords.words('english'))


def process_word_data(token_list_list, stem=False, replace_d=True, rem_stop_words=True):
    new_list_list=[]
    for token_list in token_list_list:
        new_list=[]
        for token in token_list:
            new_token=token
            if stem:
                new_token=st.stem(token)
            if replace_d:
                new_token=replace_digits(new_token)
            new_list.append(new_token)
        new_list_list.append(new_list)
    if rem_stop_words==True:
        new_list_list=stop_words_out(new_list_list)
    return(new_list_list)
           
                
def stop_words_out(token_list_list):
    #removes stopwords from list of list of tokems
    new_list_list=[]
    for token_list in token_list_list:
        new_list=[x for x in token_list if x.lower() not in stop_words.union(['u', 'ur'])]
        new_list_list.append(new_list)
    return new_list_list   

 
def replace_digits(token):
    #replaces all digits in token with #
    return(re.sub('\d', '#', token))
   
