# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 19:04:53 2018

@author: tkalliom
"""

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stopEN = set(stopwords.words('english'))

with open('answers.txt', 'r', encoding='UTF-8', errors='ignore') as  read_q:
    with open('out_q.txt', 'w', encoding='UTF-8', errors='ignore') as  out_q:
        with open('out_a.txt', 'w', encoding='UTF-8', errors='ignore') as out_a:
            q_lines=read_q.readlines()
            for q_line in q_lines:
                tokens=word_tokenize(q_line)
                qa_pair=[[],[]]
                some_letters=False
                for token in tokens:
                    if token not in stopEN:
                        qa_pair[0].append(token)
                    qa_pair[1].append(token)
                    if token.isalpha():
                        some_letters=True
                if some_letters:
                    out_q.write(' '.join(qa_pair[0]) + '\n')
                    out_a.write(q_line)