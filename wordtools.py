# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:54:14 2017

@author: tkalliom
"""
from polyglot.mapping import Embedding
from nltk.corpus import gutenberg
from nltk.stem import SnowballStemmer
from gensim.models import Word2Vec
from polyglot.text import Text, Detector
from nltk.corpus import stopwords
from nltk.corpus import nps_chat
from gensim.models import Phrases
import nltk
from nltk.tokenize import word_tokenize
stopEN = set(stopwords.words('english'))
stopSE = set(stopwords.words('swedish'))
stSE = SnowballStemmer("swedish")
stEN = SnowballStemmer("english")


def engToken(sdata):
    bbb=[]
    teksti=Text(sdata)
    sent=teksti.sentences
    for sss in sent:
        aaa=[]
        for www in sss.words:
            if www not in stopEN:
                aaa.append(stEN.stem(www))
        bbb.append(aaa)
    return bbb



def engTokenA(sdata, allw):
    bbb=[]
    teksti=Text(sdata)
    sent=teksti.sentences
    for sss in sent:
        aaa=[]
        for www in sss.words:
            if (www not in stopEN) and (stEN.stem(www) in allw):
                aaa.append(stEN.stem(www))
        bbb.append(aaa)
    return bbb

def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features
posts = nltk.corpus.nps_chat.xml_posts()[:10000]
featuresets = [(dialogue_act_features(post.text), post.get('class'))
               for post in posts]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
all_words = nltk.corpus.nps_chat.words()

t='Is this correct?'
classifier = nltk.NaiveBayesClassifier.train(train_set)
test_sent_features = {word.lower(): (word in word_tokenize(t.lower())) for word in all_words}
print(classifier.classify(dialogue_act_features(t)))


