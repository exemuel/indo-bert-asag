import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import os
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, LineTokenizer, RegexpTokenizer
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()
import string
from nltk.stem import PorterStemmer, WordNetLemmatizer
stopwords_indonesia = stopwords.words('indonesian')
stop_words = set(stopwords.words('indonesian'))


sheets_dict_Lifestyle = pd.read_excel('Data/Analisis_Essay_Grading_Lifestyle.xlsx', sheet_name=None, index_col=1, skiprows = 1)
sheets_dict_Olahraga = pd.read_excel('Data/Analisis_Essay_Grading_Olahraga.xlsx', sheet_name=None, index_col=1, skiprows = 1)
sheets_dict_Politik = pd.read_excel('Data/Analisis_Essay_Grading_Politik.xlsx', sheet_name=None, index_col=1, skiprows = 1)
sheets_dict_Teknologi = pd.read_excel('Data/Analisis_Essay_Grading_Teknologi.xlsx', sheet_name=None, index_col=1, skiprows = 1)

Lifestyle_keys = list(sheets_dict_Lifestyle.keys())
Olahraga_keys = list(sheets_dict_Olahraga.keys())
Politik_keys = list(sheets_dict_Politik.keys())
Teknologi_keys = list(sheets_dict_Teknologi.keys())


Lifestyle_Soal = sheets_dict_Lifestyle[Lifestyle_keys[0]].drop(['Unnamed: 0'], axis = 1)
Olahraga_Soal = sheets_dict_Olahraga[Olahraga_keys[0]].drop(['Unnamed: 0'], axis = 1)
Politik_Soal = sheets_dict_Politik[Politik_keys[0]].drop(['Unnamed: 0'], axis = 1)
Teknologi_Soal = sheets_dict_Teknologi[Teknologi_keys[0]].drop(['Unnamed: 0'], axis = 1)


Lifestyle_all_sheets = []
index = 1
for keys in Lifestyle_keys[index:len(Lifestyle_keys)-1]:
    new_sheets = sheets_dict_Lifestyle[keys].drop(['Unnamed: 0'], axis = 1)
    new_sheets = new_sheets.dropna()
    Lifestyle_all_sheets.append(new_sheets)

Olahraga_all_sheets = []
index = 1
for keys in Olahraga_keys[index:len(Olahraga_keys)-1]:
    new_sheets = sheets_dict_Olahraga[keys].drop(['Unnamed: 0'], axis = 1)
    new_sheets = new_sheets.dropna()
    Olahraga_all_sheets.append(new_sheets)



Politik_all_sheets = []
index = 1
for keys in Politik_keys[index:len(Politik_keys)-1]:
    new_sheets = sheets_dict_Politik[keys].drop(['Unnamed: 0'], axis = 1)
    new_sheets = new_sheets.dropna()
    Politik_all_sheets.append(new_sheets)

Teknologi_all_sheets = []
index = 1
for keys in Teknologi_keys[index:len(Teknologi_keys)-1]:
    new_sheets = sheets_dict_Teknologi[keys].drop(['Unnamed: 0'], axis = 1)
    new_sheets = new_sheets.dropna()
    Teknologi_all_sheets.append(new_sheets)
    