import time
import MeCab
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import nltk
import ssl

jp_eng_file = "jpn.txt"
cleaned_text = "cleaned_jp_en.txt"

from nltk.translate.bleu_score import sentence_bleu
from IPython.display import display


def clean_text():
    data_file = pd.read_table(jp_eng_file, names=['en-target', 'jpn-source', 'noise-data'])
    data_file = data_file.iloc[::2]
    data_file = data_file.reindex(columns = ['jpn-source', 'en-target'])
    data_file.insert(1, 'seperator', ['\t' for i in range (data_file.shape[0])])
    np_array = data_file.to_numpy()
    np.savetxt(cleaned_text, np_array, fmt="%s", encoding='utf-8')
    print(("[Created] {}".format(cleaned_text)))

def ntlk_checker():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

if __name__ == '__main__':
    clean_text()

