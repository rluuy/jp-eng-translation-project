import time
import MeCab
import numpy as np
import pandas as pd
import os.path
from os import path
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import nltk
import ssl
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

jp_eng_file = "jpn.txt"                            # Original input file. Contains JP -> Eng Translations
cleaned_text = "cleaned_jp_en.txt"                 # Our new output file after cleaning
lower_chars = [chr(i) for i in range (97,123)]     # all lower case english letters
mecab = MeCab.Tagger("-Owakati")

from nltk.translate.bleu_score import sentence_bleu
from IPython.display import display

'''
clean_text() uses pandas to clean up the specific file of jpn.txt. 

After its cleaned, it wil create a new text file named "cleaned_jp_en.txt". 

===============
Original Format
===============
Go.	行け。	CC-BY 2.0 (France) Attribution: tatoeba.org #2877272 (CM) & #7421985 (Ninja)
Go.	行きなさい。	CC-BY 2.0 (France) Attribution: tatoeba.org #2877272 (CM) & #7421986 (Ninja)

===============
New Format
===============

行け。 	 Go.
こんにちは。 	 Hi.

'''

def clean_text():
    data_file = pd.read_table(jp_eng_file, names=['en-target', 'jpn-source', 'noise-data'])
    data_file = data_file.iloc[::2]
    data_file = data_file.reindex(columns = ['jpn-source', 'en-target'])
    data_file.insert(1, 'seperator', ['\t' for i in range (data_file.shape[0])])
    np_array = data_file.to_numpy()
    np.savetxt(cleaned_text, np_array, fmt="%s", encoding='utf-8')
    print(("[Created] {}".format(cleaned_text)))

def tokenize_text():
    try:
        with open('cleaned_jp_en.txt', mode='rt', encoding='utf-8') as f:
            lines = f.read().split("\n")
    except FileNotFoundError:
        print("File does not exist. Abort Soldier.")

    jpn_data = list()  # Holds our JP data while processing and tokenizing
    en_data = list()  # Holds our EN data with processing and tokenizing

    for i in range (0, len(lines)):
        is_lower = False
        line = lines[i].split("\t")
        for char in line[0].strip():
            if char.lower() in lower_chars:
                is_lower = True
        if is_lower == False:
            jpn_line = analyze_jpn(line[0].strip(), mecab)
            jpn_data.append(jpn_line)
            eng_line = analyze_eng(line[-1].strip())
            en_data.append(eng_line)

    jpn_data = jpn_data[::-1]
    en_data = en_data[::-1]

    #print(len(jpn_data))
    #print(len(en_data))

    # Adds tag to Beginning of Sentence and End of Sentence
    for index, jpn_line in enumerate(jpn_data):
        jpn_data[index] = ["<bos>"] + jpn_line + ["<eos>"]

    for index, en_line in enumerate(en_data):
        en_data[index] = ["<bos>"] + en_line + ["<eos>"]

    source_sentence_tokenizer, source_tensor = tokenize_helper(np.array(jpn_data))
    target_sentence_tokenizer, target_tensor = tokenize_helper(np.array(en_data))
    source_tensor = tf.keras.preprocessing.sequence.pad_sequences(source_tensor, padding='post')
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, padding='post')

    source_vocab_size = len(source_sentence_tokenizer.word_index) + 1
    target_vocab_size = len(target_sentence_tokenizer.word_index) + 1
    max_len_input = source_tensor.shape[1]
    max_len_target = target_tensor.shape[1]

    #print("[SOURCE] Japanese vocabulary size:  " + str(source_vocab_size))
    #print("[TARGET] English vocabulary size:  " + str(target_vocab_size))
    #print("[SOURCE] Max Japanese Sentence Length:  " + str(max_len_input))
    #print("[TARGET] Max English Sentence Length :  " + str(max_len_target))


    input_tensor_train, input_tensor_test, target_tensor_train, target_tensor_test = train(source_tensor,target_tensor)


    #print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_test), len(target_tensor_test))




def tokenize_helper(word):
    word_tk = tf.keras.preprocessing.text.Tokenizer(filters='', char_level = False)
    word_tk.fit_on_texts(word)
    return word_tk, word_tk.texts_to_sequences(word)


def train(s_tensor, t_tensor):
    return train_test_split(s_tensor,t_tensor, test_size=0.2)

def convert_ID_to_words(language, tensor):
    ids = []
    words = []
    for t in tensor:
        if t!= 0:
            ids.append(t)
            words.append(language.index_word[t])
    return ids,words


def analyze_jpn(sentence, mecab):
    jp_line = mecab.parse(sentence).split(' ')
    jp_line.remove('\n')
    return jp_line

def analyze_eng(sentence):
    return nltk.word_tokenize(sentence.lower())


# Deals with weird error regarding not being able to download NTLK stuff
def ntlk_checker():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

if __name__ == '__main__':
    #clean_text()
    tokenize_text()



