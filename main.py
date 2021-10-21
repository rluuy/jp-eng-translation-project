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


from nltk.translate.bleu_score import sentence_bleu
from IPython.display import display


if __name__ == '__main__':
    print ("Sup")
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    #nltk.download()
    #nltk.download('punkt')

