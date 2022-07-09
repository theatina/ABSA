import os
import sys

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

from fun_lib import functions_data as funs_d
from fun_lib import functions_training as funs_t

if len(sys.argv)<2:
    print(f"\nERROR: did not specify number of parts !\n(default: all parts 1-10)\n")
    parts_to_use = [ i for i in range(10)]
else:
    parts_to_use = sys.argv[1]



