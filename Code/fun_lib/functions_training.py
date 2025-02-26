import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from fun_lib import functions_data as funs_d
from fun_lib import functions_training as funs_t



def load_df_parts(part_list=[i for i in range(10)]):
    save_dir = "..\Data\DataFrames"
    all_data_df = pd.DataFrame()
    for i in part_list:
        
        df = funs_d.data_df(f"..\Data\Parts\part{i}.xml", f"part{i}", save_dir)
        all_data_df = pd.concat( [ all_data_df , df])
        
    # labels = all_data_df["polarity"].values
    # all_data_df.pop("polarity")

    return all_data_df


def add_feats(df, save_dir=f"..{os.sep}Data{os.sep}DataFrames"):

    # add features
    # funs_d.add_feat_POS(df)
    # funs_d.add_feat_freq(df)
    # funs_d.add_feat_sentiment(df)
    # funs_d.add_feat_ngrams(df,2)
    # funs_d.add_feat_ngrams(df,3)
    # funs_d.add_feat_ngrams(df,4)
    # funs_d.add_feat_stemming(df)
    # funs_d.add_feat_lemmas(df)
    # funs_d.add_feat_CountVect(df)
    funs_d.alpha_to_numerical(df,"op_target")

    file_name = f"allFeats"
    data_df_filepath = os.path.join(save_dir,f"opinions_polarity_{file_name}.csv")
    df.to_csv(data_df_filepath, index=False, header=True)

    return df

def choose_feats(df):

    dim = max( [ int(i) for i in df.columns if i.isnumeric() ] )+1

    # text embeddings
    feats_to_keep = [ str(i) for i in range(dim) ]

    feats_to_keep.extend(["op_target"])
    feats_to_keep.extend(["op_entity"])
    feats_to_keep.extend(["op_attribute"])

    df = df[feats_to_keep]

    funs_d.alpha_to_numerical(df,"op_target")
    funs_d.alpha_to_numerical(df,"op_entity")
    funs_d.alpha_to_numerical(df,"op_attribute")


    return df

def data_proc(parts_to_use):
    
    data_df = load_df_parts(parts_to_use)
    labels_df, le = funs_d.alpha_to_numerical(data_df,"polarity")
    labels = labels_df["polarity"].values
    data_df.pop("polarity")
    
    data_df = choose_feats(data_df)

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(data_df, labels, test_size=0.2, random_state=99)

    return X_train, X_test, y_train, y_test


def data_proc_train_test_parts(train_parts, test_parts):
    
    data_df_train = load_df_parts(train_parts)
    data_df_test = load_df_parts(test_parts)

    labels_train_df, le = funs_d.alpha_to_numerical(data_df_train,"polarity")
    labels_train_df = labels_train_df["polarity"].values
    
    labels_test_df, le = funs_d.alpha_to_numerical(data_df_test,"polarity")
    labels_test_df = labels_test_df["polarity"].values

    data_df_train.pop("polarity")
    data_df_test.pop("polarity")

    data_df_train = choose_feats(data_df_train)
    data_df_test = choose_feats(data_df_test)
    
    # all_data_df.pop("polarity")
    # add_feats(data_df_train)
    # add_feats(data_df_test)

    # split dataset
    # X_train, X_test_noUse, y_train, y_test_noUse = train_test_split(labels_train_df, labels_train_df, test_size=0.0, random_state=99)
    # X_train_noUse, X_test, y_train_noUse, y_test = train_test_split(labels_test_df, labels_test_df, test_size=1.0, random_state=99)

    return data_df_train, data_df_test, labels_train_df, labels_test_df   



if __name__=="__main__":

    print("\n")
