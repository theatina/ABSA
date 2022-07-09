import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from fun_lib import functions_data as funs_d
from fun_lib import functions_training as funs_t

class_algo_dict={
    "SVM": SVC(C=1.0, kernel="rbf",probability=True),
    "RF": make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=100)),
    "LR": LogisticRegression(C=5e-5, solver="lbfgs", max_iter=1000 )
}


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
    funs_d.add_feat_POS(df)
    funs_d.add_feat_freq(df)
    funs_d.add_feat_sentiment(df)
    funs_d.add_feat_ngrams(df,2)
    funs_d.add_feat_ngrams(df,3)
    funs_d.add_feat_ngrams(df,4)
    funs_d.add_feat_stemming(df)
    funs_d.add_feat_lemmas(df)
    funs_d.add_feat_CountVect(df)
    # funs_d.alpha_to_numerical(df,"op_target")

    file_name = f"allFeats"
    data_df_filepath = os.path.join(save_dir,f"opinions_polarity_{file_name}.csv")
    df.to_csv(data_df_filepath, index=False, header=True)

    return df

def data_proc(parts_to_use):
    
    data_df = load_df_parts(parts_to_use)
    labels_df, le = funs_d.alpha_to_numerical(data_df,"polarity")
    labels = labels_df["polarity"].values
    # all_data_df.pop("polarity")
    add_feats(data_df)

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(data_df, labels, test_size=0.3, random_state=99)

    return X_train, X_test, y_train, y_test


def data_proc_train_test_parts(train_parts, test_parts):
    
    data_df_train = load_df_parts(train_parts)
    data_df_test = load_df_parts(test_parts)

    labels_train_df, le = funs_d.alpha_to_numerical(data_df_train,"polarity")
    labels_test_df = labels_train_df["polarity"].values

    labels_test_df, le = funs_d.alpha_to_numerical(data_df_test,"polarity")
    labels_test_df = labels_test_df["polarity"].values

    # all_data_df.pop("polarity")
    add_feats(data_df_train)
    add_feats(data_df_test)

    # split dataset
    # X_train, X_test_noUse, y_train, y_test_noUse = train_test_split(labels_train_df, labels_train_df, test_size=0.0, random_state=99)
    # X_train_noUse, X_test, y_train_noUse, y_test = train_test_split(labels_test_df, labels_test_df, test_size=1.0, random_state=99)

    return data_df_train, data_df_test, labels_train_df, labels_test_df   

def evaluate_model_precision(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=4, random_state=1)
    scores = cross_val_score(model, X, y, scoring='precision', cv=cv, n_jobs=-1, error_score='raise')
    return scores

def evaluate_model_recall(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=4, random_state=1)
    scores = cross_val_score(model, X, y, scoring='recall', cv=cv, n_jobs=-1, error_score='raise')
    return scores

def evaluate_model_f1(model, X, y):
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=4, random_state=1)
    scores = cross_val_score(model, X, y, scoring='f1', n_jobs=-1, error_score='raise')
    return scores


def train(X_train, X_test, y_train, y_test, classificationAlgo="RF" ):
    model=class_algo_dict[classificationAlgo]
    model.fit(X_train, y_train)

    return model


if __name__=="__main__":

    if len(sys.argv)<2:
        print(f"\nERROR: did not specify number of parts !\n(default: all parts 1-10)\n")
        parts_to_use = [ i+1 for i in range(10)]
    else:
        parts_to_use = eval(sys.argv[1])

    X_train, X_test, y_train, y_test = data_proc(parts_to_use)
    algo_name="RF"
    train(X_train, X_test, y_train, y_test, algo_name)
