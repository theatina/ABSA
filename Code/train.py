import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
from matplotlib import pyplot as plt

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from sklearn.cluster import k_means
from sympy import evaluate

import skfeature
from skfeature.function.similarity_based import fisher_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2, SelectKBest, mutual_info_classif

from fun_lib import functions_data as funs_d
from fun_lib import functions_training as funs_t
import test as tst


def load_df_parts(embeddings, part_list=[i for i in range(10)]):
    save_dir = "..\Data\DataFrames"
    all_data_df = pd.DataFrame()
    for i in part_list:
        
        df = funs_d.data_df(f"..\Data\Parts\part{i}.xml", f"part{i}", save_dir, embeddings)
        all_data_df = pd.concat( [ all_data_df , df])
        
    # labels = all_data_df["polarity"].values
    # all_data_df.pop("polarity")

    return all_data_df


def add_feats(df, save_dir=f"..{os.sep}Data{os.sep}DataFrames"):

    # add features
    df = funs_d.add_feat_POS(df)
    # funs_d.add_feat_freq(df)
    df = funs_d.add_feat_sentiment(df)
    df = funs_d.add_feat_ngrams(df,2)
    df = funs_d.add_feat_ngrams(df,3)
    df = funs_d.add_feat_ngrams(df,4)
    df = funs_d.add_feat_stemming(df)
    df = funs_d.add_feat_lemmas(df)
    df = funs_d.add_feat_CountVect(df)


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
    # feats_to_keep.extend(["POS"])
    # feats_to_keep.extend(["POS_mostCommon"])
    # feats_to_keep.extend(["sentiment"])
    # feats_to_keep.extend(["ngrams_2"])

    df = df[feats_to_keep]

    df, le_t = funs_d.alpha_to_numerical(df,"op_target")
    df, le_e = funs_d.alpha_to_numerical(df,"op_entity")
    df, le_a = funs_d.alpha_to_numerical(df,"op_attribute")

    return df

def data_proc(parts_to_use, embeddings):
    
    data_df = load_df_parts(embeddings, parts_to_use)
    labels_df, le = funs_d.alpha_to_numerical(data_df,"polarity")
    labels = labels_df["polarity"].values
    # data_df.pop("polarity")
    data_df.drop('polarity', axis=1, inplace=True)
    
    data_df = add_feats(data_df)
    data_df = choose_feats(data_df)
    

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(data_df, labels, test_size=0.2, random_state=99)

    return X_train, X_test, y_train, y_test


def data_proc_train_test_parts(train_parts, test_parts, embeddings):
    
    data_df_train = load_df_parts(embeddings, train_parts)
    data_df_test = load_df_parts(embeddings, test_parts)

    labels_train_df, le = funs_d.alpha_to_numerical(data_df_train,"polarity")
    labels_train_df = labels_train_df["polarity"].values
    
    labels_test_df, le = funs_d.alpha_to_numerical(data_df_test,"polarity")
    labels_test_df = labels_test_df["polarity"].values

    # data_df_train.pop("polarity")
    # data_df_test.pop("polarity")
    data_df_train.drop('polarity', axis=1, inplace=True)
    data_df_test.drop('polarity', axis=1, inplace=True)
    
    data_df_train = add_feats(data_df_train)
    data_df_test = add_feats(data_df_test)
    
    data_df_train = choose_feats(data_df_train)
    data_df_test = choose_feats(data_df_test)
    
    # all_data_df.pop("polarity")
    

    # split dataset
    # X_train, X_test_noUse, y_train, y_test_noUse = train_test_split(labels_train_df, labels_train_df, test_size=0.0, random_state=99)
    # X_train_noUse, X_test, y_train_noUse, y_test = train_test_split(labels_test_df, labels_test_df, test_size=1.0, random_state=99)

    return data_df_train, data_df_test, labels_train_df, labels_test_df   

def evaluate_model_precision(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=4, random_state=1)
    scores = cross_val_score(model, X, y, scoring='precision_macro', cv=cv, n_jobs=-1, error_score='raise')
    return scores

def evaluate_model_recall(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=4, random_state=1)
    scores = cross_val_score(model, X, y, scoring='recall_macro', cv=cv, n_jobs=-1, error_score='raise')
    return scores

def evaluate_model_f1(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=4, random_state=1)
    scores = cross_val_score(model, X, y, scoring='f1_macro', cv=cv, n_jobs=-1, error_score='raise')
    return scores

# Data Scaling
SVM_scaler = MinMaxScaler()
RF_scaler = StandardScaler()
LR_scaler = StandardScaler()

class_algo_dict={
    "SVM": make_pipeline(SVM_scaler, SVC(C=1.0, kernel="poly", degree=3)),
    "RF": make_pipeline(RF_scaler, RandomForestClassifier(n_estimators=100, criterion="entropy")),
    "LR": make_pipeline(LR_scaler, LogisticRegression(C=0.1, multi_class='multinomial', solver="lbfgs", max_iter=1000 ))
}

def train(X_train, y_train, classificationAlgo="RF" ):
    print(classificationAlgo)
    model = class_algo_dict[classificationAlgo]
    model.fit(X_train, y_train)
    
    # evaluate_model_recall(model, X_train, y_train)
    # evaluate_model_precision(model, X_train, y_train)
    # evaluate_model_f1(model, X_train, y_train)
    
    predictions = model.predict(X_train)
    print(classification_report(y_train, predictions, target_names=["Negative", "Neutral", "Positive"], zero_division=1))

    return model

def feat_selection(X_train,y,X_test,k_best=100):
    best_feats = mutual_info_classif(X_train,y)
    indices = (-best_feats).argsort()[:k_best]
    X_best = SelectKBest( mutual_info_classif, k=k_best).fit_transform(X_train,y)
    feat_name_list = [f for f in X_train.columns]
    feat_names = [feat_name_list[index] for index in indices]
    # print(f"\n{k_best} Best Feats\n{feat_names}")
    return X_best, X_test[feat_names]

def save_model(model, filepath):
    pickle.dump(model, open(filepath, 'wb'))

if __name__=="__main__":
    parts_to_use = [ i+1 for i in range(10)]
    algo_name="RF"
    # text embeddings: Word2Vec, sBERT, FastText, CountVect, TfidfVect
    embeddings = "sBERT"

    if len(sys.argv)<2:
        print(f"\nERROR: did not specify number of parts and classification algorithm !\ndefault parts: 1-10\ndefault algorithm: RF\n\nUsage: python test.py < algorithm name >  < parts to use >\n")
    elif len(sys.argv)==2:
        algo_name = sys.argv[1]
    else:
        parts_to_use = eval(sys.argv[2])
        algo_name = sys.argv[1]

    # initial manually selected features
    X_train, X_test, y_train, y_test = data_proc(parts_to_use, embeddings)
    model = train(X_train, y_train, algo_name)
    tst.test(X_test,y_test,model)

    # X_train_fisher = fisher_score.fisher_score(X_train.values, y_train)
    # ft_importancies = pd.Series(X_train_fisher)
    # ft_importancies.plot(kind="barh", color="teal")
    # plt.show()
    
    # feature selection method and test
    X_train_best, X_test_best = feat_selection(X_train,y_train,X_test)
    model = train(X_train_best, y_train, algo_name)
    tst.test(X_test_best,y_test,model)

    parts_str = "_".join([str(i) for i in parts_to_use])
    
    filepath=f"..{os.sep}Models{os.sep}{algo_name}_parts_{parts_str}_{embeddings}.model"
    save_model(model,filepath)
   

