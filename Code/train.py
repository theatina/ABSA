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

# import skfeature
# from skfeature.function.similarity_based import fisher_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2, SelectKBest, mutual_info_classif, VarianceThreshold, mutual_info_regression

from fun_lib import functions_data as funs_d
from fun_lib import functions_training as funs_t
import test as tst

# load the xml parts for training/testing
def load_df_parts(embeddings, part_list=[i for i in range(10)]):
    save_dir = "..\Data\DataFrames"
    all_data_df = pd.DataFrame()
    for i in part_list:
        
        df = funs_d.data_df(f"..\Data\Parts\part{i}.xml", f"part{i}", save_dir, embeddings)
        all_data_df = pd.concat( [ all_data_df , df])

    return all_data_df

# add features for experimentation
def add_feats(df, pos_list=[], save_dir=f"..{os.sep}Data{os.sep}DataFrames"):
    pos_list = []
    # add features
    # df, pos_list = funs_d.add_feat_POS(df,pos_list)
    # funs_d.add_feat_freq(df)
    # df = funs_d.add_feat_sentiment(df)
    # df = funs_d.add_feat_ngrams(df,2)
    # df = funs_d.add_feat_ngrams(df,3)
    # df = funs_d.add_feat_ngrams(df,4)
    # df = funs_d.add_feat_stemming(df)
    # df = funs_d.add_feat_lemmas(df)
    # df = funs_d.add_feat_CountVect(df)

    file_name = f"allFeats"
    data_df_filepath = os.path.join(save_dir,f"opinions_polarity_{file_name}.csv")
    df.to_csv(data_df_filepath, index=False, header=True)
    return df, pos_list

# choose the features of the dataframe that will be used
def choose_feats(df):

    # text embeddings dimension
    dim = max( [ int(i) for i in df.columns if i.isnumeric() ] )+1

    # text embeddings features
    feats_to_keep = [ str(i) for i in range(dim) ]

    # useful features
    feats_to_keep.extend(["op_target"])
    feats_to_keep.extend(["op_entity"])
    feats_to_keep.extend(["op_attribute"])

    # feats_to_keep.extend(["POS"])
    # feats_to_keep.extend(["POS_mostCommon"])
    # feats_to_keep.extend(["sentiment"])
    # feats_to_keep.extend(["ngrams_2"])

    df = df[feats_to_keep]

    # encode the categorical features
    df, le_t = funs_d.alpha_to_numerical(df,"op_target")
    df, le_e = funs_d.alpha_to_numerical(df,"op_entity")
    df, le_a = funs_d.alpha_to_numerical(df,"op_attribute")

    return df


# data processing
def data_proc(parts_to_use, embeddings, pos_list=[]):
    
    data_df = load_df_parts(embeddings, parts_to_use)
    labels_df, le = funs_d.alpha_to_numerical(data_df,"polarity")
    labels = labels_df["polarity"].values
    data_df.drop('polarity', axis=1, inplace=True)
    
    data_df, pos_list = add_feats(data_df)
    data_df = choose_feats(data_df)
    
    # split dataset
    X_train_best, X_test_best, y_train, y_test = train_test_split(data_df, labels, test_size=0.2, random_state=99)

    # feature selection method and test
    # k_top_feats = int(X_train_best.shape[1]/2)
    # technique = "mutual_info_classif"
    # technique_short = "mic"

    # print(f"\nFeature Selection\nTechnique: {technique}\nFeat num: {k_top_feats}\n")
    # X_train_best, X_test_best = feat_selection(X_train_best,y_train,X_test_best,k_best=k_top_feats,technique=technique_short)

    return X_train_best, X_test_best, y_train, y_test


# train/test set data processing 
def data_proc_train_test_parts(train_parts, test_parts, embeddings, ):
    
    # load the parts selected for the training/testing
    data_df_train = load_df_parts(embeddings, train_parts)
    data_df_test = load_df_parts(embeddings, test_parts)

    # encode label (categorical - negative, neutral, positive)
    labels_train_df, le = funs_d.alpha_to_numerical(data_df_train,"polarity")
    labels_train_df = labels_train_df["polarity"].values
    
    labels_test_df, le = funs_d.alpha_to_numerical(data_df_test,"polarity")
    labels_test_df = labels_test_df["polarity"].values

    data_df_train.drop('polarity', axis=1, inplace=True)
    data_df_test.drop('polarity', axis=1, inplace=True)
    
    # add and choose the useful features
    pos_list = []
    data_df_train, pos_list_train = add_feats(data_df_train, pos_list = pos_list)
    data_df_test, pos_list = add_feats(data_df_test, pos_list=pos_list_train)
    
    X_train_best = choose_feats(data_df_train)
    X_test_best = choose_feats(data_df_test)
    
    
    # feature selection method and test
    # k_top_feats = int(X_train_best.shape[1]/2)
    # technique = "mutual_info_classif"
    # technique_short = "mic"

    # print(f"\nFeature Selection\nTechnique: {technique}\nFeat num: {k_top_feats}\n")
    # X_train_best, X_test_best = feat_selection(X_train_best,labels_train_df,X_test_best,k_best=k_top_feats,technique=technique_short)

    return X_train_best, X_test_best, labels_train_df, labels_test_df   


# unused metrics from previous versions
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

# Classification algorithms and their parameters after manual fine-tuning and research
class_algo_dict={
    "LR": make_pipeline(LR_scaler, LogisticRegression(C=0.1, multi_class='multinomial', solver="lbfgs", max_iter=1000 )),
    "SVM": make_pipeline(SVM_scaler, SVC(C=1.0, kernel="poly", degree=3, probability=True)),
    "RF": make_pipeline(RF_scaler, RandomForestClassifier(n_estimators=100, criterion="entropy"))
    
}

# model training - main training function used by the other .py files
def train(X_train, y_train, classificationAlgo="RF" ):
    print(classificationAlgo)
    model = class_algo_dict[classificationAlgo]
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_train)
    # print(classification_report(y_train, predictions, target_names=["Negative", "Neutral", "Positive"], zero_division=1))

    return model


# feature selection techniques to discover the importance of the features used
def feat_selection(X_train,y,X_test,k_best=100, technique="mir"):
    if technique=="mic":
        best_feats = mutual_info_classif(X_train,y)
        indices = (-best_feats).argsort()[:k_best]
        X_best = SelectKBest( mutual_info_classif, k=k_best).fit_transform(X_train,y)
        feat_name_list = [f for f in X_train.columns]
        feat_names = [feat_name_list[index] for index in indices]
    
    elif technique=="mir":
        best_feats = mutual_info_regression(X_train,y)
        indices = (-best_feats).argsort()[:k_best]
        X_best = SelectKBest( mutual_info_classif, k=k_best).fit_transform(X_train,y)
        feat_name_list = [f for f in X_train.columns]
        feat_names = [feat_name_list[index] for index in indices]

    return X_best, X_test[feat_names]


def save_model(model, filepath):
    pickle.dump(model, open(filepath, 'wb'))


if __name__=="__main__":
    parts_to_use = [ i+1 for i in range(10)]
    algo_name="RF"
    # text embeddings: Word2Vec, sBERT
    # embeddings = "sBERT"
    embeddings = "Word2Vec"

    if len(sys.argv)<2:
        print(f"\nERROR: did not specify number of parts and classification algorithm !\ndefault parts: 1-10\ndefault algorithm: RF\n\nUsage: python test.py < algorithm name >  < parts to use >\n")
    elif len(sys.argv)==2:
        algo_name = sys.argv[1]
    else:
        parts_to_use = eval(sys.argv[2])
        algo_name = sys.argv[1]

    parts_str = "_".join([str(i) for i in parts_to_use])


    # train/test dataset split and useful feature selection
    X_train, X_test, y_train, y_test = data_proc(parts_to_use, embeddings)

    # Training
    model = train(X_train, y_train, algo_name)
    # Model evaluation
    tst.test(X_test,y_test,model)

    filepath=f"..{os.sep}Models{os.sep}{algo_name}_parts_{parts_str}_{embeddings}.model"
    save_model(model,filepath)

    # technique = "mutual_info_classif"
    # technique_short = "mic"
    # k_top_feats = X_train.shape[1]
    filepath=f"..{os.sep}Models{os.sep}{algo_name}_parts_{parts_str}_{embeddings}.model"
    save_model(model,filepath)

