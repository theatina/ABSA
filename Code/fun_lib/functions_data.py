import os
from regex import F
from requests import head
from sympy import arg
import sys
import pandas as pd
import re

from bs4 import BeautifulSoup

from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, FreqDist, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.util import ngrams

from sklearn.preprocessing import LabelEncoder

import functions_training as funs_t


def tok_text(text):

    tok = TweetTokenizer()
    tokens = tok.tokenize(text)
    tokens = [ t for t in tokens if t.isalnum() and t not in stopwords.words('english') ]

    return tokens


def data_df(dataset_path, file_name, save_dir):
    with open(dataset_path, "r", encoding="utf-8") as xml_reader:
        data = xml_reader.read()

    data_xml = BeautifulSoup(data, "xml")
    
    data_rev = data_xml.find('Reviews')
    data_reviews = data_rev.find_all('Review')

    rev_sentences = []
    rev_dict = {}
    sent_opinions = []
    sent_dict = {}

    all_ops = []

    r_ids = re.findall(r'rid="(.*?)"', str(data_reviews))
    for rev in data_reviews:
        r_id = re.findall(r'rid="(.*?)"', str(rev))[0]
        rev_dict[r_id]={}
        sent_op_list = []
        temp_s_dict={}
        sent_list = rev.find_all('sentence')
        if str( sent_list )!="[]":
            sentences = sent_list
            rev_sentences.append(sentences)
            for s in sentences:
                s_id = re.findall(r'id="(.*?)"', str(s))[0]
                opinions =  s.find_all('Opinion')
                for o in opinions:
                    if str(o)!="[]":       
                        sent_op_list.append(o)
                        tokens = tok_text(s.find_all('text')[0].get_text())
                        if len(re.findall(r'target="(.*?)"', str(o)))>0:
                            all_ops.append([ re.findall(r'polarity="(.*?)"', str(o))[0], r_id, s_id,  s.find_all('text')[0].get_text(), re.findall(r'category="(.*?)"', str(o))[0], re.findall(r'target="(.*?)"', str(o))[0], (re.findall(r'from="(.*?)"', str(o))[0], re.findall(r'to="(.*?)"', str(o))[0]), tokens  ] )
            
                sent_dict[s_id]=sent_op_list
                temp_s_dict[s_id]=sent_op_list

            sent_opinions.append(sent_op_list)
            rev_dict[r_id]=temp_s_dict
        
        


    # for o_list, s_list in zip(sent_opinions,rev_sentences):
    #     for o in o_list:
    #         if str(o) !="[]" and not all(x in str(o) for x in [ "category=", "target=", "polarity=", "from=", "to=" ]):
    #             print(re.findall(r'id="(.*?)"', str(s_list)))



    col_names = [ "polarity", "review_id", "sentence_id", "text", "op_category", "op_target", "op_OTE_offset", "tokens" ]
    data_df = pd.DataFrame(all_ops, columns=col_names)


    data_df_filepath = os.path.join(save_dir,f"opinions_polarity_{file_name}.csv")
    data_df.to_csv(data_df_filepath, index=False, header=True)

    return data_df

def add_feat_POS(df):
    tok_list = list(df["tokens"].values)
    pos_text_list = []
    for toks in tok_list:
        pos = pos_tag(toks)
        pos_text_list.append(pos)
    
    df["POS"] = pos_text_list

    return df

def add_feat_freq(df):
    tok_list = list(df["tokens"].values)
    f_text_list = []
    for tok_text in tok_list:
        f = FreqDist(tok_text)
        freq_dict = {k:v for k,v in f.items()}
        f_text_list.append(freq_dict)
    
    df["freq"] = f_text_list

    return df

def add_feat_sentiment(df):
    t_list = list(df["text"].values)
    s_text_list = []
    for t in t_list:
        sentAn = SentimentIntensityAnalyzer()
        # sentim_dict = {k:v for k,v in sentim.items()}
        sentim_dict = sentAn.polarity_scores(t)
        s_text_list.append(sentim_dict)
    
    df["sentiment"] = s_text_list

    return df

def extract_ngrams(tokens, num):
    n_grams = ngrams(tokens, num)
    return n_grams

def add_feat_ngrams(df,n):
    t_list = list(df["tokens"].values)
    ng_text_list = []
    for t in t_list:
        ngrams = extract_ngrams(t,n)
        ngram_list = [ gr for gr in ngrams]
        ng_text_list.append(ngram_list)
    
    df[f"ngrams_{n}"] = ng_text_list

    return df

def stemming(df):

    ps = PorterStemmer()
    
    stemmed_text=[]
    for toks in df["tokens"].values:
        stemmed_words=[]
        for w in toks:
            stemmed_words.append(ps.stem(w))

        stemmed_text.append(stemmed_words)

    df["stemmed_text"]=stemmed_text


def alpha_to_numerical(df,feat_name):
    le = LabelEncoder()
    df[feat_name] = le.fit_transform(df[feat_name])
    return df, le



if __name__=="__main__":
    save_dir = "..\Data\DataFrames"
    all_data_df = pd.DataFrame()
    for i in range(10):
        
        df = data_df(f"..\Data\Parts\part{i+1}.xml", f"part{i+1}", save_dir)
        all_data_df = pd.concat( [ all_data_df , df])


    # add features
    add_feat_POS(all_data_df)
    add_feat_freq(all_data_df)
    add_feat_sentiment(all_data_df)
    add_feat_ngrams(all_data_df,2)
    add_feat_ngrams(all_data_df,3)
    add_feat_ngrams(all_data_df,4)
    alpha_to_numerical(all_data_df,"op_target")
 
    labels = all_data_df["polarity"].values
    all_data_df.pop("polarity")

    file_name = f"allFeats"
    data_df_filepath = os.path.join(save_dir,f"opinions_polarity_{file_name}.csv")
    all_data_df.to_csv(data_df_filepath, index=False, header=True)