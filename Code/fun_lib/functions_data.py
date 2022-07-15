from cProfile import label
import os
import sys
from unicodedata import category
import pandas as pd
import re
import numpy as np

from bs4 import BeautifulSoup

from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer, sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk import pos_tag, FreqDist, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.util import ngrams

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sentence_transformers import SentenceTransformer

import gensim
from gensim.models import Word2Vec, KeyedVectors
from torch import embedding

def tok_text(text):

    tok = TweetTokenizer()
    tokens = tok.tokenize(text)
    tokens = [ t for t in tokens if t.isalnum() and t not in stopwords.words('english') ]

    return tokens

def embeddings_df(df, embeddings):

    embs = [ eval(enc) if type(enc)!=list else enc for enc in df[f"sentence_{embeddings}_emb"]  ]
    text_embed_names = [ str(i) for i in range(len(embs[0])) ]
    embs_df = pd.DataFrame(embs, columns=text_embed_names)

    return embs_df


def embs_feats_df(df, embs_df):

    embFeats_df = embs_df.join(df)

    return embFeats_df


def token_WVs(all_tokens, w2v_model):

    count_unknown=0
    
    unk_vector = np.zeros(w2v_model["the"].shape)
    nump_zero_arrs=[  unk_vector for t in all_tokens ]
    token_dict={k:v for k,v in zip(all_tokens,nump_zero_arrs)}
    for tok in all_tokens:
        if tok in w2v_model:
            token_dict[tok]=w2v_model[tok]
        else:
            count_unknown+=1
    
    token_dict["<UNK>"] = np.zeros(w2v_model["the"].shape)
    
    return token_dict, count_unknown


def text_WVs(text_list, token_vectors):
#     for text in text_list:
#         sent_vect_mean = 
    sentence_vects =[ np.array( [ token_vectors[token] if token in token_vectors else token_vectors["<UNK>"]  for token in s_tokens ] ).mean(axis=0) if len(s_tokens)>0 else token_vectors["<UNK>"] for s_tokens in text_list    ]      
    return sentence_vects


def get_tokens(sentences):
    text_token_list = []
    all_toks = []
    for text in sentences:
        tok = TweetTokenizer()
        tokens = tok.tokenize(text)
        tokens = [ t for t in tokens if t.isalnum() and t not in stopwords.words('english') ]
        text_token_list.append(tokens)
        all_toks.extend(tokens)
    
    return all_toks, text_token_list


def Word2Vec_embs(sentences):
    Word2Vec_news_emb_file = os.path.join(f"..{os.sep}Data{os.sep}GoogleNews-vectors-negative300.bin")
    google_news_vectors = KeyedVectors.load_word2vec_format(Word2Vec_news_emb_file, binary=True)
    tok_list, tok_text_list = get_tokens(sentences)
    token_vectors_dict_train, unk_words = token_WVs(tok_list, google_news_vectors)
    print(f"\nUnknown Words: {unk_words} ({unk_words/(len(token_vectors_dict_train)-1)*100:.2f}%)")
    sentence_vectors = text_WVs(tok_text_list, token_vectors_dict_train)
    sentence_vectors = [ vect.tolist() for vect in sentence_vectors ]

    return sentence_vectors


def sBERT_embs(sentences):
    # !pip install sentence-transformers
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentence_embeddings = sbert_model.encode(sentences)
    sentence_embeddings = [ vect.tolist() for vect in sentence_embeddings ]
    return sentence_embeddings


import io
def fastText_embs(sentences, fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

import csv
def sentence_enbeddings(dataset_path=f"..{os.sep}Data{os.sep}ABSA16_Restaurants_Train_SB1_v2.xml", embeddings="Word2Vec"):
    
    
    with open(dataset_path, "r", encoding="utf-8") as xml_reader:
        data = xml_reader.read()

    all_sents = []
    all_sent_ids = []
    data_xml = BeautifulSoup(data, "xml")
    data_rev = data_xml.find('Reviews')
    data_reviews = data_rev.find_all('Review')
    
    for rev in data_reviews:
        sent_list = rev.find_all('sentence')
        if str( sent_list )!="[]":
           for s in sent_list:
                opinions =  s.find_all('Opinion')
                for o in opinions:
                    if str(o)!="[]":
                        all_sents.append(s.find_all('text')[0].get_text())
                        all_sent_ids.append(re.findall(r'id="(.*?)"', str(s))[0])
    
    # Word2Vec
    sentence_embeds = []
    if embeddings=="Word2Vec":
        sentence_embeds = Word2Vec_embs(all_sents)

    # Sentence BERT 
    elif embeddings=="sBERT":
        sentence_embeds = sBERT_embs(all_sents)
    
    elif embeddings=="FastText":
        pass

    emb_dict = { s_id:s_emb for s_id, s_emb in zip(all_sent_ids, sentence_embeds)}
    
    # csv_columns = embeddings
    csv_file = f"EmbeddingDicts{os.sep}{embeddings}_sEmbeddings_dict.csv"
    with open(csv_file, 'w', newline='') as csv_file_w:  

        writer = csv.writer(csv_file_w, delimiter=',' )
        for key, value in emb_dict.items():
            writer.writerow([key, value])

    return emb_dict


def get_sEmbeddings_dict(dict_path):
    
    with open(dict_path, mode='r') as infile:
        reader = csv.reader(infile, delimiter=',')
        mydict = {rows[0]:rows[1] for rows in reader}
    
    return mydict

def data_df(dataset_path, file_name, save_dir, embeddings="Word2Vec"):

    wv_dict = get_sEmbeddings_dict(f"EmbeddingDicts{os.sep}{embeddings}_sEmbeddings_dict.csv")

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
                            polarity = re.findall(r'polarity="(.*?)"', str(o))[0]
                            s_text = s.find_all('text')[0].get_text()
                            op_category =  re.findall(r'category="(.*?)"', str(o))[0]
                            op_entity = op_category.split("#")[0]
                            op_attribute = op_category.split("#")[1]
                            op_target = re.findall(r'target="(.*?)"', str(o))[0]
                            op_OTE_offset = ( int(re.findall(r'from="(.*?)"', str(o))[0]), int(re.findall(r'to="(.*?)"', str(o))[0]))

                            all_ops.append([ polarity, r_id, s_id, s_text, wv_dict[s_id], op_category, op_entity, op_attribute, op_target, op_OTE_offset, tokens  ] )
            
                sent_dict[s_id]=sent_op_list
                temp_s_dict[s_id]=sent_op_list

            sent_opinions.append(sent_op_list)
            rev_dict[r_id]=temp_s_dict
        
        


    # for o_list, s_list in zip(sent_opinions,rev_sentences):
    #     for o in o_list:
    #         if str(o) !="[]" and not all(x in str(o) for x in [ "category=", "target=", "polarity=", "from=", "to=" ]):
    #             print(re.findall(r'id="(.*?)"', str(s_list)))



    col_names = [ "polarity", "review_id", "sentence_id", "text", f"sentence_{embeddings}_emb", "op_category", "op_entity", "op_attribute", "op_target", "op_OTE_offset", "tokens" ]
    data_df = pd.DataFrame(all_ops, columns=col_names)

    data_df=embs_feats_df(data_df,embeddings_df(data_df, embeddings=embeddings))

    data_df_filepath = os.path.join(save_dir,f"opinions_polarity_{file_name}.csv")
    data_df.to_csv(data_df_filepath, index=False, header=True)

    return data_df

def add_feat_tokens(df):
    tok_list = [ tok_text(t) for t in df["text"] ]
    df["tokens"] = tok_list
    return df

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

def add_feat_stemming(df):

    ps = PorterStemmer()
    
    stemmed_text=[]
    for toks in df["tokens"].values:
        stemmed_words=[]
        for w in toks:
            stemmed_words.append(ps.stem(w))

        stemmed_text.append(stemmed_words)

    df["stemmed_text"]=stemmed_text
    return df

    
def add_feat_lemmas(df):
    tok_text_list = df["tokens"].values
    
    lem = WordNetLemmatizer()
    lem_text_list=[]
    for toks in tok_text_list:
        l_words=[]
        for w in toks:
            l_words.append(lem.lemmatize(w))

        lem_text_list.append(l_words) 

    df["lem_text"]=lem_text_list

    return df

        
def add_feat_CountVect(df):
    #tokenizer to remove unwanted elements from out data like symbols and numbers
    cv = CountVectorizer(lowercase=True,stop_words='english', ngram_range = (1,3), tokenizer = TweetTokenizer().tokenize)
    text_cv = cv.fit_transform( df["text"])
    df["countVect"] = [ row.toarray() for row in text_cv]

    return df

def add_feat_TfidfVect(df):
    tf = TfidfVectorizer(lowercase=True,stop_words='english', ngram_range = (1,3), tokenizer = TweetTokenizer().tokenize)
    text_tf = tf.fit_transform(df['text'])
    df["tfidfVect"]= [ row.toarray() for row in text_tf]

    return df


def alpha_to_numerical(df,feat_name):
    le = LabelEncoder()
    label_list = le.fit_transform(df[feat_name]).tolist()
    # enc_labels = pd.Series(label_list)
    df.pop(feat_name)
    # df.drop(feat_name, axis = 1, inplace = True)
    # df[feat_name] = enc_labels
    df.insert( loc=len(df.columns), column=feat_name, value=label_list)
    return df, le



if __name__=="__main__":
    save_dir = "..\Data\DataFrames"
    all_data_df = pd.DataFrame()

    embeddings="Word2Vec"
    # embeddings="sBERT"
    
    for i in range(10):
        
        df = data_df(f"..\Data\Parts\part{i+1}.xml", f"part{i+1}", save_dir, embeddings=embeddings)
        all_data_df = pd.concat( [ all_data_df , df])

    # sentence_enbeddings(embeddings=embeddings)

    # embeddings_df(all_data_df,embeddings)
    # add features
    # add_feat_POS(all_data_df)
    # add_feat_freq(all_data_df)
    # add_feat_sentiment(all_data_df)
    # add_feat_ngrams(all_data_df,2)
    # add_feat_ngrams(all_data_df,3)
    # add_feat_ngrams(all_data_df,4)
    # add_feat_stemming(all_data_df)
    # add_feat_lemmas(all_data_df)
    # add_feat_CountVect(all_data_df)
    # add_feat_TfidfVect(all_data_df)

    alpha_to_numerical(all_data_df,"polarity")

    
    # print(all_data_df["countVect"])
    # print(all_data_df["tfidfVect"])
 
    # labels = all_data_df["polarity"].values
    # all_data_df.pop("polarity")

    # file_name = f"allFeats"
    # data_df_filepath = os.path.join(save_dir,f"opinions_polarity_{file_name}.csv")
    # all_data_df.to_csv(data_df_filepath, index=False, header=True)