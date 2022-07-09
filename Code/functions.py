from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import os
from requests import head
from sympy import arg
import sys
import pandas as pd
import re

def tok_text(text):

    tok = TweetTokenizer()
    tokens = tok.tokenize(text)
    tokens = [ t for t in tokens if t.isalnum() and t not in stopwords.words('english') ]

    return tokens


def data_df(dataset_path, file_name, save_dir):
    with open(dataset_path, "r", encoding="utf-8") as xml_reader:
        data = xml_reader.read()

    data_xml = BeautifulSoup(data, "xml")

    data_reviews = data_xml.find_all('Review')
    # data_text = data_xml.find_all('Review')
    # data_sentences = data_xml.find_all('Review')
    # data_review = data_xml.find_all('Review')
    rev_sentences = []
    rev_dict = {}
    sent_opinions = []
    sent_dict = {}

    all_ops = []


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
        
        


    for o_list, s_list in zip(sent_opinions,rev_sentences):
        for o in o_list:
            if str(o) !="[]" and not all(x in str(o) for x in [ "category=", "target=", "polarity=", "from=", "to=" ]):
                print(re.findall(r'id="(.*?)"', str(s_list)))



    col_names = [ "polarity", "review_id", "sentence_id", "text", "op_category", "op_target", "op_OTE_offset", "tokens" ]
    data_df = pd.DataFrame(all_ops, columns=col_names)


    data_df_filepath = os.path.join(save_dir,f"opinions_polarity_{file_name}.csv")
    data_df.to_csv(data_df_filepath, index=False, header=True)

    return data_df


if __name__=="__main__":
    for i in range(10):

        data_df(f"..\Data\Parts\part{i+1}.xml", f"part{i+1}", "..\Data\DataFrames")