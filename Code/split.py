import os
from sympy import arg
import sys
import pandas as pd
from bs4 import BeautifulSoup
import re

import functions as funs


dataset_path = ".."+os.sep+"Data"+os.sep+"ABSA16_Restaurants_Train_SB1_v2.xml"
if len(sys.argv)>1:
    dataset_path = sys.argv[1]
else:
    print(f"\nERROR: did not specify dataset path !\n(default: {dataset_path})\n")

if not os.path.exists(dataset_path):
    print(f"\nERROR: file {dataset_path} does not exist !\n")
    exit()

with open(dataset_path, "r", encoding="utf-8") as xml_reader:
    data = xml_reader.read()


data_xml = BeautifulSoup(data, "xml")

data_reviews = data_xml.find_all('Review')
# data_text = data_xml.find_all('Review')
# data_sentences = data_xml.find_all('Review')
# data_review = data_xml.find_all('Review')
rev_sentences = []
for rev in data_reviews:
   	rev_sentences.append(rev.find_all('sentence'))
print(len(rev_sentences))

sentence_list = [ [ re.findall(r'rid="(.*?)"', str(r))[0], re.findall(r'id="(.*?)"', str(s))[0].split(":")[-1], s.find_all('text')[0].get_text()  ] for r,sent_l in zip(data_reviews,rev_sentences) for s in sent_l]


print(len(sentence_list))
print(sentence_list[:5])