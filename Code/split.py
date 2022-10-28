import os
from regex import R
from requests import head
import sys
import re
import pandas as pd
from bs4 import BeautifulSoup

from fun_lib import functions_data as funs_d

# initial .xml dataset file
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

# read and split .xml file in 10 parts
data_xml = BeautifulSoup(data, "xml")
data_reviews = data_xml.find_all('Review')

for i in range(10):

    dataset_part = ".."+os.sep+"Data"+os.sep+"Parts"+os.sep+f"part{i+1}.xml"
    with open(dataset_part, "w", encoding="utf-8") as xml_writer:
        r_str = ""
        r_list = ['<?xml version="1.0" encoding="utf-8"?>', '<Reviews>']
        for r in data_reviews[ i*35:i*35+35 ]:
            r_list.append(str(r))

        r_list.append("</Reviews>")

        r_final = "\n".join(r_list)
        r_final = BeautifulSoup(r_final, "xml")

        xml_writer.write(r_final.prettify())

