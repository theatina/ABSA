import os
import sys
import random
import pandas as pd
import pickle

from sklearn.metrics import classification_report

from fun_lib import functions_data as funs_d
from fun_lib import functions_training as funs_t


def load_df_parts(embeddings, part_list=[i for i in range(10)]):
    save_dir = "..\Data\DataFrames"
    all_data_df = pd.DataFrame()
    for i in part_list:
        
        df = funs_d.data_df(f"..\Data\Parts\part{i}.xml", f"part{i}", save_dir, embeddings)
        all_data_df = pd.concat( [ all_data_df , df])

    return all_data_df



def data_proc(part_to_use, embeddings):
    
    df = load_df_parts(embeddings, part_to_use)
    
    labels_df, le = funs_d.alpha_to_numerical(df,"polarity")
    labels_df = labels_df["polarity"].values

    df.pop("polarity")
    df = funs_t.choose_feats(df)
    
    return df, labels_df



def logging(py_file, model_name, part, predictions, c_rep):
    logfile_path = f"..{os.sep}Logfiles{os.sep}{py_file}_{model_name}_test_{part}.txt"
    with open(logfile_path, "w", encoding="utf-8") as logger:
        logger.write(f"Model: {model_name}\nXML Part: {part}\n\n")
        c_rep = classification_report(y_test, predictions, target_names=["Negative", "Neutral", "Positive"])
        logger.write( c_rep )
    
    return logfile_path

def test(X,y,model):
    predictions = model.predict(X)
    c_rep = classification_report(y, predictions, target_names=["Negative", "Neutral", "Positive"], zero_division=1, output_dict=False)
    c_rep_dict = classification_report(y, predictions, target_names=["Negative", "Neutral", "Positive"], zero_division=1, output_dict=True)
    return predictions, c_rep, c_rep_dict


def test_steps(X_test, y_test, model_name, part):
    model_path = os.path.join(f"..{os.sep}Models",model_name+".model")
    model = pickle.load(open(model_path, 'rb'))

    predictions, c_rep, c_rep_dict = test(X_test, y_test, model)
    
    print(c_rep)

    logging("test", model_name, part, predictions, c_rep)

    return model


if __name__=="__main__":

    part_to_use = [random.randint(1, 10)]
    model_name = "RF_parts_1_2_3_4_5_6_7_8_9_10_Word2Vec.model"
    embeddings = "sBERT"

    # create model
    if len(sys.argv)<2:
        print(f"\nERROR: did not specify number of parts and model name !\ndefault part: random between 1-10\ndefault model: RF_parts_1_2_3_4_5_6_7_8_9_10_Word2Vec.model\n\nUsage: python test.py < part to use (int 1-10) >  <model name>\n")
        
    elif len(sys.argv)==2:
        model_name = sys.argv[1]
    else:
        part_to_use = [int(sys.argv[2])]
        model_name = sys.argv[1]


    X_test, y_test = data_proc(part_to_use, embeddings)
    test_steps(X_test, y_test, model_name, part_to_use[0])
