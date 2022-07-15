import os
import sys
import random
import pandas as pd
import pickle

from sklearn.metrics import classification_report

from fun_lib import functions_data as funs_d
from fun_lib import functions_training as funs_t


def load_df_parts(part_list=[i for i in range(10)]):
    save_dir = "..\Data\DataFrames"
    all_data_df = pd.DataFrame()
    for i in part_list:
        
        df = funs_d.data_df(f"..\Data\Parts\part{i}.xml", f"part{i}", save_dir)
        all_data_df = pd.concat( [ all_data_df , df])

    return all_data_df



def data_proc(part_to_use):
    
    df = load_df_parts(part_to_use)
    
    labels_df, le = funs_d.alpha_to_numerical(df,"polarity")
    labels_df = labels_df["polarity"].values

    df.pop("polarity")
    df = funs_t.choose_feats(df)
    
    return df, labels_df

def test(X_test, y_test, model_path):
    model = pickle.load(open(model_path, 'rb'))
    
    # evaluate_model_recall(model, X_train, y_train)
    # evaluate_model_precision(model, X_train, y_train)
    # evaluate_model_f1(model, X_train, y_train)
    
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions, target_names=["Negative", "Neutral", "Positive"]))

    return model


if __name__=="__main__":

    default_model_path = f"..{os.sep}Models{os.sep}RF_parts_1_2_3_4_5_6_7_8_9_10.model"
    # create model
    if len(sys.argv)<3:
        print(f"\nERROR: did not specify number of parts or/and model path !\ndefault part: random between 1-10\ndefault model: RF_parts_1_2_3_4_5_6_7_8_9_10.model\n\nUsage: python test.py < part to use (int 1-10) >  <model path>\n")
        part_to_use = [random.randint(1, 10)]
        model_path = default_model_path
    else:
        part_to_use = [int(sys.argv[1])]
        model_path = sys.argv[2]


    X_test, y_test = data_proc(part_to_use)
    test(X_test, y_test, model_path)