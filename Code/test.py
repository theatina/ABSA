import os
import sys
import random
import pandas as pd
import pickle
from matplotlib import pyplot as plt

from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
    print("\nTest Scores:\n\n"+c_rep)
    return predictions, c_rep, c_rep_dict


def test_steps(X_test, y_test, model_name, part):
    model_path = os.path.join(f"..{os.sep}Models",model_name+".model")
    model = pickle.load(open(model_path, 'rb'))

    predictions, c_rep, c_rep_dict = test(X_test, y_test, model)
    
    print(c_rep)

    logging("test", model_name, part, predictions, c_rep)

    return model

def plot_roc(model,X,y,model_name,part=1,mode="Test", embeddings="sBERT"):
    plt.clf()
    # generate 2 class dataset
    # X, y = make_classification(n_samples=1000, n_classes=3, n_features=20, n_informative=3, random_state=42)

    # split into train/test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    pred = model.predict(X)
    pred_prob = model.predict_proba(X)

    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh = {}
    roc_auc = {}

    n_class = 3

    for i in range(n_class):    
        fpr[i], tpr[i], thresh[i] = roc_curve(y, pred_prob[:,i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # plotting    
    plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label=f'Class 0 - Negative (auc: {roc_auc[0]:.2f})')
    plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label=f'Class 1 - Neutral (auc: {roc_auc[1]:.2f})')
    plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label=f'Class 2 - Positive (auc: {roc_auc[2]:.2f})')
    plt.title(f'Multiclass ROC curve\n{model_name} - {embeddings}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig(f'..{os.sep}Results{os.sep}Figures{os.sep}{mode}{os.sep}{model_name}_{embeddings}_testPart{part}_ROC',dpi=300)



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

    train_df_path = f"..{os.sep}Data{os.sep}DataFrames{os.sep}opinions_polarity_allFeats.csv"
    train_df = pd.read_csv(train_df_path)

    # total POS tag features : dim=13
    pos_list = [ c for c in train_df.columns[-13:] ]
    X_test, y_test = data_proc(part_to_use, embeddings, pos_list = pos_list)
    model = test_steps(X_test, y_test, model_name, part_to_use[0])

    plot_roc(model,X_test,y_test,model_name,part_to_use[0],"Test")



