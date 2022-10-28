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
import train as tr

# load the xml parts  
def load_df_parts(embeddings, part_list=[i for i in range(10)]):
    save_dir = "..\Data\DataFrames"
    all_data_df = pd.DataFrame()
    for i in part_list:
        
        df = funs_d.data_df(f"..\Data\Parts\part{i}.xml", f"part{i}", save_dir, embeddings)
        all_data_df = pd.concat( [ all_data_df , df])

    return all_data_df

# data processing
def data_proc(part_to_use, embeddings, pos_list=[]):
    
    df = load_df_parts(embeddings, part_to_use)
    
    labels_df, le = funs_d.alpha_to_numerical(df,"polarity")
    labels_df = labels_df["polarity"].values

    data_df, pos_list = tr.add_feats(data_df, pos_list)
    X_best = tr.choose_feats(data_df)
    
    # feature selection method and test
    # k_top_feats = int(X_best.shape[1]/2)
    # technique = "mutual_info_classif"
    # technique_short = "mic"

    # print(f"\nFeature Selection\nTechnique: {technique}\nFeat num: {k_top_feats}\n")
    # X_best, unused = tr.feat_selection(data_df,labels_df,data_df,k_best=k_top_feats,technique=technique_short)
    
    return X_best, labels_df

# log scores and information for the report
def logging(py_file, model_name, part, predictions, c_rep):
    logfile_path = f"..{os.sep}Logfiles{os.sep}{py_file}_{model_name}_test_{part}.txt"
    with open(logfile_path, "w", encoding="utf-8") as logger:
        logger.write(f"Model: {model_name}\nXML Part: {part}\n\n")
        c_rep = classification_report(y_test, predictions, target_names=["Negative", "Neutral", "Positive"])
        logger.write( c_rep )
    
    return logfile_path

# making predictions and generating the classification score report - main test function used by the other .py files
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
    # embeddings = "Word2Vec"

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

    # no POS features were used here, after deciding it is a non-important feature
    pos_list=[]
    X_test, y_test = data_proc(part_to_use, embeddings, pos_list = pos_list)
    model = test_steps(X_test, y_test, model_name, part_to_use[0])

    plot_roc(model,X_test,y_test,model_name,part_to_use[0],"Test")



