#  10-fold cross validation

# evaluate a logistic regression model using k-fold cross-validation
import sys
import os
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score

import train as tr
import test as tst

# Random Forest
default_class_algo = "RF"
embeddings = "sBERT"

# create model
if len(sys.argv)<2:
    print(f"\nERROR: did not specify the classification algorithm !\n(default algorithm: {default_class_algo})\n")
    algo_name = default_class_algo
else:
    algo_name=sys.argv[1]


# 10 fold with different parts each time
k_folds=10
# lists of scores for future graphs
precision_list = []
recall_list = []
f1_list = []
acc_list = []

logfile_path = f"..{os.sep}Logfiles{os.sep}experiments_{algo_name}_{k_folds}Fold_{embeddings}.txt"

with open(logfile_path, "w", encoding="utf-8") as logger:
    logger.write(f"Model: {algo_name}\n")
    for i in range(k_folds):
        part_no=i+1
        parts_test = [part_no]
        parts_train = [ i+1 for i in range(k_folds) ]
        parts_train.remove(part_no)
        logger.write(f"\nXML Parts\nTraining: {parts_train}\nTest: {part_no}\n\n")
        
        X_train, X_test, y_train, y_test = tr.data_proc_train_test_parts(parts_train, parts_test, embeddings)
        model = tr.train(X_train, y_train, algo_name )

        # Scores
        # model_scores_precision = tr.evaluate_model_precision(model, X_test, y_test)
        # tot_p.append(model_scores_precision)
        # model_scores_recall = tr.evaluate_model_recall(model, X_test, y_test)
        # tot_r.append(model_scores_recall)
        # model_scores_f1 = tr.evaluate_model_f1(model, X_test, y_test)
        # tot_r.append(model_scores_recall)

        # print(f"\nFold {part_no}\n\n> Score\nPrecision: {model_scores_precision:.3f} | Recall: {model_scores_recall:.3f} | F1: {model_scores_f1:.3f}")

        predictions, c_rep, c_rep_dict = tst.test(X_test, y_test, model)
        acc_score = accuracy_score(y_test,predictions)
        p_score = c_rep_dict["macro avg"]["precision"]
        r_score = c_rep_dict["macro avg"]["recall"]
        f1 = c_rep_dict["macro avg"]["f1-score"]

        acc_list.append(acc_score)
        precision_list.append(p_score)
        recall_list.append(r_score)
        f1_list.append(f1)

        print(f"\nPart {part_no}", c_rep)
    
        logger.write( c_rep )
        logger.write("\n\n")
        tst.plot_roc(model,X_test,y_test,algo_name,part_no,mode="Experiments",embeddings=embeddings)

    score_str = f"\n\n> Average Scores\nAccuracy: {sum(acc_list)/k_folds:.3f} | Precision: {sum(precision_list)/k_folds:.3f} | Recall: {sum(recall_list)/k_folds:.3f} | F1: {sum(f1_list)/k_folds:.3f}\n\n"
    print(score_str)

    logger.write(score_str)



# __________________________________________________________________________________________________________________________________________
# -TODO-
# more scores/ score table/ class scores etc
# plot scores over the folds?! 
# ROC Curve ?!