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

# Random Forest
default_class_algo = "RF"
# create model
if len(sys.argv)<2:
    print(f"\nERROR: did not specify the classification algorithm !\n(default algorithm: {default_class_algo})\n")
    algo_name = default_class_algo
else:
    algo_name=sys.argv[2]

print(f"\n\n{algo_name}\n\n")

# 10 fold with different parts each time
k_folds=10
# lists of scores for future graphs
tot_p = []
tot_r = []
tot_f1 = []
for i in range(k_folds):
    part_no=i+1
    parts_test = [part_no]
    parts_train = [ i+1 for i in range(k_folds) ]
    parts_train.remove(part_no)
    
    X_train, X_test, y_train, y_test = tr.data_proc_train_test_parts(parts_train, parts_test)
    model = tr.train(X_train, y_train, algo_name )

    # Scores
    # model_scores_precision = tr.evaluate_model_precision(model, X_test, y_test)
    # tot_p.append(model_scores_precision)
    # model_scores_recall = tr.evaluate_model_recall(model, X_test, y_test)
    # tot_r.append(model_scores_recall)
    # model_scores_f1 = tr.evaluate_model_f1(model, X_test, y_test)
    # tot_r.append(model_scores_recall)

    # print(f"\nFold {part_no}\n\n> Score\nPrecision: {model_scores_precision:.3f} | Recall: {model_scores_recall:.3f} | F1: {model_scores_f1:.3f}")

    predictions = model.predict(X_test)
    print(f"\nPart {part_no}",classification_report(y_test,predictions, target_names=["Negative", "Neutral", "Positive"]))

# print(f"\n\n> Total Score\nPrecision: {sum(tot_p)/len(k_folds):.3f} | Recall: {sum(tot_r)/len(k_folds):.3f} | F1: {sum(tot_f1)/len(k_folds):.3f}")


# __________________________________________________________________________________________________________________________________________
# -TODO-
# more scores/ score table/ class scores etc
# plot scores over the folds?! 
# ROC Curve ?!