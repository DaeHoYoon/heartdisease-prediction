#%%
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import learning_curve


def trainmodel(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    print(f'{model.__class__.__name__}의 train roc_auc_score :{roc_auc_score(y_train, pred_train)}')
    print(classification_report(y_train, pred_train))
    print(f'{model.__class__.__name__}의 test roc_auc_score :{roc_auc_score(y_test, pred_test)}')
    print(classification_report(y_test, pred_test))
    
    return pred_train, pred_test
    
def Roccurve(model, X_train, y_train, X_test, y_test):
    pred_train, pred_test = trainmodel(model, X_train, y_train, X_test, y_test)
    tpr, fpr, thr = roc_curve(y_test, pred_test)
    fig = plt.figure(figsize=(10,10))
    plt.plot(tpr,fpr, label = 'roc_auc_score(y_test, pred_test)')
    plt.title(f'{model.__class__.__name__} ROC_AUC_CURVE')
    plt.show()

def Learn_curve(model, X_train, y_train):
    train_size, train_score, test_score = learning_curve(model, X_train, y_train, train_sizes= np.linspace(.1,1.0,5), cv=3)
    train_score_mean = np.mean(train_score, axis=1)
    test_score_mean = np.mean(test_score, axis=1)
    fig = plt.figure(figsize=(10,10))
    plt.plot(train_size, train_score_mean, label = 'train')
    plt.plot(train_size, test_score_mean, label = 'test')
    plt.title(f'{model.__class__.__name__} learning rate')
    plt.legend()
    plt.show()