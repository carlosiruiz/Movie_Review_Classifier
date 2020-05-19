import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures
import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion Matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, weight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if i == 0:
            plt.text(j-0.1, i+0.3, format(cm[i, j], fmt), color="white" if cm[i, j] > thresh else "black")
        if i == 1:
            plt.text(j-0.1, i-0.2, format(cm[i, j], fmt), color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label', weight='bold')
    plt.xlabel('Predicted Label', weight='bold')
    plt.show()

def classification_models(X,y,models,test_size = .2,random_state = 13,params=None,grid=False,param_grid=None,cv=5):
    '''Creates models from given input and returns a table of evaluation metrics for each model.'''
    #create scaler
    # scaler = MinMaxScaler()
    # scaler.fit(X)
    #dictionary of models
    model_dict = {'logistic': LogisticRegression,'knn': KNeighborsClassifier,
                 'tree': DecisionTreeClassifier,'rf': RandomForestClassifier,
                 'xgb': xgb.XGBClassifier, 'AdaBoost': AdaBoostClassifier,
                 'GrdBoost': GradientBoostingClassifier, 'svc': SVC,
                 'Bayes': GaussianNB}
    #scale data
    # X_transformed = scaler.transform(X)
    #split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = test_size, random_state=random_state)
    #create evaluation metrics table
    summary_df = pd.DataFrame(columns=['Model','Accuracy', 'Recall', 'Precision',  'F1'])
    #variable to keep track of which parameter dictionary we are on 
    i = 0
    #if no params given and gridsearch is false, create base models
    if (params==None) & (grid == False):
        for model in models:
            for key, value in model_dict.items(): 
                if model == key: 
                    model_use = value()
                    print(f"Using {key}") #debug
                    model_use.fit(X_train,y_train)
                    predictions = model_use.predict(X_test)
                    summary_df = summary_df.append({'Model': model,
                       'Accuracy': metrics.accuracy_score(y_test, predictions),
                       'Recall': metrics.recall_score(y_test, predictions),
                       'Precision': metrics.precision_score(y_test, predictions), 
                       'F1': metrics.f1_score(y_test, predictions)},ignore_index=True)
    #if gridsearch is true, run a gridsearch and use best parameters to create models
    elif grid == True:
        for model in models:
            for key, value in model_dict.items(): 
                if model == key: 
                    model_use = GridSearchCV(estimator = value(),param_grid = param_grid[i],cv = cv)
                    model_use.fit(X_train,y_train)
                    model_grid = value(**model_use.best_params_)
                    model_grid.fit(X_train,y_train)
                    y_pred = model_grid.predict(X_test)
                    i = i+1
                    summary_df = summary_df.append({'Model': model,
                   'Accuracy': metrics.accuracy_score(y_test, y_pred),
                   'Recall': metrics.recall_score(y_test, y_pred),
                   'Precision': metrics.precision_score(y_test, y_pred), 
                   'F1': metrics.f1_score(y_test, y_pred)},ignore_index=True)
                    print(f"{model}'s best parameters are {model_use.best_params_}")
    #if previous conditions are false, then we have been given parameters and gridsearch is false
    #we create models with given parameters
    else:
        for model in models:
            for key, value in model_dict.items(): 
                if model == key: 
                    model_use = value(**params[i])
                    model_use.fit(X_train,y_train)
                    y_pred = model_use.predict(X_test)
                    i = i+1
                    summary_df = summary_df.append({'Model': model,
                   'Accuracy': metrics.accuracy_score(y_test, y_pred),
                   'Recall': metrics.recall_score(y_test, y_pred),
                   'Precision': metrics.precision_score(y_test, y_pred), 
                   'F1': metrics.f1_score(y_test, y_pred)},ignore_index=True)
                    print("params")
    return summary_df