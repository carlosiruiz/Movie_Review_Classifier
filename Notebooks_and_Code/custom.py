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
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import PolynomialFeatures
import itertools

def top_tfidf_feats(row, features, top_n=25):
    """
    Get top n tfidf values in row and return them with their corresponding feature names.

    Parameters:
    row -- Index for particular row being inspected; Array_like
    features -- All words in TF-IDF's vocabulary; List of Strings
    top_n -- Number of words to be returned; Int, default = 25

    Returns:
    df -- Pandas DataFrame with words themselves (feature names) and their
    corresponding score.
    """
    # produce the indices that would order the row by tf-idf value
    # reverse them (into descending order), and select the first top_n
    topn_ids = np.argsort(row)[::-1][:top_n]

    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['Feature', 'TF-IDF']
    return df

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    """
    Return the top n features that on average are most important amongst 
    documents in rows indentified by indices in grp_ids.

    Parameters:
    Xtr -- Tf-idf-weighted document-term matrix; 
    sparse matrix of (n_samples, n_features)
    features -- All words in TF-IDF's vocabulary; List of Strings
    grp_ids -- Row indices of particular documents we want to inspect. ‘None’ 
    indicates that we’re interested in all documents; List of Integers, default = 'None'
    min_tfidf -- Threshold TF-IDF score; Float, default = 0.1
    top_n -- Number of words to be returned; Int, default = 25

    Returns:
    df -- Pandas DataFrame with words themselves (feature names) and their
    corresponding score.
    """
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    # filter out the words with relatively low scores
    # (smaller than the provided threshold)
    D[D < min_tfidf] = 0
    # calculate the mean of each column across the selected rows
    # results in a single row of tf-idf values
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25):
    """
    Return a list of dfs, where each df holds top_n features and their mean 
    tfidf value calculated across documents with the same class label.

    Parameters:
    Xtr -- Tf-idf-weighted document-term matrix; 
    sparse matrix of (n_samples, n_features)
    y -- Target of data; Lists, numpy arrays, scipy-sparse matrices or pandas 
    dataframes
    features -- All words in TF-IDF's vocabulary; List of Strings
    min_tfidf -- Threshold TF-IDF score; Float, default = 0.1
    top_n -- Number of words to be returned; Int, default = 25

    Returns:
    dfs -- DataFrames for each document class, each containing the top n features;
    List of DataFrames
    """
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def plot_tfidf_classfeats_h(dfs):
    """
    Plot the data frames returned by the function plot_tfidf_classfeats().

    Parameters:
    dfs -- DataFrames for each document class, each containing the top n features;
    List of DataFrames
    """
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("label = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df['TF-IDF'], align='center', color='#eb9b34')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df['Feature'])
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.savefig(f'../Images/top_words_per_class.png')

## Functions above taken from following blog post:
## https://buhrmann.github.io/tfidf-analysis.html

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', 
    cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Parameters:
    cm -- Confusion matrix to plot; Confusion Matrix
    classes -- Classes of the confusion matrix; List of Strings
    normalize -- Wheteher to apply normalization or not; Bool, default = False
    title -- Title of plot. Also used as file name; String, default = "Confusion Matrix"
    cmap -- Colormap to apply to confusion matrix; Colormap, default = plt.cm.Blues
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
        plt.text(j, i, format(cm[i, j], fmt),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label', weight='bold')
    plt.xlabel('Predicted Label', weight='bold')
    plt.savefig(f'../Images/{title.replace(" ","_")}.png')

def classification_models(X, y, models, test_size = 0.2, random_state = 13, 
    params=None, grid=False, param_grid=None, cv=5):
    """
    Creates models from given data and returns a table of evaluation metrics for 
    each model.

    Parameters:
    X -- Features of data; Lists, numpy arrays, scipy-sparse matrices or pandas 
    dataframes
    y -- Target of data; Lists, numpy arrays, scipy-sparse matrices or pandas 
    dataframes
    models -- Models which should be run on given data; List of strings
    Accepted strings are 'logistic', 'knn', 'tree', 'rf', 'xgb', 'AdaBoost', 
    'GrdBoost', 'svc', and 'Bayes'.
    test_size -- If float, should be between 0.0 and 1.0 and represent the 
    proportion of the dataset to include in the test split. If int, represents 
    the absolute number of test samples. If None, the value is set to the 
    complement of the train size. If train_size is also None, it will be set to 
    0.25; Float or Int, default = 0.2
    random_state -- Controls the shuffling applied to the data before applying 
    the split. Pass an int for reproducible output across multiple function 
    calls; int or RandomState instance, default = 13
    params -- Parameters to pass to use with given models; Dict, default = None
    grid -- Whether to use gridsearch or not; Bool, default = False
    param_grid -- Dictionary with parameters names (str) as keys and lists of 
    parameter settings to try as values, or a list of such dictionaries, in 
    which case the grids spanned by each dictionary in the list are explored. 
    This enables searching over any sequence of parameter settings; Dict or 
    List of Dictionaries, default = None
    cv -- Determines the cross-validation splitting strategy. Possible inputs 
    for cv are: None (to use the default 5-fold cross validation), integer (to 
    specify the number of folds in a (Stratified)KFold), CV splitter, an iterable 
    yielding (train, test) splits as arrays of indices; Int, Cross-validation 
    Generator or an Iterable, default = 5

    Returns:
    summary_df -- Table of evaluation metric values for each model ran.
    """
    #dictionary of models
    model_dict = {'logistic': LogisticRegression,'knn': KNeighborsClassifier,
                 'tree': DecisionTreeClassifier,'rf': RandomForestClassifier,
                 'xgb': xgb.XGBClassifier, 'AdaBoost': AdaBoostClassifier,
                 'GrdBoost': GradientBoostingClassifier, 'svc': SVC,
                 'Bayes': MultinomialNB}
    #split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, 
        random_state=random_state)
    #create evaluation metrics table
    summary_df = pd.DataFrame(columns=['Model','Accuracy', 'Recall', 'Precision', 
        'F1'])
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
                       'F1': metrics.f1_score(y_test, predictions)}, 
                       ignore_index=True)
    #if gridsearch is true, run a gridsearch and use best parameters to create models
    elif grid == True:
        for model in models:
            for key, value in model_dict.items(): 
                if model == key: 
                    model_use = GridSearchCV(estimator = value(), 
                        param_grid = param_grid[i], cv = cv, verbose=1)
                    model_use.fit(X_train,y_train)
                    model_grid = value(**model_use.best_params_)
                    model_grid.fit(X_train,y_train)
                    y_pred = model_grid.predict(X_test)
                    i = i+1
                    summary_df = summary_df.append({'Model': model,
                   'Accuracy': metrics.accuracy_score(y_test, y_pred),
                   'Recall': metrics.recall_score(y_test, y_pred),
                   'Precision': metrics.precision_score(y_test, y_pred), 
                   'F1': metrics.f1_score(y_test, y_pred)}, ignore_index=True)
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
    return summary_df

def generate_confusion_indices(y_true, y_pred):
    """
    Returns lists of indexes of data classified as TP, TN, FP, or FN.

    Parameters:
    y_true -- Ground truth (correct) target values; array-like
    y_pred -- Estimated targets as returned by a classifier; array-like

    Returns:
    true_positives -- List of indexes of data classified correctly as class 1.
    true_negatives -- List of indexes of data classified correctly as class 0.
    false_positives -- List of indexes of data classified incorrectly as class 1.
    false_negatives -- List of indexes of data classified incorrectly as class 0.
    """
    true_vs_pred = list(zip(y_true, y_pred))
    
    #initialize counter
    i = 0                # (truth, pred)
    true_positives = []  # (1, 1)
    true_negatives = []  # (0 ,0)
    false_positives = [] # (0, 1)
    false_negatives = [] # (1, 0)
    
    for pair in true_vs_pred:
        if pair[0] == 1 and pair[1] == 1:
            true_positives.append(i)
        if pair[0] == 0 and pair[1] == 0:
            true_negatives.append(i)
        if pair[0] == 0 and pair[1] == 1:
            false_positives.append(i)
        if pair[0] == 1 and pair[1] == 0:
            false_negatives.append(i)
        i += 1
    
    print('True positives:', len(true_positives))
    print('True negatives:', len(true_negatives))
    print('False positives:', len(false_positives))
    print('False negatives:', len(false_negatives))
    
    return true_positives, true_negatives, false_positives, false_negatives