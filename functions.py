# import libraries
from warnings import simplefilter
simplefilter("ignore")
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
np.random.seed(33)

def clean_dataset(df, df_name=None):
    """
    Returns clean dataframe.
    
    Input: 
    df: Dataframe to be cleaned
    df_name: if dataframe is mailout_test dataframe we don't drop null rows
    
    Output:
    clean dataframe
    """
    # Convert column 'OST_WEST_KZ' from category to int
    print("Cleaning data is in progress please wait", '\n')
    print("Converting 'OST_WEST_KZ' column to 1 for 'W' and 0 for 'O'")
    df['OST_WEST_KZ'] = df['OST_WEST_KZ'].map({'W': 1, 'O': 0})
    df['LNR'] = df['LNR'].astype('category')
    print("Dropping most twenty null columns")
    print("Dropping Correlated Features")
    if df_name == 'azdias':
        # drop most twenty null values columns
        null_cols = 100 * df.isnull().sum() / df.shape[0]
        null_cols = list(null_cols.index[:20])
        # drop high correlation columns
        corr_matrix = df.corr().abs().round(2)
        corr_features = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.9:
                    col = corr_matrix.columns[i]
                    corr_features.add(col)
    
        global drop_cols
        drop_cols = []
        for i in null_cols:
            if i != 'LNR':
                drop_cols.append(i)
        for i in corr_features:
            if i != 'LNR':
                drop_cols.append(i)
        # drop 'EINGEFUEGT_AM'
        drop_cols.append('EINGEFUEGT_AM')
        drop_cols = list(set(drop_cols))
        
        df.drop(drop_cols, axis=1, inplace=True)

    else:
        df.drop(drop_cols, axis=1, inplace=True)    
    
    if df_name == 'mailout_train' or df_name == 'mailout_test':
        # Drop 'LNR' from other datasets
        df.drop('LNR', axis=1, inplace=True)
        
    # Drop rows with more than 10% missing values if the data is not mailout_test
    if df_name == 'mailout_test':
        rows_dropped = 0
    else:
        print("Dropping null rows more than 10%")
        rows_dropped = 100 * df.isnull().mean(axis=1)
        df = df[rows_dropped <= 10]
    print("Cleaning process is done! your data is ready", '\n')
    print(f"New shape after cleaning is: {df.shape}")
    return df


def evaluate(model, X, y):
    """
    Returns roc_curve metric for ML model after we select best features from the dataset.
    
    Input: 
    model: ML model to evaluate
    X: features
    y: target
    
    Output:
    plot the result with model name
    """
    print(model)
    # select best features
    FeatureSelection = SelectFromModel(estimator=model, max_features=None)
    X_reduced = FeatureSelection.fit_transform(X, y)
    #showing X Dimension 
    print('New X Shape is ' , X_reduced.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=33)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict_proba(X_train)[:,1]
    y_pred_test = model.predict_proba(X_test)[:,1]

    #train data ROC
    fpr_tr, tpr_tr, threshold = roc_curve(y_train, y_pred_train)
    roc_auc_train = auc(fpr_tr, tpr_tr)

    #test data ROC
    fpr_ts, tpr_ts, threshold = roc_curve(y_test, y_pred_test)
    roc_auc_test = auc(fpr_ts, tpr_ts)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr_tr, tpr_tr, 'g', label = 'Training AUC = %0.2f' % roc_auc_train)
    plt.plot(fpr_ts, tpr_ts, 'b', label = 'Testing AUC = %0.2f' % roc_auc_test)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    print('-'*40)