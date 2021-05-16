import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

import pickle
import argparse


def train(path):
    train_df = pd.read_csv(path, sep='\t')

    X_train, y_train = preprocess(train_df)
    y_train = np.log(y_train)
    
    model = XGBRegressor(n_estimators=900,
                         max_depth=3,
                         eta=0.05,
                         subsample=0.9,
                         colsample_bytree=1.0,
                         objective='reg:squarederror',
                         verbosity=1)
    
    
    """
    model = KNeighborsRegressor(10)

    scaler = MinMaxScaler(feature_range=(0, 1))

    x_train_scaled = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(x_train_scaled)
    """

    """
    model = AdaBoostRegressor(n_estimators=800, 
                              learning_rate=0.1,
                              loss='linear')
    """


    model.fit(X_train, y_train)

    outfile = 'model.sav'
    pickle.dump(model, open(outfile, 'wb'))

    y_pred = np.exp(model.predict(X_train))
    print(f"Train error: {eval(y_pred, np.exp(y_train))}")


def eval(y_pred, y_true):
    return np.sqrt(np.sum((np.log(y_pred + 1) - np.log(y_true + 1))**2) / len(y_true))


def preprocess(df, is_train=True):
    predictor_features = ["budget", "popularity", "runtime", "vote_count", 'vote_average', 'belongs_to_collection']
    target_feature = 'revenue'

    df_new = df.copy()
    df_new["budget"] = df_new["budget"].apply(lambda row: np.nan if row <= 100 else row)

    runtime_median = df_new['runtime'].median()
    df_new["runtime"] = df_new["runtime"].apply(lambda row: np.nan if np.isnan(row) else row)

    df_new["belongs_to_collection"] = df_new["belongs_to_collection"].apply(lambda row: 0 if pd.isna(row) else 1)

    if is_train:
        to_impute = df_new[["budget", "popularity", "revenue", "runtime", "vote_count"]].copy()
        features = ["budget", "popularity", "revenue", "runtime", "vote_count"]
    else:
        to_impute = df_new[["budget", "popularity", "runtime", "vote_count"]].copy()
        features = ["budget", "popularity", "runtime", "vote_count"]

    imputed_df = impute(to_impute.copy(), 'mice_reg', features)
    #imputed_df = impute(to_impute.copy(), 'mice_knn', features)
    df_new["budget"] = imputed_df['budget']
    df_new['runtime'] = imputed_df['runtime']

    def get_X_y(df):
        X = df[predictor_features].to_numpy()

        y = None
        if target_feature in df.columns:
            y = df[target_feature].to_numpy()
        return X, y

    return get_X_y(df_new)



def get_id_field(row):
    ids = []
    for dic in row:
        ids.append(dic['id'])
    return ids


def impute(to_impute, method, features):
    if method == 'mice_reg':
        reg = ExtraTreesRegressor(n_estimators=100, random_state=0)
        imputer = IterativeImputer(estimator=reg, random_state=0, verbose=0)
    else:
        knn = KNeighborsRegressor(n_neighbors=10)
        imputer = IterativeImputer(estimator=knn, random_state=0, verbose=0)

    imputed = imputer.fit_transform(to_impute)
    imputed_df = pd.DataFrame(imputed, columns=features)
    return imputed_df


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='Process input')
    parser.add_argument('tsv_path', type=str, help='tsv file path')
    args = parser.parse_args()

    train(path=args.tsv_path)