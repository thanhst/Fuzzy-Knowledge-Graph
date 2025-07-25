from sklearn.feature_selection import SelectKBest, f_classif

def select_features(X, y, k=20):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new

# def select_features_df(X_df, y, k=10):
#     selector = SelectKBest(score_func=f_classif, k=k)
#     selector.fit(X_df, y)
#     selected_columns = X_df.columns[selector.get_support()]
#     X_new = X_df[selected_columns]
#     return X_new, selected_columns