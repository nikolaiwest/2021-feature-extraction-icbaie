if __name__ == '__main__':

    import timeit
    import pandas as pd
    import matplotlib.pyplot as plt 
    
    from matplotlib.lines import Line2D
    from tsfresh import extract_features
    from pyts.datasets import load_gunpoint

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.metrics import accuracy_score, recall_score, precision_score

    X_train, X_test, y_train, y_test = load_gunpoint(return_X_y = True) # get gunpoint data from pyts
    print('Train: ' + str(X_train.shape))
    print('Test:  ' + str(X_test.shape))

    '''
    for data in [X_train]: #, X_test]: # explore values by visualization
        fig= plt.figure(figsize=(6,4))
        colors = [ 'red' if y==1 else 'blue' for y in y_train ]
        for x,c in zip(data, colors):
            plt.plot(x, color=c, linewidth=0.35)
        plt.title('gunpoint data (train)')
        plt.ylabel('movement reading')
        plt.xlabel('time')
        # add custom labels
        custom_lines = [Line2D([0], [0], color='red', lw=2), Line2D([0], [0], color='blue', lw=2)]
        plt.legend(custom_lines, ['Class 1 (point)', 'Class 2 (draw)'])
        plt.show()
    '''

    def prepare_for_tsfresh(data): # prepare for a tsfresh format
        v = i = t = k = [] # ['values', 'id', 'time', 'kind']
        for j in range(len(data)):
            v = v + list(data[j])
            i = i + [j] * len(data[j])
            t = t + list(range(1,len(data[j])+1))
            k = k + ['gp'] * len(data[j])
        return pd.DataFrame(data={'values': v, 'id': i, 'time':t, 'kind': k}, index=None)

    # extract features using tsfresh with stacked (or long) DataFrame
    df_train = prepare_for_tsfresh(X_train)
    start1 = timeit.default_timer()
    f_train = extract_features(df_train, column_id='id', column_sort='time', column_kind='kind', column_value='values')
    print('Train - computation time: ', timeit.default_timer() - start1) 

    # to avoid misconceptions about splitting issues, we calculate both independently
    df_test = prepare_for_tsfresh(X_test)
    start2 = timeit.default_timer()
    f_test = extract_features(df_test, column_id='id', column_sort='time', column_kind='kind', column_value='values')
    print('Test - computation time: ', timeit.default_timer() - start2) 

    # remove nan features due to calculation mismatch
    nan_train = f_train.columns[f_train.isna().any()].tolist()
    nan_test = f_test.columns[f_train.isna().any()].tolist()
    f_train = f_train.drop(list(set(nan_train + nan_test)), axis=1)
    f_test = f_test.drop(list(set(nan_train + nan_test)), axis=1)
    print(str(len(list(dict.fromkeys(list(set(nan_train + nan_test)))))) + ' features were dropped due to mismatch.')

    # apply various sklearn classifier
    names = ["Nearest Neighbors", "Naive Bayes", "Decision Tree", 
            "Neural Net (MLP)", "Random Forest", "AdaBoost", ] 

    classifiers = [
        KNeighborsClassifier(5, weights='uniform'),
        GaussianNB(),
        DecisionTreeClassifier(max_depth=None),
        MLPClassifier(alpha=1, max_iter=1000),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(), ]

    for name, clf in zip(names, classifiers):
        start = timeit.default_timer()
        clf.fit(f_train, y_train)
        y_pred = clf.predict(f_test)
        duration = timeit.default_timer() - start
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        print(str(' - '+ name + ': \n accuracy: ' + '%.4f' % accuracy + ' | recall: ' + '%.4f' % recall \
            + ' | precision: ' + '%.4f' % precision + ' | duration: ' + '%.4f' % duration))
