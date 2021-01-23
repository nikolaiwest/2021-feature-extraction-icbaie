if __name__ == '__main__':

    import timeit
    import numpy as np
    import pandas as pd
    import datetime as dt

    from ast import literal_eval
    from pyts.datasets import load_gunpoint
    from sklearn.metrics import accuracy_score, recall_score, precision_score

    from sktime_dl.deeplearning import CNNClassifier, FCNClassifier, MLPClassifier, \
        InceptionTimeClassifier, ResNetClassifier, EncoderClassifier

    X_train, X_test, y_train, y_test = load_gunpoint(return_X_y = True) # get gunpoint data from pyts

    # as of the date of this publication, please note the following issue needs to be fixed mannually before running
    # https://github.com/sktime/sktime-dl/issues/79

    '''
    def prepare_for_sktime_dl(data):
        df = pd.DataFrame(data)
        cols = df.columns
        df['dim_0'] = df[cols].apply(lambda row: ', '.join(row.values.astype(str)), axis=1)
        df['dim_0'] = ['[' + i + ']' for i in df.dim_0 ]
        df['dim_0'] = df['dim_0'].apply(literal_eval)
        return df.drop(columns=cols)'''

    X_train = np.reshape(X_train, X_train.shape + (1,))
    X_test = np.reshape(X_test, X_test.shape + (1,))

    print('Train: ' + str(X_train.shape))
    print('Test:  ' + str(X_test.shape))

    names = ['CNN', 'FCN', 'MLP', 'InceptionTime', 'ResNet', 'Encoder']

    epochs = 500
    batch = 16

    classifiers = [
        CNNClassifier(nb_epochs=epochs, batch_size=batch, verbose=False),
        FCNClassifier(nb_epochs=epochs, batch_size=batch, verbose=False),
        MLPClassifier(nb_epochs=epochs, batch_size=batch, verbose=False),
        InceptionTimeClassifier(nb_epochs=epochs, verbose=False),
        ResNetClassifier(nb_epochs=epochs, batch_size=batch, verbose=False),
        EncoderClassifier(nb_epochs=epochs, batch_size=batch, verbose=False),]

    for name, clf in zip(names, classifiers):
        start = timeit.default_timer()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        duration = timeit.default_timer() - start
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        print(str(' - '+ name + ': \n accuracy: ' + '%.4f' % accuracy + ' | recall: ' + '%.4f' % recall \
            + ' | precision: ' + '%.4f' % precision + ' | duration: ' + '%.4f' % duration)) 

    '''
    # Convolutional Neural Network Classifier: 
    from sklearn.model_selection import GridSearchCV # Parameter tuning

    network = CNNClassifier(nb_epochs=200, verbose=False)
    network.fit(X_train, y_train)
    score_cnnc = network.score(X_test, y_test)
    print('score cnnc: ' + str(score_cnnc))

    param_grid = {'nb_epochs': [50, 100, 200, 500, 1000],
                'kernel_size': [5, 7, 9, 11, 13] }
    grid = GridSearchCV(network, param_grid=param_grid, cv=5)   
    grid.fit(X_train, y_train)    
    print("Best cross-validation accuracy: {:.4f}".format(grid.best_score_))
    print("Test set score: {:.4f}".format(grid.score(X_test, y_test)))
    print("Best parameters: {}".format(grid.best_params_))
    '''
