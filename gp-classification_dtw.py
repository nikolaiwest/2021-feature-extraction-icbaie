if __name__ == '__main__':

    import timeit
    import numpy as np

    from scipy import interpolate
    from dtaidistance import dtw
    from shapely.geometry import Point
    from shapely.geometry import LineString
    from sklearn.preprocessing import MinMaxScaler
    from pyts.datasets import load_gunpoint

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.metrics import accuracy_score, recall_score, precision_score

    def longest_streak(condition):
        c = np.r_[False,condition,False]
        idx = np.flatnonzero(c[:-1]!=c[1:]) # get indices where switch between True/False occurs
        return (idx[1::2]-idx[::2]).max() # return longest distance between switches

    def extract_features(path):
        return np.array([
        path.max(),                             
        #np.argmax(path),    
        path.min(),          
        #np.argmin(path),    
        sum(path[path < 0]),
        sum(path[path > 0]),
        # slope of linear spline approximation
        max(np.diff(path[::9])), 
        min(np.diff(path[::9]))
        #longest_streak(path > path.mean()), 
        #longest_streak(path < path.mean()) 
        ])

    def reduce(x):
        x = x[~np.isnan(x)]
        x = x[-200:]
        return x[x >= 1]

    def projected_warp_path(curve, refCurve):
        
        refCurve = scaler.fit_transform(refCurve.reshape(-1,1)).squeeze()
        curve = scaler.fit_transform(curve.reshape(-1,1)).squeeze()
        
        dist, cumcost = dtw.warping_paths_fast(refCurve, curve)
        steps = dtw.best_path(cumcost) # return steps of warp path
        
        # define Euclidean line l
        l_start = np.array([0,0])
        l_end = np.array([len(refCurve)-1,len(curve)-1])
        l = LineString([l_start, l_end])
        
        # calculate orthogonal distance to l (sign corresponds to direction of projection) 
        dist_orthogonal = np.cross(l_end-l_start,steps-l_start)/np.linalg.norm(l_end-l_start)
        
        # calculate coordinates z on l of orthogonal projections
        z = list()
        for step in steps:
            p = Point(step)
            z.append(l.project(p))
        z = np.array(z)/l.length*100
        
        # map projected warp path to vector of length 100
        f = interpolate.interp1d(z, dist_orthogonal)    
        mapped_projection = f(np.arange(0,100))

        return dist, mapped_projection

    def chunks(l, n):
        for i in np.arange(0, len(l), n):
            yield l[i:i+n]

    def calculate_features(df, neighbors, reduce = False):
        
        ff = list()  
        for data in chunks(np.array(df), neighbors+1): 
            
            for i, x in enumerate(data): 
                
                if reduce == True: x = reduce(x)
                paths = list()
                dists = list()
                for y in np.delete(data, i, axis = 0):
                    
                    if reduce == True: y = reduce(y)
                    
                    dist, path = projected_warp_path(x,y)
                    paths.append(path)
                    dists.append(dist)
                
                avg_path = np.median(np.array(paths), axis = 0)
                                
                ff.append(np.append(extract_features(avg_path), np.array(dists).mean()))

        return np.vstack(ff)

    scaler = MinMaxScaler()

    X_train, X_test, y_train, y_test = load_gunpoint(return_X_y = True) # get gunpoint data from pyts

    # extract features using the dynamic time warping approach
    start1 = timeit.default_timer()
    f_train = calculate_features(X_train, neighbors = 5, reduce = False)
    print('Train - computation time: ', timeit.default_timer() - start1) 

    # as for tsfresh, we calcuate train and test independently
    start2 = timeit.default_timer()
    f_test = calculate_features(X_test, neighbors = 5, reduce = False)
    print('Test - computation time: ', timeit.default_timer() - start2) 

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
# %%
