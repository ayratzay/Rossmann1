from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint, uniform
from time import time

#If store is closed, sales will be always zero.
#Unless we have not found any other utilization of this information,
#I will drop all rows using mask train['Open'] == 0 and 'Open' column
#Also we should add to preprocess of test data same logic,
#for mask test['Open'] == 0 test['Sales'] = 0
#
# Do we need assume 0 for this
# column Open in test data file for store 622 is blank
# Is the store closed meaning open=0? is it the right assumption ID=480 1336 2192 3048 4760 5616 6472 7328 8184 9040 10752



def read_test_df():
    t = pd.read_csv("data/test.csv")
    t = process_date(t)
    t = data_cleaning(t)
    t.loc[t['Open'].isnull(), 'Open'] = 1
    return t



def window_array(lowest, highest, window_width):
    temp_range = range(lowest, highest + 1)
    while len(temp_range):
        yield temp_range[0:window_width]
        temp_range = temp_range[window_width:]

# Thanks to Chenglong Chen for providing this in the forum
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w


def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe
########################################################

def cross_validate(x, y, nweeks, cv_number, estimator = False):
    mmin, mmax = min(nweeks), max(nweeks)
    length = int((mmax - mmin) / cv_number)

    RMSPE = []
    for wl in window_array(mmin, mmax, length):
        test_mask = np.in1d(nweeks, wl) # array of indexes in test sample
        X_train, y_train =  x[~test_mask], y[~test_mask]
        X_test, y_test =  x[test_mask], y[test_mask]

        if not estimator:
            clf = ensemble.GradientBoostingRegressor(n_estimators=100) # more is better
        else:
            clf = estimator
        clf.fit(X_train, y_train)
        RMSPE_cv = rmspe(clf.predict(X_test), y_test)
        RMSPE.append(RMSPE_cv)
        print (np.mean(clf.predict(X_test)), np.std(clf.predict(X_test)), np.mean(clf.predict(y_test)), np.std(clf.predict(y_test)))
        print ([int(i) for i in X_test[0, [5,6,7]]], [int(i) for i in X_test[-1, [5,6,7]]], RMSPE_cv)
    print("RMSPE: %.4f +- %.4f" % np.mean(RMSPE), 2 * np.mean(RMSPE))




# col_x = np.delete(train.columns, [2, 3, 10])  # 2 : Sales, 3 : Customers, 10 : NWeek
#
# np_x = train.as_matrix(columns=col_x)
# np_weekInd = train.as_matrix(columns=['NWeek'])
# np_y = train.as_matrix(columns=['Sales'])



# clf = ensemble.GradientBoostingRegressor(n_estimators=300)
# param_dist = {"max_depth": randint(3, 7),
#               "max_features": uniform(loc = 0.1, scale = 0.9),
#               "min_samples_split": randint(2, 11),
#               "min_samples_leaf": randint(1, 11),
#               'learning_rate': uniform(loc = 0.01, scale = 0.09)}
# n_iter_search = 100
# random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=2)
# start = time()
# random_search.fit(np_x, np_y.ravel())
# print("RandomizedSearchCV took %.2f seconds for %d candidates"
#       " parameter settings." % ((time() - start), n_iter_search))
#
#
# cross_validate(np_x, np_y, np_weekInd, 10, estimator=random_search.best_estimator_)