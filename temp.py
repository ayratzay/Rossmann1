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

train_x = pd.read_csv('pickle_cellar/train_full_x_1.csv')
train_y = pd.read_csv('pickle_cellar/train_full_y.csv', header=None)


np_x = train_x.as_matrix()
np_y = train_y.as_matrix().ravel()



X_train, X_test, y_train, y_test = train_test_split(np_x, np_y, test_size=0.8)



param_grid = {'n_estimators': 5000, # 5000
              'max_depth': 16,     # 16
              # 'max_features': 14,  # 16
              'min_samples_split': 20, #20 is just good
              'min_samples_leaf': 3,  #3 is just good
              'learning_rate': 0.01, # 0.01
              'subsample': 0.8,     # 0.8
              'loss': 'lad'}

print ('started at '  + time.strftime("%X"))
# clf = train_model(param_grid, X_train, y_train)
clf = train_model(param_grid, np_x, np_y)
print ('clf_is ready on '  + time.strftime("%X"))


n_estimators = len(clf.estimators_)
test_dev, train_dev = np.empty(n_estimators), np.empty(n_estimators)
for i, pred in enumerate(clf.staged_predict(X_test)):
    test_dev[i] = clf.loss_(y_test, pred)
    train_dev[i] = clf.train_score_[i]
plt.plot(test_dev, label="test error")
plt.plot(train_dev, label="train error")
plt.legend()

aa = pd.DataFrame(clf.feature_importances_, index = [cols]).sort_values(by=0, axis=0, ascending=False)/clf.feature_importances_.max()
aa.to_csv('feature_importance.csv')


test_dev1 # 647 MAE 8, 0.7, 20, 3, 1, 1, no_get_dummies, extra features: Mean_Sales
train_dev1

test_dev2   # 3000, 8, 0.7, 20, 3, 0.1, 0.5 all data
train_dev2

plt.plot(test_dev2, label="test error")
plt.plot(train_dev2, label="train error")
plt.legend()


test = pd.read_csv('pickle_cellar/test_data.csv')
np_test_x = test.as_matrix()
test_y_hat = clf.predict(np_test_x)
ind = range(1, test_y_hat.shape[0] + 1)
result = zip(ind, test_y_hat)
submission = pd.DataFrame(result, columns=["Id","Sales"])
submission.to_csv('submissions/gb_storeid_dow_model.csv', index=False)

from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence

for i in range(0, 16):
    for t in range(0, 16):
        if i != t:
            fig, axs = plot_partial_dependence(clf, np_x, [(i,t)], feature_names=train_x.columns,
                                   n_jobs=-1, grid_resolution=20)

features = [3, 14, (3, 14)]
fig, axs = plot_partial_dependence(clf, X_train, features, feature_names=train_x.columns,
                                   n_jobs=-1, grid_resolution=20)


from itertools import combinations
aa = combinations(range(0, 16), 2)
for i,t in aa:
    fig, axs = plot_partial_dependence(clf, np_x, [(i,t)], feature_names=train_x.columns,
                                   n_jobs=-1, grid_resolution=20)



pred = clf.predict(X_test)
new_df = np.hstack([X_test, y_test.reshape(y_test.shape[0], 1), pred.reshape(pred.shape[0], 1)])
new_df = pd.DataFrame(new_df, columns=list(cols) + ['Sales', 'Pred'])
new_df['delta'] = new_df['Sales'] - new_df['Pred']

