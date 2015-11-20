import pandas as pd
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint, uniform
from time import time
import numpy as np

def process_date(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df['Year'] = df['Date'].apply(lambda x: x.year)
    df['Month'] = df['Date'].apply(lambda x: x.month)
    df['Week'] = df['Date'].apply(lambda x: x.week)
    df = df.drop(['Date'], axis = 1)
    return df


def stateHolidayColumnHandler(val):
    if val == '0':
        return 0
    elif val == 0:
        return 0
    elif val == 'a':
        return 1
    elif val == 'b':
        return 2
    elif val == 'c':
        return 3
    else:
        return val


def data_cleaning(df):
    df['StateHoliday'] = df['StateHoliday'].apply(stateHolidayColumnHandler)
    # df['DayOfWeek'] = df['DayOfWeek'].apply(lambda x: str(x))
    return df


#If store is closed, sales will be always zero.
#Unless we have not found any other utilization of this information,
#I will drop all rows using mask train['Open'] == 0 and 'Open' column
#Also we should add to preprocess of test data same logic,
#for mask test['Open'] == 0 test['Sales'] = 0

def drop_closed_days_rows(t):
    t = t[t['Open'] != 0]
    t = t.drop(['Open'], axis = 1)
    return t

def read_train_df():
    t = pd.read_csv("data/train.csv", low_memory=False)
    t = drop_closed_days_rows(t)
    t = process_date(t)
    t = data_cleaning(t)
    # t['Date'] = (pd.to_datetime('2015-08-01') - t['Date']).astype('timedelta64[D]')
    return t


def read_test_df():
    t = pd.read_csv("data/test.csv")
    t = process_date(t)
    t = data_cleaning(t)
    return t


def get_dummies(t):
    dow_df = pd.get_dummies(t['DayOfWeek'], prefix='DOW')
    sh_df = pd.get_dummies(t['StateHoliday'], prefix='SH')
    t = t.drop(['DayOfWeek', 'StateHoliday'], axis = 1)
    t = pd.concat([t, dow_df, sh_df], axis =1)
    return t


def one_hot_encoder(t):
    # dow_df = pd.get_dummies(t['DayOfWeek'], prefix='DOW')
    # sh_df = pd.get_dummies(t['StateHoliday'], prefix='SH')
    # t = t.drop(['DayOfWeek', 'StateHoliday'], axis = 1)
    # t = pd.concat([t, dow_df, sh_df], axis =1)
    return t


def set_weeks(t):
    t['NWeek'] = t[[u'Year', u'Week']].apply(lambda row: (row['Year'] - 2013) * 52 + row['Week'], axis=1)
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
        X_train, y_train =  x[~test_mask], y[~test_mask].ravel()
        X_test, y_test =  x[test_mask], y[test_mask].ravel()

        if not estimator:
            clf = ensemble.GradientBoostingRegressor(n_estimators=20) # more is better
        else:
            clf = estimator
        clf.fit(X_train, y_train)
        RMSPE_cv = rmspe(clf.predict(X_test), y_test)
        RMSPE.append(RMSPE_cv)
        print (X_test[0, -3:], X_test[-1, -3:], RMSPE_cv)
    print("RMSPE: %.4f" % np.mean(RMSPE))


train = read_train_df()
train = set_weeks(train)


col_x = np.delete(train.columns, [2, 3, 10])  # 2 : Sales, 3 : Customers, 10 : NWeek

np_x = train.as_matrix(columns=col_x)
np_weekInd = train.as_matrix(columns=['NWeek'])
np_y = train.as_matrix(columns=['Sales'])


cross_validate(np_x, np_y, np_weekInd, 10)

clf = ensemble.GradientBoostingRegressor(n_estimators=100)
param_dist = {"max_depth": randint(3, 7),
              "max_features": uniform(loc = 0.1, scale = 0.9),
              "min_samples_split": randint(2, 11),
              "min_samples_leaf": randint(1, 11),
              'learning_rate': uniform(loc = 0.01, scale = 0.09)}
n_iter_search = 100
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search)
start = time()
random_search.fit(np_x, np_y.ravel())
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))



cross_validate(np_x, np_y, np_weekInd, 10, estimator=random_search.best_estimator_)



train_d = get_dummies(train)

np_x = train_d.as_matrix(columns=[u'Store', u'Promo', u'SchoolHoliday', u'Year', u'Month',
                                   u'Week', u'DOW_1', u'DOW_2', u'DOW_3', u'DOW_4', u'DOW_5', u'DOW_6',
                                   u'DOW_7', u'SH_0', u'SH_1', u'SH_2', u'SH_3'])
np_y = train_d.as_matrix(columns=['Sales']).ravel()
np_weekInd = train_d.as_matrix(columns=['NWeek'])

clf = ensemble.GradientBoostingRegressor(n_estimators=100)
param_dist = {"max_depth": randint(3, 7),
              "max_features": uniform(loc = 0.1, scale = 0.9),
              "min_samples_split": randint(2, 11),
              "min_samples_leaf": randint(1, 11),
              'learning_rate': uniform(loc = 0.01, scale = 0.09)}
n_iter_search = 100
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search)
start = time()
random_search.fit(np_x, np_y.ravel())
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))

cross_validate(np_x, np_y, np_weekInd, 10, estimator=random_search.best_estimator_)

print (random_search.grid_scores_)

# # plt.plot(train.groupby(['Year', 'Month', 'Week']).sum()['Sales'])




test = read_test_df()  # test df has first column Id
test = set_weeks(test)


# store = pd.read_csv("data/store.csv")

# print(store.shape)
# print (store.head(5))
# print (store.dtypes)

# store['PromoInterval'].str.split(',')
# test = pd.read_csv("../input/test.csv")


# print(train.describe())
# print(train.shape)
# print(store.shape)
# print (store.head(5))
# print (train.columns)
# print (train.dtypes)
# print(test.shape)
# print (train.head(5))
# print (train['Date'][0])

# print (train['Store'].nunique())
# print (train['Date'].unique())