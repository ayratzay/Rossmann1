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


def categorical_value_handler(val):
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
    elif val == 'd':
        return 4
    else:
        return val


def data_cleaning(df):
    df['StateHoliday'] = df['StateHoliday'].apply(categorical_value_handler)
    return df


#If store is closed, sales will be always zero.
#Unless we have not found any other utilization of this information,
#I will drop all rows using mask train['Open'] == 0 and 'Open' column
#Also we should add to preprocess of test data same logic,
#for mask test['Open'] == 0 test['Sales'] = 0
#
# Do we need assume 0 for this
# column Open in test data file for store 622 is blank
# Is the store closed meaning open=0? is it the right assumption ID=480 1336 2192 3048 4760 5616 6472 7328 8184 9040 10752

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


def str_month_to_int(str_val):
    if str_val == 'Jan':
        return 1
    elif str_val == 'Feb':
        return 2
    elif str_val == 'Apr':
        return 3
    elif str_val == 'Mar':
        return 4
    elif str_val == 'May':
        return 5
    elif str_val == 'Jun':
        return 6
    elif str_val == 'Jul':
        return 7
    elif str_val == 'Aug':
        return 8
    elif str_val == 'Sept':
        return 9
    elif str_val == 'Oct':
        return 10
    elif str_val == 'Nov':
        return 11
    elif str_val == 'Dec':
        return 12
    else:
        return str_val


train = read_train_df()
train = set_weeks(train)


store = pd.read_csv("data/store.csv")
store['StoreType'] = store['StoreType'].apply(categorical_value_handler)
store['Assortment'] = store['Assortment'].apply(categorical_value_handler)
train['Promo2'] = 0
train['StoreType'] = 0
train['Assortment'] = 0
train['CompetitionDistance'] = 0

for i in store.columns:
    print (i, store[i].unique()[:10])


for row in store.itertuples():
    id =  row[1]
    print (id)
    mask = (train['Store'] == id)

    storeType =  row[2]
    assort =  row[3]
    train.loc[mask, 'StoreType'] = storeType
    train.loc[mask, 'Assortment'] = assort

    comp_distance = row[4]
    comp_month = row[5]
    comp_year = row[6]

    train.loc[mask, 'Assortment'] = assort


    train.loc[mask, 'CompetitionDistance'] = comp_distance



    has_promo2 = row[7]
    promo2week = row[8]
    promo2year = row[9]
    interval = row[10]

    if has_promo2:
        if promo2year > 2012:
            mask1 = (train['Store'] == id) & (train['Year'] == promo2year) & (train['Week'] == promo2week)
            mask2 = (train['Store'] == id) & (train['Year'] > promo2year)
            mask = mask1 & mask2

        for m in interval.split(','):
            pm = str_month_to_int(m)
            train.loc[(mask) & (train['Month'] == pm), 'Promo2'] = 1






    # Promo2 periods  #convert to promo2 column in train
    # competitors #competitor since months to today #has_competitor for each day{0,1} #distance {far, near, close}


aa = store.itertuples()
row = next(aa)
print (row)

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



