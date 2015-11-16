__author__ = 'Freeman'

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint, uniform
from time import time

def process_date(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df['Year'] = df['Date'].apply(lambda x: x.year)
    df['Month'] = df['Date'].apply(lambda x: x.month)
    df['Week'] = df['Date'].apply(lambda x: x.week)
    df = df.drop(['Date'], axis = 1)
    return df


def data_cleaning(df):
    df['StateHoliday'] = df['StateHoliday'].apply(lambda x: '0' if x == 0 else x)
    df['DayOfWeek'] = df['DayOfWeek'].apply(lambda x: str(x))
    return df


def read_train_df():
    t = pd.read_csv("data/train.csv")
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


aa = window_array(9, 10, 10)
aa.next()

# cross_validate same year
# cross_validate same periods

train = read_train_df()
train = set_weeks(train)

train = get_dummies(train)
# plt.plot(train.groupby(['Year', 'Month', 'Week']).sum()['Sales'])


X_train = train.as_matrix(columns=[u'Store', u'Open', u'Promo', u'SchoolHoliday', u'Year', u'Month',
                                   u'Week', u'DOW_1', u'DOW_2', u'DOW_3', u'DOW_4', u'DOW_5', u'DOW_6',
                                   u'DOW_7', u'SH_0', u'SH_a', u'SH_b', u'SH_c'])
y_train = train.as_matrix(columns=['Sales']).ravel()


param_dist = {"max_depth": randint(3, 7),
              "max_features": uniform(loc = 0.1, scale = 0.9),
              "min_samples_split": randint(2, 11),
              "min_samples_leaf": randint(1, 11),
              'learning_rate': uniform(loc = 0.01, scale = 0.09)}

clf = ensemble.GradientBoostingRegressor(n_estimators=1000) # more is better

n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)
start = time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))



clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)

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