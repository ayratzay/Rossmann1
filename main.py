import pandas as pd
import matplotlib.pyplot as plt
from sklearn import ensemble
import numpy as np
from sklearn.cross_validation import train_test_split
import pickle
import time

def train_model(params, X, y):
    model = ensemble.GradientBoostingRegressor(**param_grid)
    model.fit(X, y)
    return model


train_x = pd.read_csv('pickle_cellar/train_full_x.csv')
train_y = pd.read_csv('pickle_cellar/train_full_y.csv', header=None)

with open('pickle_cellar/nr_list.pickle', 'rb') as handle:
    nr_list = pickle.load(handle)
with open('pickle_cellar/oe_list.pickle', 'rb') as handle:
    oe_list = pickle.load(handle)
with open('pickle_cellar/ue_list.pickle', 'rb') as handle:
    ue_list = pickle.load(handle)

np_x = train_x.as_matrix()
np_y = train_y.as_matrix().ravel()


np_x_nr, np_y_nr = np_x[np.in1d(np_x[:, 0], nr_list), :], np_y[np.in1d(np_x[:, 0], nr_list)]
np_x_oe, np_y_oe = np_x[np.in1d(np_x[:, 0], oe_list), :], np_y[np.in1d(np_x[:, 0], oe_list)]
np_x_ue, np_y_ue = np_x[np.in1d(np_x[:, 0], ue_list), :], np_y[np.in1d(np_x[:, 0], ue_list)]


X_train_nr, X_test_nr, y_train_nr, y_test_nr = train_test_split(np_x_nr, np_y_nr)
X_train_oe, X_test_oe, y_train_oe, y_test_oe = train_test_split(np_x_oe, np_y_oe)
X_train_ue, X_test_ue, y_train_ue, y_test_ue = train_test_split(np_x_ue, np_y_ue)


param_grid = {'n_estimators': 1000,
              'max_depth': 6,
              'max_features': 'auto',
              'min_samples_split': 3,
              'min_samples_leaf': 3,
              'learning_rate': 0.05,
              'subsample': 0.5,
              'loss': 'ls'}


print ('started at '  + time.strftime("%X"))
clf_oe = train_model(param_grid, X_train_oe, y_train_oe)
print ('clf_oe is ready on '  + time.strftime("%X"))
clf_ue = train_model(param_grid, X_train_ue, y_train_ue)
print ('clf_ue is ready on '  + time.strftime("%X"))
clf_nr = train_model(param_grid, X_train_nr, y_train_nr)
print ('clf_nr is ready on '  + time.strftime("%X"))

#
# n_estimators = len(clf_oe.estimators_)
#
# test_dev = np.empty(n_estimators)
#
# for i, pred in enumerate(clf_oe.staged_predict(X_test_oe)):
#     test_dev[i] = clf_oe.loss_(y_test_oe, pred)
#
# plt.plot(test_dev)

##########################################
##### NOW WE WILL SUBMIT RESULT ##########
##########################################

test = read_test_df()
test.loc[test['Open'].isnull(), 'Open'] = 1

test['Promo2'] = 0
test['StoreType'] = 0
test['Assortment'] = 0
test['CompetitionDistance'] = 0
test['HasCompetitor'] = -1
test['CompetingMonths'] = 0

test = merge_df(test, store)
test.loc[test['CompetingMonths'] < 0, 'CompetingMonths'] = 0

test[u'CompetitionDistance'] = test[u'CompetitionDistance'].apply(dist_to_int)
test[u'CompetingMonths'] = test[u'CompetingMonths'].apply(month_to_int)

test = get_dummies(test)

ndf = pd.DataFrame()
for col in train.columns:
    if col in test.columns:
        ndf[col] = test[col]
    else:
        ndf[col] = 0

# for col in test.columns:
#     print (col, test[col].unique()[:10])

col_x = np.delete(ndf.columns, [1, 2, 6])  # 0 : Id, 3 : Open
np_x_test = ndf.as_matrix(columns=col_x)

y_results = clf.predict(np_x_test)
submition = pd.read_csv('data/sample_submission.csv')
submition['Sales'] = y_results
submition.to_csv('data/submission.csv', index=False)

######################################################
######## Creating new models for data ###############
#####################################################






y_hat = clf.predict(X_test)

rmspe(y_hat, y_test)

X_test.shape
np_test = np.hstack([X_test, y_test.reshape(y_test.shape[0], 1), y_hat.reshape(y_hat.shape[0], 1)])

col_x1 = list(col_x) + ['Sales'] + ['Sales_hat']
test_df = pd.DataFrame(np_test, columns=col_x1)
test_df['delta'] = test_df['Sales'] - test_df['Sales_hat']
mean_error_bby_store = list(test_df.groupby('Store').mean()['delta'].sort_values())
plt.plot(mean_error_bby_store)

# fi_df = pd.DataFrame(zip(col_x1, clf.feature_importances_))
# fi_df.sort_values(1, ascending=False)

plt.plot(list(test_df.groupby('Store').mean()['delta'].order()))
test_df.groupby('Store').std()['delta'].order()

over_estimated_stores = test_df.groupby('Store').mean()['delta'].order().iloc[:200].index
under_estimated_stores = test_df.groupby('Store').mean()['delta'].order().iloc[950:].index
normal_estimated_stores = test_df.groupby('Store').mean()['delta'].order().iloc[200:950].index

oe_mask = train['Store'].isin(over_estimated_stores)
ue_mask = train['Store'].isin(under_estimated_stores)
nr_mask = train['Store'].isin(normal_estimated_stores)

# import pickle
# with open('pickle_cellar/oe_list.pickle', 'wb') as handle:
#   pickle.dump([int(i) for i in list(over_estimated_stores)], handle)
# with open('pickle_cellar/ue_list.pickle', 'wb') as handle:
#   pickle.dump([int(i) for i in list(under_estimated_stores)], handle)
# with open('pickle_cellar/nr_list.pickle', 'wb') as handle:
#   pickle.dump([int(i) for i in list(normal_estimated_stores)], handle)

# with open('pickle_cellar/nr_list.pickle', 'rb') as handle:
#   b = pickle.load(handle)


train_oe = train[oe_mask]
train_ue = train[ue_mask]
train_ne = train[nr_mask]

col_x = np.delete(train.columns, [1, 2, 6])  # 2 : Sales, 3 : Customers, 10 : NWeek

np_x = train.as_matrix(columns=col_x)
np_weekInd = train.as_matrix(columns=['NWeek']).ravel()
np_y = train.as_matrix(columns=['Sales']).ravel()

plt.plot(list(test_df[test_df['Year_2013'] == 1].groupby('Week').mean()['delta']))
plt.plot(list(test_df[test_df['Year_2014'] == 1].groupby('Week').mean()['delta']))
plt.plot(list(test_df[test_df['Year_2015'] == 1].groupby('Week').mean()['delta']))

plt.plot(list(test_df[test_df['Promo'] == 1].groupby('Week').mean()['delta']))
plt.plot(list(test_df[test_df['Promo'] == 0].groupby('Week').mean()['delta']))

plt.plot(list(test_df[test_df['Year_2013'] == 1].groupby('Week').mean()['Sales']))
plt.plot(list(test_df[test_df['Year_2014'] == 1].groupby('Week').mean()['Sales']))
plt.plot(list(test_df[test_df['Year_2015'] == 1].groupby('Week').mean()['Sales']))

plt.plot(list(test_df[(test_df['Year_2014'] == 1) & (test_df['Promo'] == 0)].groupby('Week').mean()['Sales']))
plt.plot(list(test_df[(test_df['Year_2014'] == 1) & (test_df['Promo'] == 1)].groupby('Week').mean()['Sales']))

test_over_df = test_df[test_df['Store'].isin(over_estimated_stores)]
over_heat_df = pd.pivot_table(test_over_df, values='delta', index=['Store'], columns=['DayOfWeek'], aggfunc=np.mean)
over_heat_df.to_csv('temp/month_year_OE.csv')

test_over_df.groupby([u'Year', 'Month']).mean()['delta']
test_over_df.columns

test_under_df = test_df[test_df['Store'].isin(under_estimated_stores)]
test_under_df.groupby(['HasCompetitor']).mean()['delta']
test_under_df.columns

plt.plot(over_heat_df.iloc[0, :])
over_heat_df.plot()
plt.pcolor(over_heat_df)
plt.yticks(np.arange(0.5, len(over_heat_df.index), 1), over_heat_df.index)
plt.xticks(np.arange(0.5, len(over_heat_df.columns), 1), over_heat_df.columns)
plt.show()
