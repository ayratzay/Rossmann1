import pandas as pd
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


def data_cleaning(df):
    df['StateHoliday'] = df['StateHoliday'].apply(categorical_value_handler)
    return df


def read_test_df():
    t = pd.read_csv("data/test.csv")
    t = process_date(t)
    t = data_cleaning(t)
    t.loc[t['Open'].isnull(), 'Open'] = 1
    return t


def dist_to_int(val):
    if val == 0:
        return 0
    elif val <= 1000:
        return 1
    elif val <= 3000:
        return 2
    elif val <= 9000:
        return 3
    elif val <= 27000:
        return 4
    elif val > 27000:
        return 5
    else:
        return val


def month_to_int(val):
    if val == 0:
        return 0
    elif val <= 3:
        return 1
    elif val <= 9:
        return 2
    elif val <= 27:
        return 3
    elif val <= 81:
        return 4
    elif val > 81:
        return 5
    else:
        return val


def read_store_df():
    df = pd.read_csv("data/store.csv")
    df.loc[df['CompetitionDistance'].isnull(), 'CompetitionDistance'] = df['CompetitionDistance'].mean().round()
    df.loc[814, 'CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].mean().round()
    df['StoreType'] = df['StoreType'].apply(categorical_value_handler)
    df['Assortment'] = df['Assortment'].apply(categorical_value_handler)
    return df


def merge_df(t, s):
    for row in s.itertuples():
        id = row[1]
        print (id)
        mask = t['Store'] == id

        storeType = row[2]
        assort = row[3]
        t.loc[mask, 'StoreType'] = storeType
        t.loc[mask, 'Assortment'] = assort

        comp_distance = row[4]
        comp_month = row[5]
        comp_year = row[6]

        t.loc[mask, 'CompetitionDistance'] = comp_distance

        if not np.isnan(comp_month):

            t.loc[mask, 'CompetingMonths'] = \
                t.loc[mask, ['Year', 'Month']].apply(lambda row: (row['Year'] - comp_year) * 12 + row['Month'] - comp_month, axis=1)

            if comp_year < 2013:
                maskc = mask
            else:
                mask1c = (mask) & (t['Year'] == comp_year) & (t['Month'] >= comp_month)
                mask2c = (mask) & (t['Year'] > comp_year)
                maskc = mask1c | mask2c

            t.loc[maskc, 'HasCompetitor'] = 1
        else:
            t.loc[mask, 'HasCompetitor'] = 0



        has_promo2 = row[7]
        promo2week = row[8]
        promo2year = row[9]
        interval = row[10]

        if has_promo2:
            if promo2year < 2013:
                maskp = mask
            else:
                mask1p = (mask) & (t['Year'] == promo2year) & (t['Week'] >= promo2week)
                mask2p = (mask) & (t['Year'] > promo2year)
                maskp = mask1p | mask2p

            for m in interval.split(','):
                pm = str_month_to_int(m)
                t.loc[(maskp) & (t['Month'] == pm), 'Promo2'] = 1

    # Promo2 periods  #convert to promo2 column in train
    # competitors #competitor since months to today #has_competitor for each day{0,1} #distance {far, near, close}

    return t


def get_dummies(t):
    dow_df = pd.get_dummies(t['DayOfWeek'], prefix='DOW')
    sh_df = pd.get_dummies(t['StateHoliday'], prefix='SH')
    year_df = pd.get_dummies(t['Year'], prefix='Year')
    month_df = pd.get_dummies(t['Month'], prefix='Month')
    st_df = pd.get_dummies(t['StoreType'], prefix='ST')
    as_df = pd.get_dummies(t['Assortment'], prefix='AS')
    cd_df = pd.get_dummies(t['CompetitionDistance'], prefix='CD')
    cm_df = pd.get_dummies(t['CompetingMonths'], prefix='CM')
    hc_df = pd.get_dummies(t['HasCompetitor'], prefix='HC')


    t = t.drop(['DayOfWeek', 'StateHoliday', 'Year', 'Month', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetingMonths', 'HasCompetitor'], axis = 1)
    t = pd.concat([t, dow_df, sh_df, year_df, month_df, st_df, as_df, cd_df, cm_df, hc_df], axis =1)
    return t


test = read_test_df()
store = read_store_df()

test['Promo2'] = 0
test['StoreType'] = 0
test['Assortment'] = 0
test['CompetitionDistance'] = 0
test['HasCompetitor'] = -1
test['CompetingMonths'] = 0

test = merge_df(test, store)
test.loc[test['CompetingMonths'] < 0, 'CompetingMonths'] = 0
test.loc[test['Open'].isnull(), 'Open'] = 1
test[u'CompetitionDistance'] = test[u'CompetitionDistance'].apply(dist_to_int)
test[u'CompetingMonths'] = test[u'CompetingMonths'].apply(month_to_int)
test = get_dummies(test)

train_columns = pd.read_csv('pickle_cellar/train_full_x.csv', nrows = 1).columns

ndf = pd.DataFrame()
for col in train_columns:
    if col in test.columns:
        ndf[col] = test[col]
    else:
        ndf[col] = 0

ndf.to_csv('pickle_cellar/test_data.csv', index=False)