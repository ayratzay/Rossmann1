import pandas as pd
import numpy as np


def process_date(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df['Year'] = df['Date'].apply(lambda x: x.year)
    df['Month'] = df['Date'].apply(lambda x: x.month)
    df['Week'] = df['Date'].apply(lambda x: x.week)
    df = df.drop(['Date'], axis=1)
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


# If store is closed, sales will be always zero.
# Unless we have not found any other utilization of this information,
# I will drop all rows using mask train['Open'] == 0 and 'Open' column
# Also we should add to preprocess of test data same logic,
# for mask test['Open'] == 0 test['Sales'] = 0
#
# Do we need assume 0 for this
# column Open in test data file for store 622 is blank
# Is the store closed meaning open=0? is it the right assumption ID=480 1336 2192 3048 4760 5616 6472 7328 8184 9040 10752

def drop_closed_days_rows(t):
    t = t[t['Open'] != 0]
    t = t.drop(['Open'], axis=1)
    return t


def read_train_df():
    t = pd.read_csv("data/train.csv", low_memory=False)
    t = drop_closed_days_rows(t)
    t = process_date(t)
    t = data_cleaning(t)
    # t['Date'] = (pd.to_datetime('2015-08-01') - t['Date']).astype('timedelta64[D]')
    return t


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


def read_store_df():
    df = pd.read_csv("data/store.csv")
    df.loc[df['CompetitionDistance'].isnull(), 'CompetitionDistance'] = df['CompetitionDistance'].mean().round()
    df.loc[814, 'CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].mean().round()
    df['StoreType'] = df['StoreType'].apply(categorical_value_handler)
    df['Assortment'] = df['Assortment'].apply(categorical_value_handler)
    return df


def new_merge_df(df):
    tl = []
    df_columns = list(df.columns)
    yi = df_columns.index('Year')
    wi = df_columns.index('Week')
    cm = df_columns.index('CompetitionOpenSinceMonth')
    cy = df_columns.index('CompetitionOpenSinceYear')
    p2ind = df_columns.index('Promo2')
    st = df_columns.index('Store')
    dw = df_columns.index('DayOfWeek')
    pr = df_columns.index('Promo')
    cd = df_columns.index('CompetitionDistance')

    for row in df.itertuples():
        row = list(row)[1:]
        year, month, week = row[yi:wi+1]
        comp_since_month, comp_since_year = row[cm:cy+1]
        p2, p2sw, p2sy, p2i = row[p2ind:]
        line = row[st:cd+1] # compdist last val
        if np.isnan(comp_since_month): #
             has_comp, comp_months = -999999, -999999 #has no competitor #has_comp, comp_months
        else:
            comp_months = (year - comp_since_year) * 12 + month - comp_since_month
            if comp_months <= 0: # competitor has not opened yet
                has_comp, comp_months = -1, -1
            else:
                has_comp = 1
        line += [has_comp, comp_months]
        promo2 = 0
        if not p2:
            promo2 = -999999
        else:
            if p2sy < year:
                if month in [str_month_to_int(m) for m in p2i.split(',')]:
                    promo2 = 1
            elif comp_since_year == year:
                if p2sw <= week:
                    if month in [str_month_to_int(m) for m in p2i.split(',')]:
                        promo2 = 1
        line += [promo2]
        tl.append(line)

    newcols = list(df.columns[:-6]) + ['HasCompetitor', 'CompetingMonths', 'Promo2']
    ndf = pd.DataFrame(tl, columns = newcols)
    return ndf


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

            t.loc[mask, 'CompetingMonths'] = t.loc[mask, ['Year', 'Month']].apply(
                    lambda r: (r['Year'] - comp_year) * 12 + r['Month'] - comp_month, axis=1)

            if comp_year < 2013:
                maskc = mask
            else:
                mask1c = (mask) & (t['Year'] == comp_year) & (t['Month'] >= comp_month)
                mask2c = (mask) & (t['Year'] > comp_year)
                maskc = mask1c | mask2c

            t.loc[maskc, 'HasCompetitor'] = 1
        else:
            t.loc[mask, 'HasCompetitor'] = -99999999

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


def read_test_df():
    t = pd.read_csv("data/test.csv")
    t = process_date(t)
    t = data_cleaning(t)
    t.loc[t['Open'].isnull(), 'Open'] = 1
    return t


train = read_train_df()
store = read_store_df()

train['Promo2'] = 0
train['StoreType'] = 0
train['Assortment'] = 0
train['CompetitionDistance'] = 0
train['HasCompetitor'] = 0
train['CompetingMonths'] = -99999999

train = merge_df(train, store)
train.loc[train['CompetingMonths'] < 0, 'CompetingMonths'] = 0


store_sales = pd.DataFrame()
store_sales = train[['Sales', 'Store']].groupby('Store').mean()
store_sales.reset_index(inplace=True)
store_sales.columns = ['Store', 'Mean_Sales']
train = train.merge(store_sales, how='left', on='Store')

week_sales = pd.DataFrame()
week_sales = train[['DayOfWeek', 'Sales']].groupby(['DayOfWeek']).mean()
week_sales.reset_index(inplace=True)
week_sales.columns = ['DayOfWeek', 'DOW_Sales']
train = train.merge(week_sales, how='left', on=['DayOfWeek'])


col_x = np.delete(train.columns, [2, 3])  # 0: Store 2 : Sales, 3 : Customers, 10 : NWeek
train[col_x].to_csv('pickle_cellar/train_full_x_1.csv', index=False)
# train['Sales'].to_csv('pickle_cellar/train_full_y.csv', index=False)

#
# test = read_test_df()
#
# test['Promo2'] = 0
# test['StoreType'] = 0
# test['Assortment'] = 0
# test['CompetitionDistance'] = 0
# test['HasCompetitor'] = -1
# test['CompetingMonths'] = 0
#
# test = merge_df(test, store)
# test.loc[test['CompetingMonths'] < 0, 'CompetingMonths'] = 0
# test.loc[test['Open'].isnull(), 'Open'] = 1
#
# test = test.merge(store_sales, how='left', on='Store')
# test = test.merge(week_sales, how='left', on=['DayOfWeek'])
#
# col_test = np.delete(test.columns, [0, 3])
# test[col_test].to_csv('pickle_cellar/test_data.csv', index=False)
#
