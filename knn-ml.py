import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def featurize(dat):
    '''
    Input u.base dataset and output:
        1. featues as dataframe
        2. targets as series
    '''
    Y = dat['Rating'] #.apply(lambda x: 1 if x > 2.5 else 0)
    X = pd.merge(pd.merge(dat, ITEM, on='ItemId'), USER, on='UserId')
    X['Year'].loc[X['Year'].isnull()] = 0
    X = X.drop(['UserId', 'ItemId', 'Rating', 'ReleaseDt', 'Timestamp', 'Title', 'VideoReleaseDt', 'Url', 'Unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Noir',\
                      'Horror', 'Musical','Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'Occupation', 'Zip'], axis=1)
    return X, Y

def gender(x):
    if x == 'M':
        return 0
    else:
        return 1

def genre2(df):
    genre_list = [0]*len(df.index.values)
    for i in df.index.values:
        g = ()
        for gen in ['Unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Noir',\
                      'Horror', 'Musical','Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']:
            if df.loc[i, gen] == 1:
                g += (gen,)
    df['Genre'] = pd.Series(genre_list)
    return df

def genre(df):
    genre_list = [0]*len(df.index.values)
    
    for i in df.index.values:
        g = ()
        for gen in ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Noir',\
                      'Horror', 'Musical','Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']:
            if df.loc[i, gen] == 1:
                g += (gen,)
        if 'Documentary' in g:
            g = 'Documentary'
        elif 'Western' in g:
            g = 'Western'
        elif 'War' in g:
            g = 'War'
        elif 'Crime' in g:
            g = 'Crime'
        elif 'Children' in g:
            g = 'Children'
        elif 'Fantasy' in g:
            g = 'Fantasy'
        elif len(g) > 1:
            if "Comedy" in g and "Drama" in g:
                g = "Comedy Drama"
            elif "Action" in g and "Adventure" in g:
                g = "Action Adventure"
            elif "Thriller" in g:
                g = "Thriller"
            elif "Action" in g and "Comedy" in g:
                g = "Action Comedy"
            elif "Animation" in g:
                g = "Animation"
            elif "Romance" in g:
                g = "Romance"
            else:
                g = " ".join(i for i in g)
        elif len(g) == 1:
            g = g[0]
        genre_list[i] = g
        genre_num = [i+1 for i in xrange(len(genre_list))]
    #df['Genre'] = pd.Series(genre_list)
    df['Genre#'] = pd.Series(genre_num)
    return df
    
    
def titleParse(title):
    """
    Parse title to retrieve properly formated Title and Year
    """
    p = re.compile('^([^,(]*)([^(]*)?\s?\(?(.*)?\)?\s?\((\d{4})\)')
    title,s1,s2,year = re.match(p, title).groups()
    if len(s1) > 2:
        title = s1[2:]+' '+title
    return title.strip(), year

ITEM = pd.read_table('./data/u.item', sep='|',
                    names=['ItemId', 'Title', 'ReleaseDt', 'VideoReleaseDt', 'Url', \
                                                      'Unknown', 'Action', 'Adventure', 'Animation', 'Children',\
                                                      'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Noir',\
                                                      'Horror', 'Musical','Mystery', 'Romance', 'Sci-Fi', 'Thriller',\
                                                      'War', 'Western'], index_col=0)
USER = pd.read_table('./data/u.user', sep='|', names=['UserId', 'Age', 'Gender','Occupation', 'Zip'])
RUNTIME = pd.read_table('./data/u.runtime', sep='|', names = ['ItemId', 'Runtime'], header=0, index_col=0)
OCCUPATION = pd.read_table('./data/u.occupation', sep='|', names=['Occupation']).reset_index()
OCCUPATION = OCCUPATION.rename(columns = {'index':'Occupation#'})

USER = pd.merge(USER, OCCUPATION, on='Occupation')
USER['Gender'] = USER['Gender'].apply(gender)

# Add runtime data to item dataset
ITEM = pd.merge(ITEM, RUNTIME, how='left', left_index=True, right_index=True).reset_index()
ITEM = genre(ITEM)
ITEM['temp'] = ITEM['Title'].loc[ITEM['Title']!= 'unknown'].apply(titleParse)
ITEM[['Title', 'Year']] = ITEM['temp'].apply(pd.Series)
ITEM.drop('temp', axis=1, inplace=True)

train_data, test_data = [], []
for i in range(1, 5):
    train_data.append(pd.read_table('./data/u{}.base'.format(i), names=['UserId', 'ItemId', 'Rating', 'Timestamp']))
    test_data.append(pd.read_table('./data/u{}.test'.format(i), names=['UserId', 'ItemId', 'Rating', 'Timestamp']))

train_data = pd.concat(train_data)
test_data = pd.concat(test_data)

x_train, y_train = featurize(train_data)
x_test, y_test = featurize(test_data)
y_test = np.array(y_test)

#knn = KNeighborsClassifier()
#knn.fit(x_train, y_train)

#y_predict = knn.predict(x_test)

rfc = RandomForestClassifier(n_estimators = 30)
rfc.fit(x_train, y_train)

#print rfc.get_params()

#print rfc.score(x_test, y_test)
y_predict = rfc.predict(x_test)

def count_vals(y):
    vals = sorted(list(set(y)))
    c = [0]*len(vals)
    for i in y:
        c[i-1]+=1
    return vals, c

error = 0
for i in xrange(len(y_test)):
    error += abs(y_test[i] - y_predict[i])

print "MSE: {}".format(float(error)/len(y_test))

result = [0, 0, 0, 0, 0]
for i in xrange(len(y_test)):
    error = abs(y_predict[i] - y_test[i])
    if error == 0:
        result[0] += 1
    elif error == 1:
        result[1] += 1
    elif error == 2:
        result[2] += 1
    elif error == 3:
        result[3] += 1
    elif error == 4:
        result[4] += 1

print "# Correct: {0} | Off by 1: {1} | Off by 2: {2} | Off by 3: {3} | Off by 4: {4}".format(*result)

#x1, y1 = count_vals(y_test)
#x2, y2 = count_vals(y_predict)
#plt.figure()
#plt.plot(x1, y1, x2, y2)
#plt.legend(['Real', 'Predicted'], loc=2)
#plt.show()
