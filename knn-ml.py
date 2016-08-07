import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from functions import *

###############################################################################
#                               Read datasets                                 #
###############################################################################
item = pd.read_table('./data/u.item', sep='|',
                    names=['itemId', 'Title', 'ReleaseDt', 'VideoReleaseDt', 'Url', \
                                                      'Unknown', 'Action', 'Adventure', 'Animation', 'Children',\
                                                      'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Noir',\
                                                      'Horror', 'Musical','Mystery', 'Romance', 'Sci-Fi', 'Thriller',\
                                                      'War', 'Western'], index_col=0)
user = pd.read_table('./data/u.user', sep='|', names=['UserId', 'Age', 'Gender','Occupation', 'Zip'])

###############################################################################
#                         Transform Item Dataset                              #
###############################################################################
# convert genre fields to single genre column
item = genre(item)
# parse title and year from Title column
item['temp'] = item['Title'].loc[item['Title']!= 'unknown'].apply(titleParse)

# get runtime data from OMDB
'''
This takes forever to run (2-3 minutes)
I've exported the results to a csv so it can be imported each time 
rather than having to grab the data from OMDB each time
'''
#import tqdm
#rt = []
#for i in tqdm.tqdm(item['temp']):
#    try:
#        rt.append(getRuntime(i))
#    except:
#        rt.append(-1)
#runtimes = pd.DataFrame(zip(item.index.values, rt))
#runtimes.to_csv('./data/u.runtime', sep='|')
#runtimes.rename(index=str, columns={0: "ItemId", 1: "Runtime"}, inplace=True)
#runtimes.set_index('ItemId', inplace=True)
runtime = pd.read_table('./data/u.runtime', sep='|', header = 0, names = ['ItemId', 'Runtime'], index_col=0)

# Add runtime data to item dataset
item = pd.merge(item, runtime, how='left', left_index=True, right_index=True).reset_index()

# put title and year data in appropriate columns and drop temp field
item[['Title', 'Year']] = item['temp'].apply(pd.Series)
item.drop('temp', axis=1, inplace=True)

###############################################################################
#             Convert Categorical Text Variables to Integer Values            #
###############################################################################
le_gender = LabelEncoder()
le_occ = LabelEncoder()
le_genre = LabelEncoder()

user['Gender'] = le_gender.fit_transform(user['Gender'])
user['Occupation'] = le_occ.fit_transform(user['Occupation'])
item['Genre'] = le_genre.fit_transform(item['Genre'])


train_data, test_data = [], []
for i in range(1, 2):
    train_data.append(pd.read_table('./data/u{}.base'.format(i), names=['UserId', 'itemId', 'Rating', 'Timestamp']))
    test_data.append(pd.read_table('./data/u{}.test'.format(i), names=['UserId', 'itemId', 'Rating', 'Timestamp']))

train_data = pd.concat(train_data)
test_data = pd.concat(test_data)

###############################################################################
#             Split Data into Feature Matrix and Target Array                 #
###############################################################################
x_train, y_train = split_features_targets(train_data, item, user)
x_test, y_test = split_features_targets(test_data, item, user)
y_test = np.array(y_test)


###############################################################################
#                        Initialize and Fit Model                             #
###############################################################################
'''
K-Nearest Neighbors Classifier

Model will try to group the data into discrete groups
'''
#knn = KNeighborsClassifier()
#knn.fit(x_train, y_train)

#y_predict = knn.predict(x_test)

'''
Random Forest Classifer

Uses a series of descision trees to classify inputs.
This model should use the information provided (i.e. viewer's age, gender,
occupation. length of movie) to find patterns which indicate what a given person
would rate a movie with certain characteristics.
'''
rfc = RandomForestClassifier(n_estimators = 30)
rfc.fit(x_train, y_train)

#print rfc.get_params()

#print rfc.score(x_test, y_test)
y_predict = rfc.predict(x_test)

###############################################################################
#                        Measure Error                                        #
###############################################################################
# Mean Squared Error
error = 0
for i in xrange(len(y_test)):
    error += abs(y_test[i] - y_predict[i])**2

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

