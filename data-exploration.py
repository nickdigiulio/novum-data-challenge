import pandas as pd
from ranking import *
from functions import *

###############################################################################
#                               Read datasets                                 #
###############################################################################
data = pd.read_table('./data/u.data', names=['UserId', 'ItemId', 'Rating', 'Timestamp'], index_col=[0,1])
item = pd.read_table('./data/u.item', sep='|', names=['ItemId', 'Title', 'ReleaseDt', 'VideoReleaseDt', 'Url', \
                                                      'Unknown', 'Action', 'Adventure', 'Animation', 'Children',\
                                                      'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Noir',\
                                                      'Horror', 'Musical','Mystery', 'Romance', 'Sci-Fi', 'Thriller',\
                                                      'War', 'Western'], index_col=0)

###############################################################################
#                   Create Alternate Rating Values                            #
###############################################################################

vals = pd.DataFrame()
# Thumbs Up/Down ranking; unfortunately ratings of 3 must be ignored for this type of rating
vals['Fresh'] = data['Rating'].apply(lambda x: x > 3)
vals['Rotten'] = data['Rating'].apply(lambda x: x < 3)

# Create buckets for ratings 1-5
# This will be easier to handle for some ranking systems
for i in xrange(1,6):
    vals[str(i)] = data['Rating'].apply(lambda x: x == i)
vals = vals.groupby(level=1).sum()

###############################################################################
#                            Transform item data                              #
###############################################################################
# call genre function on item
item = genre(item)
item['temp'] = item['Title'].loc[item['Title']!= 'unknown'].apply(titleParse)
item[['Title', 'Year']] = item['temp'].apply(pd.Series)
item.drop('temp', axis = 1, inplace=True)

score = pd.merge(item, data.groupby(level=1).agg({'Rating':{'Average':'mean', 'Total':'count'}}), left_index=True, right_index=True) #.sort_values(by = 'Rating', ascending=False)
score = pd.merge(score, vals, left_index=True, right_index=True)
score = score.rename(index=str, columns={(u'Rating', u'Average'):'Average',(u'Rating', u'Total'):'Ratings',(u'Fresh', u'Fresh'):'Fresh',(u'Rotten', u'Rotten'):'Rotten'})
score[['Title', 'Average', 'Ratings', 'Fresh', 'Rotten']].sort_values(by=['Average', 'Ratings'], ascending=False).head()

genre_average = pd.merge(item, data, left_index=True, right_index=True).groupby('Genre').mean()['Rating']
score = pd.merge(score, pd.DataFrame(genre_average), left_on='Genre', right_index=True)

score = pd.merge(score, runtimes, how='left', left_index=True, right_index=True)
    
score['Confidence'] = score[['Fresh', 'Rotten']].apply(confidence, axis = 1)    
score['CredInt'] = score[['Average', 'Rating', 'Ratings']].apply(bayes, axis=1)
score['Polarizing'] = score[['Fresh', 'Rotten', 'Ratings']].apply(polarizing, axis=1)
score['Polarizing2'] = score[['1', '2', '3', '4', '5']].apply(polarizing2, axis=1)    
    