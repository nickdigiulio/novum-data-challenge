import pandas as pd
import string
from matplotlib import pyplot as plt
from matplotlib import mlab
from math import sqrt
import re
import requests
import json
import tqdm

def urlParse(url):
    """
    Parse Url to retrieve properly formatted Title and Year
    """
    p1 = re.compile('http://us\.imdb\.com/M/title-exact\?([^\(]*).*?(\d{4})')
    t, year = re.match(p1, url).groups()
    p2 = re.compile('%20')
    title = re.sub(p2, ' ', t)
    return title, year
    
def titleParse(title):
    """
    Parse title to retrieve properly formated Title and Year
    """
    p = re.compile('^([^,(]*)([^(]*)?\s?\(?(.*)?\)?\s?\((\d{4})\)')
    title,s1,s2,year = re.match(p, title).groups()
    if len(s1) > 2:
        title = s1[2:]+' '+title
    return title.strip(), year

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
        genre_list[i-1] = g
        #genre_num = [i+1 for i in xrange(len(genre_list))]
    df['Genre'] = pd.Series(genre_list)
    #df['Genre#'] = pd.Series(genre_num)
    return df   

def confidence(x):
    fresh, rotten = x[0],x[1]
    n = fresh+rotten
    if n == 0:
        return 0
    z = 1.5
    phat = float(fresh) / n
    return 5*((phat + z*z/(2*n) - z * sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n))
    
def bayes(x):
    R,C,v = x[0], x[1], x[2]
    m = 10
    w = float(v)/(v+m)
    return w*R + (1-w)*C

data = pd.read_table('./data/u.data', names=['UserId', 'ItemId', 'Rating', 'Timestamp'], index_col=[0,1])
item = pd.read_table('./data/u.item', sep='|', names=['ItemId', 'Title', 'ReleaseDt', 'VideoReleaseDt', 'Url', \
                                                      'Unknown', 'Action', 'Adventure', 'Animation', 'Children',\
                                                      'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Noir',\
                                                      'Horror', 'Musical','Mystery', 'Romance', 'Sci-Fi', 'Thriller',\
                                                      'War', 'Western'], index_col=0)
user = pd.read_table('./data/u.user', sep='|', names=['UserId', 'Age', 'Gender','Occupation', 'Zip'], index_col=0)


vals = pd.DataFrame()
vals['Fresh'] = data['Rating'].apply(lambda x: x > 3)
vals['Rotten'] = data['Rating'].apply(lambda x: x < 3)
vals['1'] = data['Rating'].apply(lambda x: x == 1)
vals['2'] = data['Rating'].apply(lambda x: x == 2)
vals['3'] = data['Rating'].apply(lambda x: x == 3)
vals['4'] = data['Rating'].apply(lambda x: x == 4)
vals['5'] = data['Rating'].apply(lambda x: x == 5)
vals = vals.groupby(level=1).sum()

item = genre(item)

item['temp'] = item['Title'].loc[item['Title']!= 'unknown'].apply(titleParse)
item[['Title', 'Year']] = item['temp'].apply(pd.Series)

score = pd.merge(item, data.groupby(level=1).agg({'Rating':{'Average':'mean', 'Total':'count'}}), left_index=True, right_index=True) #.sort_values(by = 'Rating', ascending=False)
score = pd.merge(score, vals, left_index=True, right_index=True)
score = score.rename(index=str, columns={(u'Rating', u'Average'):'Average',(u'Rating', u'Total'):'Ratings',(u'Fresh', u'Fresh'):'Fresh',(u'Rotten', u'Rotten'):'Rotten'})
score[['Title', 'Average', 'Ratings', 'Fresh', 'Rotten']].sort_values(by=['Average', 'Ratings'], ascending=False).head()

genre_average = pd.merge(item, data, left_index=True, right_index=True).groupby('Genre').mean()['Rating']
score = pd.merge(score, pd.DataFrame(genre_average), left_on='Genre', right_index=True)

score['Confidence'] = score[['Fresh', 'Rotten']].apply(confidence, axis = 1)

'''
This takes forever to run (2-3 minutes)
I've exported the results to a csv so it can be imported each time 
rather than having to grab the data from OMDB each time
'''
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
runtimes = pd.read_table('./data/u.runtime', sep='|', names = ['ItemId', 'Runtime'], index_col=0)

score = pd.merge(score, runtimes, how='left', left_index=True, right_index=True)

genres = item['Genre'].unique()
#for n, g in enumerate(genres):
#    plt.subplot(3, 6, n+1)
#    plt.scatter(score['Year'].loc[score['Genre']==g], score['Average'].loc[score['Genre']==g])
#    plt.title(g)
    


    
score['CredInt'] = score[['Average', 'Rating', 'Ratings']].apply(bayes, axis=1)
def polarizing(x):
    fresh, rotten, n = x[0], x[1], x[2]
    
    return n/max(abs(fresh-rotten),0.75)

def polarizing2(x):
    n1,n2,n3,n4,n5 = x[0], x[1], x[2], x[3], x[4]
    n = n1 + n2 + n4 + n5
    d = max(abs(n1-n5) + 2*abs(n2-n4) + 3*n3, 0.75)
    return n/d

score['Polarizing'] = score[['Fresh', 'Rotten', 'Ratings']].apply(polarizing, axis=1)
score['Polarizing2'] = score[['1', '2', '3', '4', '5']].apply(polarizing2, axis=1)    
    