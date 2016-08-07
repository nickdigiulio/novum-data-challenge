import re
import pandas as pd
import requests
import json

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
    Parse Title to retrieve properly formated Title and Year
    """
    p = re.compile('^([^,(]*)([^(]*)?\s?\(?(.*)?\)?\s?\((\d{4})\)')
    title,s1,s2,year = re.match(p, title).groups()
    if len(s1) > 2:
        title = s1[2:]+' '+title
    return title.strip(), year
    
def getRuntime(title):
        '''
        IMDB url is parsed to retrieve Title and Year of release for film
        '''
        title, year = title
        url = 'http://omdbapi.com/?'
        #title, year = titleParse(url)
        params = {'t':title, 'y':year}
        try:
            resp = requests.get(url, params=params)
        except e:
           raise Exception('Request returned status'.format(s))
        dat = json.loads(resp.text)
        return int(dat['Runtime'][:-4])

def genre(df, output='text', drop_cols=True):
    '''
    The 19 genre fields in the u.item data is converted to a single Genre field.
    Also removes unique genre combinations to reduce the total number of genre
    groups in the data.
    '''
    genre_list = [0]*len(df.index.values)
    genres = ['Unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Noir',\
                      'Horror', 'Musical','Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    for i in df.index.values:
        g = ()
        for gen in genres:
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
    
    if output == 'num':
        genre_num = [i+1 for i in xrange(len(genre_list))]
        df['Genre#'] = pd.Series(genre_num)
    elif output == 'both':
        genre_num = [i+1 for i in xrange(len(genre_list))]
        df['Genre#'] = pd.Series(genre_num)
        df['Genre'] = pd.Series(genre_list)
    else:
        df['Genre'] = pd.Series(genre_list)
    
    if drop_cols:
        df.drop(genres, axis=1, inplace=True)
    return df 
    
def split_features_targets(dat, item, user):
    '''
    Input u.base dataset and output:
        1. featues as dataframe
        2. targets as series
    '''
    Y = dat['Rating'] #.apply(lambda x: 1 if x > 2.5 else 0)
    X = pd.merge(pd.merge(dat, item, on='itemId'), user, on='UserId')
    X['Year'].loc[X['Year'].isnull()] = 0
    X = X.drop(['UserId', 'itemId', 'Rating', 'ReleaseDt', 'Timestamp', 'Title', 'VideoReleaseDt', 'Url', 'Zip'], axis=1)
    return X, Y

def gender(x):
    if x == 'M':
        return 0
    else:
        return 1