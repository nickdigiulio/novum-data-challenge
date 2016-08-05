import pandas as pd
import string
from matplotlib import pyplot as plt
from matplotlib import mlab
import re
import requests
import json


data = pd.read_table('./data/u.data', names=['UserId', 'ItemId', 'Rating', \
                                             'Timestamp'], index_col=[0,1])
item = pd.read_table('./data/u.item', sep='|', names=['ItemId', 'Title', \
                                        'ReleaseDt', 'VideoReleaseDt', 'Url',\
                                        'Unknown', 'Action', 'Adventure', \
                                        'Animation', 'Children', 'Comedy', \
                                        'Crime', 'Documentary', 'Drama', \
                                        'Fantasy', 'Noir', 'Horror', 'Musical',\
                                        'Mystery', 'Romance', 'Sci-Fi', \
                                        'Thriller', 'War', 'Western'], index_col=0)
user = pd.read_table('./data/u.user', sep='|', names=['UserId', 'Age', 'Gender'\
                                            ,'Occupation', 'Zip'], index_col=0)

score = data.groupby(level=1).mean().drop(['Timestamp'], axis=1).sort('Rating')
score.head()
