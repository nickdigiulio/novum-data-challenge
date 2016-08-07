from math import sqrt

def confidence(x):
    '''
    Wilson Score
    found here: http://www.evanmiller.org/how-not-to-sort-by-average-rating.html
    
    95% confidence interval using Fresh/Rotten ratings (Fresh == Rating > 3, Rotten == Rating < 3)
    '''
    fresh, rotten = x[0],x[1]
    n = fresh+rotten
    if n == 0:
        return 0
    z = 1.96
    phat = float(fresh) / n
    return 5*((phat + z*z/(2*n) - z * sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n))
    
def bayes(x):
    '''
    Baysian Credible Interval
    
    This uses the average rating for the given film's genre as a prior estimate
    for 
    '''
    R,C,v = x[0], x[1], x[2]
    m = 10
    w = float(v)/(v+m)
    return w*R + (1-w)*C

def polarizing(x):
    '''
    Polarization Value
    
    This will give a ranking based on one over the difference between the 
    number of likes and dislikes, weighted by the total number of reviews.
    '''
    fresh, rotten, n = x[0], x[1], x[2]
    
    return n/max(abs(fresh-rotten),0.75)

def polarizing2(x):
    '''
    Polarization Value #2
    
    This algorith ranks polarization by looking at the difference of 1star-5star 
    and 2star-4star reviews and weights the values such that a film with more 1 
    and 5 star reviews will ranked higher than a film with more 2 and 4 star 
    reviews.
    '''
    n1,n2,n3,n4,n5 = x[0], x[1], x[2], x[3], x[4]
    n = n1 + n2 + n4 + n5
    d = max(abs(n1-n5) + 2*abs(n2-n4) + 3*n3, 0.75)
    return n/d