Overview
I was given the task of analyzing the Movie Lens 100k (ml-100k) dataset to produce interesting insight about the data.  I created 5 different ranking algorithms and analyzed the results of each and I created a Random Forest Classifier to predict a user’s rating of a given film from an out of sample population.
Analysis
Comparison of Ranking Algorithms
I compared 5 different ranking algorithms – 3 to sort of the best movie, and 2 to sort of the most polarizing. 
The main script for this section is called ranking-comparison.py.  It utilized the rankings.py and functions.py, all of which are located in the root of this repo.

The results of this section are explained in detail in the first section of the Novum Data Challege.docx document.
 
Random Forest Classifier to Predict Film Rankings
I created a Random Forest Classifier model using Scikit Learn to predict the rating that a user would give a movie based on a set of features which describe the user and the film. 

The main script for this section is titled rfc-ml.py. It utilizes the functions.py script.

The results of this section are explained in detail in the second section of the Novum Data Challege.docx document.