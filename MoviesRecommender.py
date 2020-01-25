#!/usr/bin/env python
# coding: utf-8

# # # Item-Based CF
# # If you want to add your own user_id, open 'u.data' file and add your own user as user_id = 0, with your own movie ratings


import numpy as np
import pandas as pd
import os
from pathlib import Path
ratings_file = os.path.join(Path().absolute(), "ml-100k/u.data")
movies_file = os.path.join(Path().absolute(), "ml-100k/u.item")

rating_cols = ["user_id","movie_id","rating"]
ratings = pd.read_csv(ratings_file, sep='\t', names=rating_cols, usecols=range(3),encoding='latin-1')

movie_cols = ['movie_id','title']
movies = pd.read_csv(movies_file, sep='|', names=movie_cols, usecols=range(2),encoding='latin-1')

ratings = pd.merge(movies,ratings)


# In[ ]:


userRatings = ratings.pivot_table(index=['user_id'], columns=['title'],values='rating')
correlationTable = userRatings.corr(method='pearson', min_periods=100)


# In[ ]:


#user variable should be an integer of the number of user_id that you are using
def recommendMovies(user):
    myRatings = userRatings.loc[user].dropna().sort_values(ascending=False)
    
    similarCandidates = pd.Series()
    for i in range(0, len(myRatings.index)):
        print ("Adding similarities for " + myRatings.index[i] + "...")

        #Retrieving similar movies to the one that is rated
        sims = correlationTable[myRatings.index[i]].dropna()

        #Scale the similaritiy by how high the rating of this movie was
        #This will give more strength to the movies rated 5, and less strength
        #to the movies rated 1
        sims = sims.map(lambda x: x*myRatings[i])
        #series .map(function) , replaces each value in the Series with the value output of the function inside map


        #Add the score "sims" to the list of similar candidates of the movie
        similarCandidates = similarCandidates.append(sims)

        
    print ("\n\nSorting...\n")
    similarCandidates = similarCandidates.sort_values(ascending=False)
    
    #We use groupby() to add together the scores from movies 
    similarCandidates = similarCandidates.groupby(similarCandidates.index).sum()
    similarCandidates = similarCandidates.sort_values(ascending=False)
    
    
    #A for loop that checks if the recommended movie has already been watched and rated, and removes it if it is.
    for x in myRatings.index:
        if x in similarCandidates:
            similarCandidates = similarCandidates.drop(x)

        else:
            pass
        
        
    print("Your Top 15 Recommended Movies Are: ")
    return similarCandidates


# In[ ]:


#Change the parameter value "8" to any user_id you want to check the recommended Movies for that user.


recommendMovies(8).head(15)


# In[ ]:




