# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 20:34:44 2024

@author: Dnyaneshwari...
"""

#Recommendation System
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
anime = pd.read_csv("C:/Datasets/anime.csv", encoding='utf8')

# Fill missing values in 'genre' column with 'general'
anime['genre'] = anime['genre'].fillna('general')

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the 'genre' column
tfidf_matrix = tfidf.fit_transform(anime['genre'])

# Compute cosine similarity matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a Series to map anime titles to their corresponding index
anime_index = pd.Series(anime.index, index=anime['name']).drop_duplicates()


def get_recommendations(name, topN):
    # Get the index of the input anime name
    anime_id = anime_index[name]

    # Compute cosine similarity scores for the input anime
    cosine_scores = list(enumerate(cosine_sim_matrix[anime_id]))

    # Sort the cosine similarity scores in descending order
    cosine_scores = sorted(cosine_scores, key=lambda x: x[1], reverse=True)

    # Get the top N similar anime
    top_anime_scores = cosine_scores[1: topN + 1]

    # Get the indices and scores of the top N similar anime
    top_anime_indices = [i[0] for i in top_anime_scores]
    top_anime_scores = [i[1] for i in top_anime_scores]

    # Create a DataFrame to store the top N similar anime
    top_anime_df = pd.DataFrame(columns=['name', 'score'])
    top_anime_df['name'] = anime.iloc[top_anime_indices]['name']
    top_anime_df['score'] = top_anime_scores

    return top_anime_df

# Test the function
recommendations = get_recommendations('Bad Boys (1995)', topN=10)
print(recommendations)

'''
Output
                                            Name                score
10919                              No Game No Life Movie  1.000000
10436  Super Real Mahjong: Mahjong Battle Scramble - ...  0.859206
4290                        Raising Victor Vargas (2002)  0.827579
5882                      xXx: State of the Union (2005)  0.800258
5968            Pusher II: With Blood on My Hands (2004)  0.800258
6116                                   Revolution (1985)  0.800258
6677              World on a Wire (Welt am Draht) (1973)  0.800258
10435  Super Real Mahjong: Kasumi Miki Shouko no Haji...  0.800258
4628                             Italian Job, The (1969)  0.787476
6812                     Midnight Meat Train, The (2008)  0.739464
'''

'''
Accuracy, as a metric, is typically used in classification tasks where the 
prediction is compared directly with ground truth labels. In the context of 
recommendation systems, there's no direct ground truth label for a user-item 
interaction, which makes it difficult to define accuracy in a meaningful way.

Recommendation systems aim to predict user preferences or item relevance 
rather than binary outcomes. Therefore, evaluating a recommendation system 
solely based on accuracy might not be appropriate or informative.
'''