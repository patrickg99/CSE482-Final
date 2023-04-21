#!/usr/bin/env python
# coding: utf-8


import streamlit as st
import pandas as pd
import pandasql as ps
import numpy as np
import pymysql
import json
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import collections 
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


#  Lines for autocomplete in selectbox
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')
titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])

#-------------------------------------------------------------------------------------------------------------------------------------------

#  Titles/Front page
st.markdown("<h1 style='text-align: center; color: white;'>Movie Recommender</h1>", unsafe_allow_html=True)
with st.columns(6)[1]:
    image = Image.open('title_image.jpg')
    st.image(image, use_column_width = False)
    
st.markdown("<h5 style='text-align: center; color: white;'>By: Nick White, Stephen Lee, Chaeyeon Yim, \
              Mark Carravallah, Patrick Govan</h5>", unsafe_allow_html=True)

#-------------------------------------------------------------------------------------------------------------------------------------------
st.divider()

#  Setting up columns/inputs
col1, col2, col3 = st.columns(3, gap = 'large')

with col1:
    st.subheader('Chose A Movie')
    movie = st.selectbox('Please type a movie',
    (titles))

with col2:
    st.subheader('Select if you want the recommendation based on the movie, the genre, or both.')
    method = st.selectbox('Please Select One',
    ('Movie','Genre', 'Both'))
    
with col3:
    st.subheader('Chose Any Genre (One or Two)')
    genre_sel = st.multiselect('Please Select a genre',
    ['Adventure', 'Comedy', 'Action', 'Drama', 'Crime', 'Children',
       'Mystery', 'Documentary', 'Animation', 'Thriller', 'Horror',
       'Fantasy', 'Western', 'Film-Noir', 'Romance', 'War', 'Sci-Fi',
       'Musical', 'IMAX'],
    ['Comedy'])
    
    
st.divider()

#-------------------------------------------------------------------------------------------------------------------------------------------
col11, col12, col13 = st.columns(3)

with col12:
    with st.columns(3)[1]:
        if st.button('Execute Query'):
            a = 1
        else:
            a = 0
with col11:
    if a == 1:
        if method == 'Both':
            st.text('Recommended Movies based on your choice:')
            
            #  Content based recommendation algorithm 
            
            tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
            tfidf_matrix = tf.fit_transform(movies['genres'])
            cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
            
            def recommendations(title):
                idx = indices[title]
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:21]
                movie_indices = [i[0] for i in sim_scores]
                return titles.iloc[movie_indices]
            rec = recommendations(movie)
            st.dataframe(rec)
                
with col13:
    if a == 1:
        if method == 'Both':
            st.text('Based on your genre selection, you may be interested in:')
            outercount = 0
            innercount = 0

            temp = movies.iloc[:,3:].values
            temp = temp.tolist()

            arr = [[] for i in range(len(temp))]
            for innerarray in temp:
                innercount = 0
                for var in innerarray:
                    if (var != None): 
                        arr[outercount].append(var)
                        innercount += 1
                outercount += 1
            
            counts = collections.defaultdict(int)
            for collab in arr:
                collab.sort()
                for pair in itertools.combinations(collab, 2):
                    counts[pair] += 1

            topresults = sorted(counts.items(),  key=lambda x:x[1], reverse=True)
            #st.write(topresults)
            
            # Query for top 10 rated movied for specified genre
            def searchtop10(genre):
                
                # Merge the movies and ratings tables
                merged_df = pd.merge(left=movies, right=ratings, how='inner', on='movieId')

                # Query for top10results
                sqlinput = "genres LIKE '%" + genre + "%'" # user input
                query = '''
                    SELECT 
                        DISTINCT m.title
                    FROM 
                        merged_df m 
                    WHERE 
                        {}
                    ORDER BY 
                        rating DESC 
                    LIMIT 10
                '''.format(sqlinput)
                results = ps.sqldf(query, locals())

                return results
            # Create list of top combinations
            filtered_topresults = {
                "Comedy": "Drama",
                "Drama": "Romance",
                "Action": "Thriller",
                "Crime": "Drama",
                "Horror": "Thriller",
                "Adventure": "Comedy",
                "Children": "Comedy",
                "Mystery": "Thriller",
                "Animation": "Children",
                "Sci-Fi": "Thriller",
                "Fantasy": "Romance",
                "Musical": "Romance",
                "IMAX": "Sci-Fi",
                "Film-Noir": "Thriller",
                "Documentary": "Drama",
                "War": "Western",
            }

            # Find most movies the user may also like based on genre
            selected_genres = (genre_sel)
            you_may_also_like = []

            # Loop through selected genres and search for top 10 associated movies by genre
            for genre in selected_genres:
                similar_genre = filtered_topresults[genre]
                if (similar_genre not in selected_genres):

                    # Search for top 10
                    top10movies = searchtop10(genre) 

                    # Add top 10 to list of 'you may also like'
                    for i in range(10):
                        you_may_also_like.append(top10movies['title'][i])

            # Remove duplicates from list
            you_may_also_like = list(set(you_may_also_like))
            genre_df = pd.DataFrame({'Movies': you_may_also_like})
            # Print results
            st.dataframe(genre_df)
                    
                    

#-------------------------------------------------------------------------------------------------------------------------------------------
with col11:
    if a == 1:
        if method == 'Movie':
            st.text('Recommended Movies based on your choice:')
            
            #  Content based recommendation algorithm 
            
            tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
            tfidf_matrix = tf.fit_transform(movies['genres'])
            cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
            
            def recommendations(title):
                idx = indices[title]
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:21]
                movie_indices = [i[0] for i in sim_scores]
                return titles.iloc[movie_indices]
            rec = recommendations(movie)
            st.dataframe(rec)
            
#---------------------------------------------------------------------------------------------------------------------------------------    
with col13:
    if a == 1:
        if method == 'Genre':
            st.text('Based on your genre selection, you may be interested in:')
            outercount = 0
            innercount = 0

            temp = movies.iloc[:,3:].values
            temp = temp.tolist()

            arr = [[] for i in range(len(temp))]
            for innerarray in temp:
                innercount = 0
                for var in innerarray:
                    if (var != None): 
                        arr[outercount].append(var)
                        innercount += 1
                outercount += 1
            
            counts = collections.defaultdict(int)
            for collab in arr:
                collab.sort()
                for pair in itertools.combinations(collab, 2):
                    counts[pair] += 1

            topresults = sorted(counts.items(),  key=lambda x:x[1], reverse=True)
            #st.write(topresults)
            
            # Query for top 10 rated movied for specified genre
            def searchtop10(genre):
                
                # Merge the movies and ratings tables
                merged_df = pd.merge(left=movies, right=ratings, how='inner', on='movieId')

                # Query for top10results
                sqlinput = "genres LIKE '%" + genre + "%'" # user input
                query = '''
                    SELECT 
                        DISTINCT m.title
                    FROM 
                        merged_df m 
                    WHERE 
                        {}
                    ORDER BY 
                        rating DESC 
                    LIMIT 10
                '''.format(sqlinput)
                results = ps.sqldf(query, locals())

                return results
            # Create list of top combinations
            filtered_topresults = {
                "Comedy": "Drama",
                "Drama": "Romance",
                "Action": "Thriller",
                "Crime": "Drama",
                "Horror": "Thriller",
                "Adventure": "Comedy",
                "Children": "Comedy",
                "Mystery": "Thriller",
                "Animation": "Children",
                "Sci-Fi": "Thriller",
                "Fantasy": "Romance",
                "Musical": "Romance",
                "IMAX": "Sci-Fi",
                "Film-Noir": "Thriller",
                "Documentary": "Drama",
                "War": "Western",
            }

            # Find most movies the user may also like based on genre
            selected_genres = (genre_sel)
            you_may_also_like = []

            # Loop through selected genres and search for top 10 associated movies by genre
            for genre in selected_genres:
                similar_genre = filtered_topresults[genre]
                if (similar_genre not in selected_genres):

                    # Search for top 10
                    top10movies = searchtop10(genre) 

                    # Add top 10 to list of 'you may also like'
                    for i in range(10):
                        you_may_also_like.append(top10movies['title'][i])

            # Remove duplicates from list
            you_may_also_like = list(set(you_may_also_like))
            genre_df = pd.DataFrame({'Movies': you_may_also_like})
            # Print results
            st.dataframe(genre_df)
            
#-------------------------------------------------------------------------------------------------------------------------------------------