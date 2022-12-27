import streamlit as st
import pandas as pd
import requests
from ui_functions import display,display_genre
st.title("""
Movie Recommender""")


userID = st.text_input("Enter user ID")

if userID is not None:
    if st.button('Recommend'):
        movies = display(userID)
        movies = movies.split('"')[1:-1]
        # products = [if product == ""for product in products ]
        i = 1
        for movie in movies:
            if movie == ',' or movie == str(i) or movie == ":":
                i += 1
                continue
            # i += 1
            st.write(movie)
    
movieName= st.text_input("Enter movie name")

if movieName is not None:
    if st.button('RECOMMEND'):
        movies_genre = display_genre(movieName)
        movies_genre = movies_genre.split('"')[1:-1]
        i = 1
        for movie in movies_genre:
            if movie == ',' or movie == str(i) or movie == ":":
                i += 1
                continue
            st.write(movie)