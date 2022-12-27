from fastapi import FastAPI
from model import hybrid_content_svd_model
from model import get_recommendations_based_on_genres
from model import get_recommendation_content_model
from pydantic import BaseModel
app = FastAPI()

class data(BaseModel):
    userId : int
    moviename: str
@app.get('/')
def index():
    return {'message': 'Hello, World'}

   
@app.post('/recommend_hybrid')
async def recommend(user_input : data):
    
    # model integration
    movies = hybrid_content_svd_model(user_input.userId)
    
    return movies

@app.post('/recommend_movies_content')
async def recommend(user_input : data):
    
    # model integration
    movies = get_recommendations_based_on_genres (user_input.moviename)
    
    return movies

@app.post('/recommend_content')
async def recommend(user_input : data):
    
    # model integration
    movies = get_recommendation_content_model(user_input.userId)
    
    return movies