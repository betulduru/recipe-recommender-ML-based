# Recipe Recommender v2 

A content based ML recipe recommendation system built with Python.

## How It Works

Each recipe is represented as a nutrition vector (calories, protein, fat, carbs).
Euclidean distance is used to find similar recipes.
A user profile is built by averaging liked recipes, recommendations are ranked by proximity to that profile.

## Stack
Python, pandas, scikit-learn, Spoonacular API

## Run

    python3 fetch_recipes.py       : collect data
    python3 recommender.py         : recipe to recipe similarity
    python3 user_profile.py        : personalized recommendations