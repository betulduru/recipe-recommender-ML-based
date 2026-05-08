import requests
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("SPOONACULAR_API_KEY")

def fetch_recipes(query, number=20):
    url = "https://api.spoonacular.com/recipes/complexSearch"
    params = {
        "apiKey": API_KEY,
        "query": query,
        "number": number,
        "addRecipeNutrition": True,
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data.get("results", [])

def parse_recipes(recipes):
    rows = []
    for r in recipes:
        nutrients = {n["name"]: n["amount"] for n in r.get("nutrition", {}).get("nutrients", [])}
        rows.append({
            "id": r["id"],
            "title": r["title"],
            "calories": nutrients.get("Calories", 0),
            "protein": nutrients.get("Protein", 0),
            "fat": nutrients.get("Fat", 0),
            "carbs": nutrients.get("Carbohydrates", 0),
        })
    return pd.DataFrame(rows)

queries = ["pasta", "salad", "chicken", "soup", "dessert"]
all_recipes = []

for q in queries:
    print(f"Fetching: {q}...")
    results = fetch_recipes(q, number=20)
    all_recipes.extend(results)

df = parse_recipes(all_recipes)
df.drop_duplicates(subset="id", inplace=True)
df.to_csv("recipes.csv", index=False)
print(f"Saved {len(df)} recipes to recipes.csv")