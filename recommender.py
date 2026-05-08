import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. Load data 
df = pd.read_csv("recipes.csv")

#  2. Normalize nutrition features 
features = ["calories", "protein", "fat", "carbs"]
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

# 3. Recommend function (Euclidean distance) 
def recommend(title, top_n=5):
    title = title.lower()

    matches = df[df["title"].str.lower().str.contains(title)]
    if matches.empty:
        print(f"No recipe found containing '{title}'")
        return

    idx = matches.index[0]
    recipe_name = df.loc[idx, "title"]
    print(f"\nRecipes similar to: '{recipe_name}'")
    print(f"Calories: {df.loc[idx,'calories']} | Protein: {df.loc[idx,'protein']}g | Fat: {df.loc[idx,'fat']}g | Carbs: {df.loc[idx,'carbs']}g")
    print("-" * 60)

    # Get the target vector
    target = df_scaled.iloc[idx].values

    # Compute euclidean distance from target to all others
    distances = np.linalg.norm(df_scaled.values - target, axis=1)

    # Sort by distance (smallest = most similar)
    sorted_indices = np.argsort(distances)

    count = 0
    for i in sorted_indices:
        if i == idx:
            continue
        print(f"{count+1}. {df.loc[i,'title']}")
        print(f"   Calories: {df.loc[i,'calories']} | Protein: {df.loc[i,'protein']}g | Fat: {df.loc[i,'fat']}g | Carbs: {df.loc[i,'carbs']}g")
        print(f"   Distance: {distances[i]:.4f}")
        count += 1
        if count == top_n:
            break

#4. Test 
recommend("pasta")
recommend("chicken")