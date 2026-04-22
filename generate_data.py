import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

# Features
size = np.random.normal(120, 40, n).clip(30, 300)
rooms = np.random.randint(1, 7, n)
location = np.random.choice(["city", "suburb", "rural"], n, p=[0.4, 0.4, 0.2])
year_built = np.random.randint(1950, 2023, n)
condition = np.random.choice(["poor", "fair", "good", "excellent"], n)

# Base price
price = size * 3000

# Adjustments
price += rooms * 10000

location_multiplier = {
    "city": 1.5,
    "suburb": 1.2,
    "rural": 0.8
}

condition_bonus = {
    "poor": -20000,
    "fair": 0,
    "good": 20000,
    "excellent": 50000
}

price = price * [location_multiplier[l] for l in location]
price += [condition_bonus[c] for c in condition]

# Newer houses are more expensive
price += (year_built - 1950) * 500

# Noise
price += np.random.normal(0, 20000, n)

df = pd.DataFrame({
    "size": size,
    "rooms": rooms,
    "location": location,
    "year_built": year_built,
    "condition": condition,
    "price": price
})

df.to_csv("dataset.csv", index=False)

print("Dataset generated: dataset.csv")