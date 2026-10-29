

# SPOTIFY SONG POPULARITY PREDICTION USING MACHINE LEARNING
# OBJECTIVE 1: DATA PREPROCESSING, CLEANING AND EDA

import pandas as pd
import matplotlib.pyplot as plt

# STEP 1: LOAD THE DATASET

df = pd.read_csv(r"C:\Users\hp\Downloads\spotify_data_cleaned.csv")

print("ORIGINAL DATASET")
print("----------------")
print("Number of rows and columns:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst Five Records:")
print(df.head())


# STEP 2: CHECK DATA QUALITY

print("\nMISSING VALUES BEFORE CLEANING")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())


# STEP 3: HANDLE MISSING VALUES

df["artist_name"] = df["artist_name"].fillna("Unknown Artist")

df["artist_genres"] = df["artist_genres"].fillna("Unknown")


# STEP 4: REMOVE DUPLICATE RECORDS

df = df.drop_duplicates()


# STEP 5: CHECK THE CLEANED DATASET

print("\nCLEANED DATASET")
print("----------------")
print("Number of rows and columns:", df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Records After Cleaning:")
print(df.duplicated().sum())


# STEP 6: DESCRIPTIVE STATISTICS

print("\nSTATISTICAL SUMMARY")
print(df.describe())


# STEP 7: ANALYSE TRACK POPULARITY

print("\nTRACK POPULARITY SUMMARY")
print(df["track_popularity"].describe())


# STEP 8: VISUALIZE TRACK POPULARITY

plt.hist(df["track_popularity"], bins=10)

plt.title("Distribution of Spotify Track Popularity")
plt.xlabel("Track Popularity")
plt.ylabel("Number of Tracks")

plt.show()


# STEP 9: CORRELATION ANALYSIS

print("\nCORRELATION BETWEEN NUMERICAL FEATURES")

print(df.corr(numeric_only=True))


# STEP 10: SAVE THE CLEANED DATASET

df.to_csv("spotify_data_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")
print("OBJECTIVE 1 COMPLETED")










# SPOTIFY SONG POPULARITY PREDICTION
# OBJECTIVE 2: ARTIST INFLUENCE AND TRACK CHARACTERISTICS

import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the cleaned dataset
df = pd.read_csv("spotify_data_cleaned.csv")

# Step 2: Analyse numerical features
print("ARTIST AND TRACK CHARACTERISTICS")
print("--------------------------------")

columns = [
    "track_popularity",
    "artist_popularity",
    "artist_followers",
    "track_duration_min",
    "album_total_tracks"
]

print(df[columns].describe())

# Step 3: Correlation with track popularity
print("\nCorrelation with Track Popularity:")

print(df[columns].corr()["track_popularity"].sort_values(
    ascending=False
))

# Step 4: Artist popularity vs track popularity
plt.scatter(df["artist_popularity"], df["track_popularity"])

plt.title("Artist Popularity vs Track Popularity")
plt.xlabel("Artist Popularity")
plt.ylabel("Track Popularity")

plt.show()

# Step 5: Artist followers vs track popularity
plt.scatter(df["artist_followers"], df["track_popularity"])

plt.title("Artist Followers vs Track Popularity")
plt.xlabel("Artist Followers")
plt.ylabel("Track Popularity")

plt.show()

# Step 6: Explicit content vs track popularity
print("\nAverage Popularity by Explicit Status:")

print(df.groupby("explicit")["track_popularity"].mean())

df.groupby("explicit")["track_popularity"].mean().plot(kind="bar")

plt.title("Explicit Content vs Track Popularity")
plt.xlabel("Explicit Content")
plt.ylabel("Average Track Popularity")

plt.show()

# Step 7: Album type vs track popularity
print("\nAverage Popularity by Album Type:")

print(df.groupby("album_type")["track_popularity"].mean())

df.groupby("album_type")["track_popularity"].mean().plot(kind="bar")

plt.title("Album Type vs Track Popularity")
plt.xlabel("Album Type")
plt.ylabel("Average Track Popularity")

plt.show()

# Step 8: Album size vs track popularity
print("\nAverage Popularity by Album Size:")

print(df.groupby("album_total_tracks")["track_popularity"].mean())

plt.scatter(df["album_total_tracks"], df["track_popularity"])

plt.title("Album Size vs Track Popularity")
plt.xlabel("Number of Tracks in Album")
plt.ylabel("Track Popularity")

plt.show()

# Step 9: Track duration vs track popularity
plt.scatter(df["track_duration_min"], df["track_popularity"])

plt.title("Track Duration vs Track Popularity")
plt.xlabel("Track Duration (Minutes)")
plt.ylabel("Track Popularity")

plt.show()

print("\nOBJECTIVE 2 COMPLETED")









# SPOTIFY SONG POPULARITY PREDICTION
# OBJECTIVE 3: FEATURE ENGINEERING AND SELECTION

import pandas as pd

# Step 1: Load the cleaned dataset
df = pd.read_csv("spotify_data_cleaned.csv")

# Step 2: Select useful features
features = [
    "artist_popularity",
    "artist_followers",
    "track_duration_min",
    "explicit",
    "album_type",
    "album_total_tracks"
]

target = "track_popularity"

# Step 3: Keep the selected columns
data = df[features + [target]].copy()

# Step 4: Handle missing values
data = data.dropna()

# Step 5: Convert categorical features into numerical values
data = pd.get_dummies(
    data,
    columns=["explicit", "album_type"],
    drop_first=True,
    dtype=int
)

# Step 6: Separate features and target
X = data.drop(columns=[target])
y = data[target]

# Step 7: Display the selected features
print("Selected Features:")
print(X.columns.tolist())

print("\nNumber of Records:", X.shape[0])
print("Number of Features:", X.shape[1])

print("\nFirst Five Records:")
print(X.head())

# Step 8: Save the prepared dataset
prepared_data = X.copy()
prepared_data[target] = y

prepared_data.to_csv(
    "spotify_features_prepared.csv",
    index=False
)

print("\nPrepared dataset saved successfully!")
print("OBJECTIVE 3 COMPLETED")





# SPOTIFY SONG POPULARITY PREDICTION
# OBJECTIVE 4: MACHINE-LEARNING MODEL DEVELOPMENT

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, r2_score


# Step 1: Load the prepared dataset
df = pd.read_csv("spotify_features_prepared.csv")

# Step 2: Separate input features and target
X = df.drop(columns=["track_popularity"])
y = df["track_popularity"]

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# Step 4: Create four regression models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# Step 5: Train and test each model
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nModel:", name)
    print("MAE:", round(mae, 2))
    print("R2 Score:", round(r2, 3))

print("\nOBJECTIVE 4 COMPLETED")






# SPOTIFY SONG POPULARITY PREDICTION
# OBJECTIVE 5: MODEL EVALUATION AND INTERPRETATION

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the prepared dataset
df = pd.read_csv("spotify_features_prepared.csv")

# Step 2: Separate features and target
X = df.drop(columns=["track_popularity"])
y = df["track_popularity"]

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create four models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# Step 5: Train and evaluate the models
results = []

for name, model in models.items():

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5

    r2 = r2_score(y_test, predictions)

    results.append([name, mae, rmse, r2])

# Step 6: Display the comparison
results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "RMSE", "R2 Score"]
)

print("\nMODEL COMPARISON")
print(results_df.round(3))

# Step 7: Identify the best model by RMSE
best_model_name = results_df.loc[
    results_df["RMSE"].idxmin(), "Model"
]

print("\nBest model based on RMSE:", best_model_name)

# Step 8: Visualize model performance
results_df.plot(
    x="Model",
    y=["MAE", "RMSE"],
    kind="bar"
)

plt.title("Regression Model Error Comparison")
plt.ylabel("Error")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# Step 9: Display feature importance for tree-based models
for name in ["Decision Tree", "Random Forest", "Gradient Boosting"]:

    model = models[name]

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    )

    print("\nFeature Importance -", name)
    print(importance.sort_values(ascending=False))

print("\nOBJECTIVE 5 COMPLETED")









# SPOTIFY SONG POPULARITY PREDICTION
# OBJECTIVE 6: ERROR ANALYSIS

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error

# Step 1: Load the prepared dataset
df = pd.read_csv("spotify_features_prepared.csv")

# Step 2: Separate features and target
X = df.drop(columns=["track_popularity"])
y = df["track_popularity"]

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create four regression models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# Step 5: Find the best model using RMSE
best_model = None
best_rmse = float("inf")
best_name = ""

for name, model in models.items():

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = mean_squared_error(y_test, predictions) ** 0.5

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_name = name

print("Best Model:", best_name)
print("Best RMSE:", round(best_rmse, 2))

# Step 6: Predict using the best model
predictions = best_model.predict(X_test)

# Step 7: Compare actual and predicted popularity
results = pd.DataFrame({
    "Actual Popularity": y_test,
    "Predicted Popularity": predictions
})

results["Error"] = (
    results["Actual Popularity"] -
    results["Predicted Popularity"]
)

results["Absolute Error"] = results["Error"].abs()

print("\nActual and Predicted Popularity:")
print(results.head(10))

# Step 8: Identify the largest prediction errors
print("\nTracks with the Largest Prediction Errors:")
print(
    results.sort_values(
        by="Absolute Error",
        ascending=False
    ).head(10)
)

# Step 9: Group tracks by actual popularity
results["Popularity Group"] = pd.cut(
    results["Actual Popularity"],
    bins=[-1, 33, 66, 100],
    labels=["Low", "Medium", "High"]
)

print("\nAverage Prediction Error by Popularity Group:")
print(
    results.groupby(
        "Popularity Group",
        observed=False
    )["Absolute Error"].mean()
)

# Step 10: Plot actual vs predicted popularity
plt.scatter(
    results["Actual Popularity"],
    results["Predicted Popularity"]
)

plt.title("Actual vs Predicted Track Popularity")
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")

plt.show()

# Step 11: Plot absolute prediction errors
plt.hist(results["Absolute Error"], bins=10)

plt.title("Distribution of Prediction Errors")
plt.xlabel("Absolute Error")
plt.ylabel("Number of Tracks")

plt.show()

print("\nOBJECTIVE 6 COMPLETED")




























