# SPOTIFY SONG POPULARITY PREDICTION USING MACHINE LEARNING
# OBJECTIVE 1: DATA PREPROCESSING, CLEANING AND EDA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
# Enhanced Histogram

plt.figure(figsize=(10, 6))

plt.hist(
    df["track_popularity"],
    bins=20,
    edgecolor="black",
    alpha=0.8
)

plt.title("Distribution of Spotify Track Popularity", fontsize=14)
plt.xlabel("Track Popularity", fontsize=12)
plt.ylabel("Number of Tracks", fontsize=12)

plt.xticks(range(0, 101, 10))
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.savefig("Objective1_Popularity_Histogram.png", dpi=300)

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
df = pd.read_csv(r"C:\Users\hp\Downloads\spotify_data_cleaned.csv")

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


# STEP 4: ARTIST POPULARITY VS TRACK POPULARITY
# Scatter Plot with Trendline

plt.figure(figsize=(10, 6))

plt.scatter(
    df["artist_popularity"],
    df["track_popularity"],
    alpha=0.35,
    s=25,
    label="Tracks"
)

# Calculate and display trendline
x = df["artist_popularity"]
y = df["track_popularity"]

valid_data = df[
    ["artist_popularity", "track_popularity"]
].dropna()

z = np.polyfit(
    valid_data["artist_popularity"],
    valid_data["track_popularity"],
    1
)

p = np.poly1d(z)

plt.plot(
    valid_data["artist_popularity"],
    p(valid_data["artist_popularity"]),
    color="red",
    linewidth=2,
    label="Trendline"
)

plt.title(
    "Artist Popularity vs Track Popularity",
    fontsize=14
)

plt.xlabel("Artist Popularity")
plt.ylabel("Track Popularity")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "Objective2_Artist_Popularity_Trendline.png",
    dpi=300
)

plt.show()





# STEP 5: ARTIST FOLLOWERS VS TRACK POPULARITY
# Box Plot by Artist Follower Groups

followers_data = df[
    ["artist_followers", "track_popularity"]
].dropna().copy()

# Divide artists into four follower groups
followers_data["Follower Group"] = pd.qcut(
    followers_data["artist_followers"],
    q=4,
    duplicates="drop"
)

box_data = [
    group["track_popularity"].values
    for _, group in followers_data.groupby(
        "Follower Group",
        observed=True
    )
]

group_labels = [
    str(group)
    for group, _ in followers_data.groupby(
        "Follower Group",
        observed=True
    )
]

plt.figure(figsize=(11, 6))

plt.boxplot(
    box_data,
    tick_labels=group_labels,
    showfliers=False,
    patch_artist=True
)

plt.title("Track Popularity Across Artist Follower Groups")
plt.xlabel("Artist Follower Groups")
plt.ylabel("Track Popularity")

plt.xticks(rotation=15)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "Objective2_Artist_Followers_Boxplot.png",
    dpi=300
)

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



# STEP 8: ALBUM SIZE VS TRACK POPULARITY
# Violin Plot

album_data = df[
    ["album_total_tracks", "track_popularity"]
].dropna().copy()

# Create four album-size categories
album_data["Album Size Category"] = pd.cut(
    album_data["album_total_tracks"],
    bins=[0, 10, 20, 40, float("inf")],
    labels=[
        "Small (1-10)",
        "Medium (11-20)",
        "Large (21-40)",
        "Very Large (40+)"
    ]
)

# Remove empty categories
album_data = album_data.dropna(
    subset=["Album Size Category"]
)

# Prepare data for violin plot
album_categories = [
    "Small (1-10)",
    "Medium (11-20)",
    "Large (21-40)",
    "Very Large (40+)"
]

violin_data = [
    album_data[
        album_data["Album Size Category"] == category
    ]["track_popularity"].values
    for category in album_categories
]

# Create violin plot
plt.figure(figsize=(11, 6))

plt.violinplot(
    violin_data,
    showmeans=True,
    showmedians=True
)

plt.title(
    "Track Popularity Distribution Across Album Sizes",
    fontsize=14
)

plt.xlabel("Album Size Category")
plt.ylabel("Track Popularity")

plt.xticks(
    range(1, 5),
    album_categories,
    rotation=10
)

plt.ylim(0, 100)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    "Objective2_Album_Size_Violin.png",
    dpi=300
)

plt.show()



# STEP 9: TRACK DURATION VS TRACK POPULARITY
# Box Plot by Track Duration Categories

duration_data = df[
    ["track_duration_min", "track_popularity"]
].dropna().copy()

# Create duration categories
duration_data["Duration Category"] = pd.cut(
    duration_data["track_duration_min"],
    bins=[0, 2, 4, 6, float("inf")],
    labels=[
        "Short (0-2 min)",
        "Medium (2-4 min)",
        "Long (4-6 min)",
        "Very Long (6+ min)"
    ]
)

duration_data = duration_data.dropna(
    subset=["Duration Category"]
)

# Prepare popularity data
duration_categories = [
    "Short (0-2 min)",
    "Medium (2-4 min)",
    "Long (4-6 min)",
    "Very Long (6+ min)"
]

box_data = [
    duration_data[
        duration_data["Duration Category"] == category
    ]["track_popularity"].values
    for category in duration_categories
]

# Create box plot
plt.figure(figsize=(11, 6))

plt.boxplot(
    box_data,
    tick_labels=duration_categories,
    showmeans=True,
    showfliers=False
)

plt.title(
    "Track Popularity Across Duration Categories",
    fontsize=14
)

plt.xlabel("Track Duration Categories")
plt.ylabel("Track Popularity")

plt.xticks(rotation=10)
plt.ylim(0, 100)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    "Objective2_Track_Duration_Boxplot.png",
    dpi=300
)

plt.show()







# SPOTIFY SONG POPULARITY PREDICTION
# OBJECTIVE 3: FEATURE ENGINEERING AND SELECTION

import pandas as pd

# Step 1: Load the cleaned dataset
df = pd.read_csv(r"C:\Users\hp\Downloads\spotify_data_cleaned.csv")

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
    r"C:\Users\hp\Downloads\spotify_features_prepared.csv",
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


### Step 1: Load the prepared dataset
##df = pd.read_csv(r"C:\Users\hp\Downloads\spotify_data_cleaned.csv")
##
### Step 2: Separate input features and target
##X = df.drop(columns=["track_popularity"])
##y = df["track_popularity"]

# Step 1: Load the prepared dataset
df = pd.read_csv(r"C:\Users\hp\Downloads\spotify_features_prepared.csv")

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

# OBJECTIVE 4: RANDOM FOREST LEARNING CURVE

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

# Create a lightweight Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=50,
    random_state=42,
    n_jobs=1
)

# Calculate learning curve
train_sizes, train_scores, validation_scores = learning_curve(
    rf_model,
    X_train,
    y_train,
    cv=3,
    scoring="r2",
    train_sizes=np.linspace(0.2, 1.0, 4),
    n_jobs=1
)

# Calculate average scores
train_mean = np.mean(train_scores, axis=1)
validation_mean = np.mean(validation_scores, axis=1)

# Plot graph
plt.figure(figsize=(10, 6))

plt.plot(
    train_sizes,
    train_mean,
    marker="o",
    linewidth=2,
    label="Training R² Score"
)

plt.plot(
    train_sizes,
    validation_mean,
    marker="s",
    linewidth=2,
    label="Validation R² Score"
)

plt.title("Objective 4: Random Forest Learning Curve")
plt.xlabel("Number of Training Samples")
plt.ylabel("R² Score")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig("Objective4_Learning_Curve.png", dpi=300)
plt.show()

print("Learning Curve Generated Successfully!")




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
#df = pd.read_csv(r"C:\Users\hp\Downloads\spotify_data_cleaned.csv")
df = pd.read_csv(r"C:\Users\hp\Downloads\spotify_features_prepared.csv")
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


# STEP 1: LOAD THE PREPARED DATASET

df = pd.read_csv(
    r"C:\Users\hp\Downloads\spotify_features_prepared.csv"
)


# STEP 2: SEPARATE FEATURES AND TARGET

X = df.drop(columns=["track_popularity"])
y = df["track_popularity"]


# STEP 3: SPLIT THE DATA

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# STEP 4: CREATE FOUR REGRESSION MODELS

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}


# STEP 5: FIND THE BEST MODEL USING RMSE

best_model = None
best_rmse = float("inf")
best_name = ""

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = mean_squared_error(
        y_test, predictions
    ) ** 0.5

    if rmse < best_rmse:

        best_rmse = rmse
        best_model = model
        best_name = name


print("\nBest Model:", best_name)
print("Best RMSE:", round(best_rmse, 2))


# STEP 6: PREDICT USING THE BEST MODEL

predictions = best_model.predict(X_test)


# STEP 7: COMPARE ACTUAL AND PREDICTED POPULARITY

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


# STEP 8: IDENTIFY THE LARGEST PREDICTION ERRORS

print("\nTracks with the Largest Prediction Errors:")

print(
    results.sort_values(
        by="Absolute Error",
        ascending=False
    ).head(10)
)


# STEP 9: GROUP TRACKS BY ACTUAL POPULARITY

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


# STEP 10: ACTUAL VS PREDICTED POPULARITY
# BAR GRAPH COMPARISON

comparison = results.groupby(
    "Popularity Group",
    observed=False
).agg(
    Actual=("Actual Popularity", "mean"),
    Predicted=("Predicted Popularity", "mean")
)

comparison.plot(
    kind="bar",
    figsize=(10, 6),
    width=0.8
)

plt.title(
    "Actual vs Predicted Track Popularity",
    fontsize=14
)

plt.xlabel("Popularity Groups")
plt.ylabel("Average Track Popularity")

plt.xticks(rotation=0)

plt.legend(
    ["Actual Popularity", "Predicted Popularity"]
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    "Objective6_Actual_vs_Predicted_Bar.png",
    dpi=300
)

plt.show()


# STEP 11: DISTRIBUTION OF PREDICTION ERRORS

plt.figure(figsize=(10, 6))

plt.hist(
    results["Absolute Error"],
    bins=15,
    edgecolor="black",
    alpha=0.8
)

plt.title(
    "Distribution of Prediction Errors",
    fontsize=14
)

plt.xlabel("Absolute Error")
plt.ylabel("Number of Tracks")

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    "Objective6_Prediction_Error_Histogram.png",
    dpi=300
)

plt.show()


print("\nOBJECTIVE 6 COMPLETED")

