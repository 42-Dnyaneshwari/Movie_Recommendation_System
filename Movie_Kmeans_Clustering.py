import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# Read the dataset from CSV
df = pd.read_csv('C:/DMWL/Main_Data.csv')
df.columns
# One-Hot encode categorical variables
onehot_encoder = OneHotEncoder(sparse=False)
genre_encoded = onehot_encoder.fit_transform(df[["genre"]])
rating_encoded = onehot_encoder.fit_transform(df[["rating"]])

# Label encode year
label_encoder = LabelEncoder()
year_encoded = label_encoder.fit_transform(df["year"])

# Standardize numerical variables
scaler = StandardScaler()
numerical_features = ['score(IMDB)', 'score( Rotten Tomatoes)', 'score','votes', "runtime"]
numerical_encoded = scaler.fit_transform(df[numerical_features])

# Combine encoded features
encoded_features = pd.concat([pd.DataFrame(genre_encoded), pd.DataFrame(rating_encoded), pd.DataFrame({"year_encoded": year_encoded}), pd.DataFrame(numerical_encoded)], axis=1)

# Optionally, drop the original categorical and numerical features
df.drop(["genre", "rating", "year"] + numerical_features, axis=1, inplace=True)

# Concatenate the encoded features with the original dataframe
new_df = pd.concat([df, encoded_features], axis=1)

print(new_df)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
# Load the movie dataset
data =pd.read_csv('C:/DMWL/Main_Data.csv')

#Heatmap

sns.heatmap(df.corr())

# Select features for clustering
features = ['score(IMDB)', 'score( Rotten Tomatoes)', 'score','votes', "runtime"]

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

# Determine epsilon using k-distance plot
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(data_scaled)
distances, indices = nbrs.kneighbors(data_scaled)
distances = sorted(distances[:, 1], reverse=True)

# Plot the distances to determine epsilon

# Based on the plot, select epsilon
epsilon = 1.0

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=epsilon, min_samples=5)
clusters = dbscan.fit_predict(data_scaled)

# Add cluster labels to the original dataset
data['cluster'] = clusters

# Function to get movie recommendations for a given cluster
def get_recommendations(cluster_id, n=5):
    # Select movies from the specified cluster
    cluster_movies = data[data['cluster'] == cluster_id]
    
    # Sort movies by rating or any other criteria
    sorted_movies = cluster_movies.sort_values(by='score(IMDB)', ascending=False)
    
    # Get top n movie recommendations
    top_n = sorted_movies.head(n)
    
    # Return the movie names
    recommendations = top_n['name'].values
    return recommendations

# Example: Get recommendations for Cluster 0
cluster_id = 1
recommendations = get_recommendations(cluster_id)
print("Top 5 movie recommendations for Cluster", cluster_id, ":")
for movie_name in recommendations:
    print(movie_name)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from sklearn.metrics import mean_squared_error
import numpy as np

# Calculate centroids of the clusters
centroids = []
for cluster_id in np.unique(clusters):
    cluster_data = data_scaled[clusters == cluster_id]
    centroid = np.mean(cluster_data, axis=0)
    centroids.append(centroid)

# Assign each data point to its nearest centroid
cluster_assignments = [np.argmin(np.linalg.norm(data_point - centroids, axis=1)) for data_point in data_scaled]

# Calculate MSE using cluster centroids
mse = mean_squared_error(data_scaled, [centroids[cluster_id] for cluster_id in cluster_assignments])
print("MSE:", mse)

# Calculate RMSE
rmse = np.sqrt(mse)
print("RMSE:", rmse)
