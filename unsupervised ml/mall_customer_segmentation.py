# Mall Customer Segmentation using K-Means Clustering
# A Step-by-Step Beginner-Friendly ML Project

# ==========================================
# STEP 1: IMPORTING LIBRARIES
# ==========================================
# We import the tools we need for loading data, analyzing it, scaling, clustering, and plotting.
import pandas as pd  # Used to handle data in tabular format (DataFrames)
import numpy as np   # Used for numerical computations
import matplotlib.pyplot as plt  # Used to create static, interactive visualizations
import seaborn as sns  # Built on top of matplotlib, used for making beautiful statistical graphics
from sklearn.preprocessing import StandardScaler  # Used to scale features to a similar range
from sklearn.cluster import KMeans  # The K-Means algorithm class
from sklearn.metrics import silhouette_score  # Metric to evaluate clustering quality

print("--- Step 1: Libraries imported successfully! ---")

# ==========================================
# STEP 2: DATA LOADING
# ==========================================
# We load the dataset directly from a public raw URL using pandas.
url = "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2025%20-%20Hierarchical%20Clustering/Mall_Customers.csv"
df = pd.read_csv(url)

print("\n--- Step 2: Data loaded! Here are the first 5 rows: ---")
print(df.head())

# ==========================================
# STEP 3: BASIC EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
# Let's understand the size of the dataset and look for any missing values.
print("\n--- Step 3: Exploratory Data Analysis (EDA) ---")
print(f"Dataset Shape (Rows, Columns): {df.shape}")

print("\nMissing values in each column:")
print(df.isnull().sum())  # Sums up True (1) values for missing cells; we expect all 0s.

print("\nBasic Summary Statistics:")
print(df.describe())  # Shows mean, standard deviation, min, max, etc., for numerical columns.

# ==========================================
# STEP 4: PREPROCESSING & FEATURE SELECTION
# ==========================================
# We choose "Annual Income" and "Spending Score" for our clustering.
# This is a standard 2D feature set that reveals distinct customer groups.
# We select columns at index 3 (Annual Income) and 4 (Spending Score).
# Note: Python uses 0-based indexing (0: CustomerID, 1: Gender, 2: Age, 3: Income, 4: Score).
X = df.iloc[:, [3, 4]].values

print("\n--- Step 4: Features selected for clustering ---")
print(f"First 5 rows of our selected features (Income & Spending Score):\n{X[:5]}")

# ==========================================
# STEP 5: FEATURE SCALING
# ==========================================
# K-Means calculates distance between points to form clusters. 
# Since Annual Income can range up to 137k while Spending Score is 1-100,
# the income feature would dominate the distance calculation if not scaled.
# We use StandardScaler to scale the data so it has a mean of 0 and variance of 1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n--- Step 5: Feature scaling completed ---")
print(f"First 5 rows of scaled features:\n{X_scaled[:5]}")

# ==========================================
# STEP 6: ELBOW METHOD (FINDING OPTIMAL K)
# ==========================================
# WCSS (Within-Cluster Sum of Squares) measures how compact/close the points are inside each cluster.
# We run K-Means for 1 to 10 clusters and plot the WCSS.
wcss = []
for k in range(1, 11):
    # n_init=10 runs K-Means 10 times with different starting points and picks the best one.
    # random_state=42 ensures we get the same results every time we run the code.
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)  # inertia_ is the WCSS value

# Plotting the Elbow Graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='teal', linewidth=2)
plt.title('Elbow Method to Find Optimal Clusters (K)', fontsize=14)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('WCSS (Error/Inertia)', fontsize=12)
plt.xticks(range(1, 11))
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('elbow_plot.png') # Saves plot as image
print("\n--- Step 6: Elbow plot generated and saved as 'elbow_plot.png' ---")

# ==========================================
# STEP 7: K-MEANS MODEL TRAINING (K=5)
# ==========================================
# Looking at the elbow plot, the rate of decrease slows down significantly after K=5.
# This makes K=5 the "elbow" point, which is our optimal number of clusters.
optimal_k = 5
kmeans_model = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
y_clusters = kmeans_model.fit_predict(X_scaled)

# Add the cluster label back to our original DataFrame for interpretation
df['Cluster'] = y_clusters
print("\n--- Step 7: Model trained on 5 clusters! ---")
print("First 5 rows with their cluster labels:")
print(df.head())

# ==========================================
# STEP 8: SILHOUETTE SCORE
# ==========================================
# The Silhouette Score ranges from -1 to 1. 
# A score close to 1 indicates that points are far from neighboring clusters and close to their own.
sil_score = silhouette_score(X_scaled, y_clusters)
print(f"\n--- Step 8: Silhouette Score for K=5 is: {sil_score:.4f} ---")

# ==========================================
# STEP 9: CLUSTER VISUALIZATION
# ==========================================
# We plot the data points colored by their cluster label and mark the centroids.
plt.figure(figsize=(11, 7))

# We need to find the centroids of the clusters in the original scale
centroids = scaler.inverse_transform(kmeans_model.cluster_centers_)

# Dynamically assign labels based on centroids' average income and spending score
cluster_labels = {}
for i in range(optimal_k):
    income = centroids[i, 0]
    spend = centroids[i, 1]
    
    if income < 40 and spend > 60:
        desc = "Low Income, High Spend (Careless)"
    elif income > 70 and spend > 60:
        desc = "High Income, High Spend (Target)"
    elif income > 70 and spend < 40:
        desc = "High Income, Low Spend (Conservative)"
    elif income < 40 and spend < 40:
        desc = "Low Income, Low Spend (Frugal)"
    else:
        desc = "Medium Income, Medium Spend (Standard)"
        
    cluster_labels[i] = f"Cluster {i}: {desc}"

# Define colors for our 5 clusters
colors = ['#FF4136', '#2ECC40', '#0074D9', '#FFDC00', '#B10DC9']

for i in range(optimal_k):
    # Select points belonging to cluster i
    plt.scatter(X[y_clusters == i, 0], X[y_clusters == i, 1], 
                s=70, c=colors[i], label=cluster_labels[i], alpha=0.8, edgecolors='black')

# Plotting the Centroids of the clusters
plt.scatter(centroids[:, 0], centroids[:, 1], 
            s=250, c='black', marker='X', label='Centroids', edgecolors='white', linewidth=2)

plt.title('Customer Segments using K-Means Clustering', fontsize=16, fontweight='bold')
plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig('customer_clusters.png') # Saves plot as image
print("--- Step 9: Cluster visualization plot saved as 'customer_clusters.png' ---")

# ==========================================
# STEP 10: INTERPRETATION
# ==========================================
print("\n--- Step 10: Interpretation of Customer Segments ---")
for i in range(optimal_k):
    cluster_subset = df[df['Cluster'] == i]
    mean_income = cluster_subset['Annual Income (k$)'].mean()
    mean_spending = cluster_subset['Spending Score (1-100)'].mean()
    count = len(cluster_subset)
    print(f"\n{cluster_labels[i]}:")
    print(f"  - Count: {count} customers")
    print(f"  - Avg Annual Income: ${mean_income:.1f}k")
    print(f"  - Avg Spending Score: {mean_spending:.1f}/100")
