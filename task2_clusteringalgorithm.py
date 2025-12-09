# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the dataset
customer_data = pd.read_csv('Mall_Customers.csv')

# Select the features for clustering
X = customer_data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering
kmeans_model = KMeans(n_clusters=5, random_state=42)
clusters = kmeans_model.fit_predict(X_scaled)
customer_data['Cluster_ID'] = clusters

# Get cluster centers back to original scale
cluster_centers = scaler.inverse_transform(kmeans_model.cluster_centers_)

# Map clusters to meaningful labels
cluster_names = {
    0: 'High Income, Low Spending',
    1: 'Low Income, Low Spending',
    2: 'Low Income, High Spending',
    3: 'Average Income & Spending',
    4: 'High Income, High Spending'
}
customer_data['Customer_Segment'] = customer_data['Cluster_ID'].map(cluster_names)

# Display a sample of the labeled data
print("\n✅ Sample of Customers with Cluster Labels:\n")
print(customer_data[['Annual Income (k$)', 'Spending Score (1-100)', 'Cluster_ID', 'Customer_Segment']].head())

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Customer_Segment',
    data=customer_data,
    palette='Paired',
    s=70
)

# Plot cluster centroids
plt.scatter(
    cluster_centers[:, 0],
    cluster_centers[:, 1],
    color='black',
    marker='X',
    s=250,
    alpha=0.7,
    label='Centroids'
)

plt.title('Mall Customer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Segments', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()