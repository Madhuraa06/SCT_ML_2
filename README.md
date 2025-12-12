**Mall Customer Segmentation – KMeans Clustering**

This project performs customer segmentation using the K-Means clustering algorithm on the Mall Customers dataset. The model groups customers based on Annual Income and Spending Score, helping identify key customer types.

📌 Features

Loads and preprocesses the dataset

Standardizes inputs using StandardScaler

Applies KMeans (5 clusters)

Assigns human-readable customer segment labels

Visualizes clusters with Matplotlib

Displays sample output with predicted segments

🚀 How to Run pip install pandas matplotlib scikit-learn python task2_clusteringalgorithm.py

📊 Output

Clustered scatter plot showing customer groups

Centroids marked with black “X”

Printed sample table with:

Annual Income

Spending Score

Cluster ID

Customer Type

📁 Files

Mall_Customers.csv – dataset

task2_clusteringalgorithm.py – main clustering script

🧠 Segments Identified

High Income, Low Spending

Low Income, Low Spending

Low Income, High Spending

Average Income & Spending

High Income, High Spending
