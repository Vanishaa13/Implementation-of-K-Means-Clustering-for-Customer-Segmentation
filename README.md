# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import required libraries.

2.Load the dataset (Mall_Customers.csv).

3.Select important features (e.g., Annual Income & Spending Score).

4.Use the Elbow Method to find optimal K.

5.Apply K-Means clustering.

6.Visualize the clusters. 
 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Vanishaa Harshini.B.R
RegisterNumber:  212225040481

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


data = pd.read_csv("Mall_Customers.csv")

X = data.iloc[:, [3, 4]].values


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)


plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')


plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=300, c='yellow', label='Centroids')

plt.title("Customer Segments")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()
*/
```

## Output:
<img width="771" height="772" alt="Screenshot 2026-02-25 113033" src="https://github.com/user-attachments/assets/4a8649e8-5e4a-481f-a9aa-9645135200e4" />



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
