import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
train_data = pd.read_csv('C:\\Microsoft VS Code\\IDS\\Train_data.csv')
test_data = pd.read_csv('C:\\Microsoft VS Code\\IDS\\Test_data.csv')
X_train = train_data.drop('label', axis=1)
y_train = train_data['label']
X_test = test_data.drop('label', axis=1)
y_test = test_data['label']
isolation_forest = IsolationForest(contamination=0.1, random_state=42) 
isolation_forest.fit(X_train)
outliers_train = isolation_forest.predict(X_train)
outliers_train = pd.Series(outliers_train).replace({1: 0, -1: 1}) 
outliers_test = isolation_forest.predict(X_test)
outliers_test = pd.Series(outliers_test).replace({1: 0, -1: 1})
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_scaled)
centroids_train = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=outliers_test, cmap='viridis', edgecolor='k')
plt.scatter(centroids_train[:, 0], centroids_train[:, 1], marker='x', s=200, color='red', label='Centroids')
plt.title('Anomaly Detection and Clustering (Test Set)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.colorbar(label='Anomaly')
plt.show()
anomaly_indices_test = outliers_test[outliers_test == 1].index
print("Detected Anomalies in Test Set:")
print(X_test.loc[anomaly_indices_test])
print("Potential Vulnerable Points based on Training Set:")
print(pd.DataFrame(centroids_train, columns=X.columns))
print("Resolve the Vulnerable points")
