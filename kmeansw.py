import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from regresssion import line
import time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = pd.read_csv("german_credit_data.csv")
line()
print("THE SHAPE OF THE DATA : ")
print(data.shape)
line()
print("THE TOP 5 VALUES IN THE DATA : ")
print(data.head())
line()
print("INFO")
print(data.info())
line()
print("OVERALL DESCRIPTION OF DATA: ")
print(data.describe())
line()

print("THE NUMBER OF MISSING VALUES :")
print(data.isnull().sum())
line()

numerical_values = ['Credit amount', 'Age', 'Duration']
categorical_values = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
unused = ['Unnamed: 0']

print("DATA AFTER DROPPING UNUSED COLUMN : ")
data = data.drop(columns=unused)
print(data)
line()

for cat in categorical_values:
    data[cat] = data[cat].fillna(data[cat].mode().values[0])

print("THE NUMBER OF MISSING VALUES AFTER FILLING")
print(data.isnull().sum())
line()

corr = data.corr(method='pearson', numeric_only=True)

mask = np.array(corr)

mask[np.tril_indices_from(mask)] = False

data_cluster = pd.DataFrame()

for num in numerical_values:
    data_cluster[num] = data[num]

print("THE TOP 5 VALUES OF DATA CLUSTER: \n", data_cluster.head())
line()

data_cluster_log = np.log(data_cluster[['Age', 'Credit amount', 'Duration']])
print("THE TOP 5 VALUES OF DATA CLUSTER LOG: \n", data_cluster_log.head())
line()

scaler = StandardScaler()
clustur_scaled = scaler.fit_transform(data_cluster_log)

print("THE CLUSTER VALUES AFTER SCALING : \n", clustur_scaled)

sum_of_squared_distance = []
K = range(1, 15)
for k in K:
    km = KMeans(n_clusters=k, n_init=10)
    km = km.fit(clustur_scaled)
    sum_of_squared_distance.append(km.inertia_)

line()
print("THE SUM OF SQUARED DISTANCES : ")
print(sum_of_squared_distance)
line()

time.sleep(0.5)
fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(corr, mask=mask, vmax=0.9, square=True, annot=True)
plt.show()

plt.plot(K, sum_of_squared_distance)
plt.xlabel("K")
plt.ylabel("SUM OF SQUARED DISTANCE")
plt.title("ELBOW METHOD FOR OPTIMAL K")
plt.show()

model = KMeans(n_clusters=3, n_init=10)
model.fit(clustur_scaled)
kmeans_label = model.labels_

fig1 = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='r')
ax1 = plt.axes(projection="3d")
ax1.scatter3D(data_cluster['Age'], data_cluster['Credit amount'], data_cluster['Duration'], c=kmeans_label, cmap='rainbow')
xLabel = ax1.set_xlabel('Age', linespacing=3.2)
yLabel = ax1.set_ylabel('Credit amount', linespacing=3.1)
zLabel = ax1.set_zlabel('Duration', linespacing=3.4)
plt.show()

cluster = pd.DataFrame(data_cluster)
labels = pd.DataFrame(model.labels_)

clustered_data = cluster.assign(Cluster=labels)

print("CLUSTER RESULT BY K-MEANS : ")
print(clustered_data.groupby(['Cluster']).mean().round(1))
