from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import pairwise_distances
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 64)  # Hiển thị tối đa 10 hàng
pd.set_option('display.max_columns', 10)

df = pd.read_csv('online_retail_new.csv')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
analysis_date = np.max(df['InvoiceDate'])

rfm = df.groupby('CustomerID').agg(Recency=('InvoiceDate', lambda x: (analysis_date - x.max()).days),
                                   Frequency=('InvoiceNo', lambda x: x.nunique()),
                                   Monetary=('SaleAmount', 'sum'))

# Kiểm định Hopkins
def generate_random_data(data, n):
    """ Sinh dữ liệu ngẫu nhiên có cùng phân phối với dữ liệu gốc """
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return np.random.uniform(min_vals, max_vals, size=(n, data.shape[1]))

def hopkins_statistic(data):
    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # Số lượng điểm dữ liệu
    n = data.shape[0]

    # Chọn một số điểm mẫu
    sample_size = int(n / 2)

    # Chọn một số điểm mẫu
    data_sample = data[np.random.choice(data.shape[0], sample_size, replace=False)]

    # Sinh dữ liệu ngẫu nhiên
    random_data = generate_random_data(data, sample_size)

    # Tính khoảng cách
    dist_real = pairwise_distances(data_sample, data)
    dist_rand = pairwise_distances(data_sample, random_data)

    # Tính toán chỉ số Hopkins
    H = np.sum(np.min(dist_rand, axis=1)) / (np.sum(np.min(dist_real, axis=1)) + np.sum(np.min(dist_rand, axis=1)))

    return H

rfm_scaled = rfm[['Recency', 'Frequency', 'Monetary']]
H_data = rfm_scaled.to_numpy()
H = hopkins_statistic(H_data)
print(f'Chỉ số Hopkins :{H}')


# Chuẩn hóa min-max
scaler = MinMaxScaler()
rfm_scaled = scaler.fit_transform(rfm_scaled)

# # Chọn số lượng cụm tối ưu bằng thuật toán Elbow
# kmeans = KMeans()
# elbow = KElbowVisualizer(kmeans, k=(2, 20))
# elbow.fit(rfm_scaled)
# elbow.show()
#
# # Lấy giá trị elbow
# optimal_clusters = elbow.elbow_value_
# print(f'\nSố lượng cụm tối ưu: {optimal_clusters}\n')
# Xác định số lượng cụm tối ưu (có thể thử các số lượng cụm khác nhau)
# k_values = range(2, 21)
# dbi_scores = []
#
# for k in k_values:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     clusters = kmeans.fit_predict(rfm_scaled)
#     dbi = davies_bouldin_score(rfm_scaled, clusters)
#     dbi_scores.append(dbi)
#     print(f'Số lượng cụm: {k}, Davies-Bouldin Index: {dbi:.3f}')
# Truy xuất trung tâm cụm
optimal_clusters = 2
kmeans = KMeans(optimal_clusters).fit(rfm_scaled)
cluster_centers = kmeans.cluster_centers_

# Chuyển đổi ngược tâm cụm về không gian gốc
centroids_original = scaler.inverse_transform(cluster_centers)
for i, cluster_center in enumerate(centroids_original):
    print(f'Cluster {i + 1}: Recency = {cluster_center[0]:.2f}, Frequency = {cluster_center[1]:.2f}, Monetary = {cluster_center[2]:.2f}')
print(f'Inertia của mô hình KMeans: {kmeans.inertia_}')

# Gán label vào các cụm tương ứng
clusters = kmeans.labels_
rfm_kmeans = rfm[['Recency', 'Frequency', 'Monetary']]
rfm_kmeans['Cluster'] = clusters + 1
rfm_kmeans = rfm_kmeans.reset_index()
print(rfm_kmeans)
rfm_kmeans.to_csv('kmeans.csv', index=False)

# Tính điểm Silhouette
silhouette_avg = silhouette_score(rfm_scaled, clusters)
print(f'Điểm Silhouette trung bình: {silhouette_avg:.3f}')

# Tạo scatter plot 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
for cluster in range(1, optimal_clusters+1):
    cluster_data = rfm_kmeans[rfm_kmeans['Cluster'] == cluster]
    ax.scatter(cluster_data['Recency'], cluster_data['Frequency'], cluster_data['Monetary'],
               label=f'Cluster {cluster}')
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
ax.zaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%1.0fK' % (x * 1e-3)))
ax.set_title('3D Scatter Plot of Clusters (Recency vs Frequency vs Monetary)')
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.show()














# Đánh giá kết quả phân cụm

# So sánh với phương pháp quintiles
# quintiles = pd.read_csv('quintiles.csv')
# rfm_kmeans = pd.merge(rfm_kmeans, quintiles, on='CustomerID', how='inner', )
# rfm_kmeans = rfm_kmeans[['CustomerID', 'Recency_x', 'Frequency_x', 'Monetary_x', 'Cluster', 'RFM_Score', 'Segments']]
# rfm_kmeans.rename(columns={'Recency_x': 'Recency', 'Frequency_x': 'Frequency',
#                            'Monetary_x': 'Monetary', 'Cluster': 'Quintiles Segment'}, inplace=True)
# rfm_kmeans.sort_values(by='Cluster', ascending=True, inplace=True)
# print(rfm_kmeans)

