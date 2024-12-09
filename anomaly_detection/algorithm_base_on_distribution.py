import numpy as np
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
from scipy.stats import wasserstein_distance, ttest_ind
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

data = np.array([
    [0, 0, 0, 0, 0],
    [624, 751, 908, 1122, 893],
    [200, 300, 410, 100, 510],
    [585, 694, 873, 1032, 859],
    [703, 863, 1057, 1220, 1023]
])

def t_test_anomaly(data, threshold=0.05):
    """
    Phát hiện mảng bất thường dựa trên kiểm định T-test.
    Args:
        data (list of lists or numpy array): Dữ liệu đầu vào
    Returns:
        list: Danh sách các mảng bất thường (dựa trên p-value < 0.05)
    """
    anomalies = []
    for i in range(len(data)):
        current = data[i]
        others = np.delete(data, i, axis=0).flatten()
        t_stat, p_value = ttest_ind(current, others)
        if p_value < threshold:
            anomalies.append({
                "index": i,
                "item": data[i].tolist(),
                "p_value": p_value
            })
    return anomalies

def isolation_forest_anomaly(data, contamination=0.2):
    """
    Phát hiện mảng bất thường bằng Isolation Forest.
    Args:
        data (list of lists or numpy array): Dữ liệu đầu vào
        contamination (float): Tỷ lệ bất thường trong dữ liệu
    Returns:
        list: Danh sách các mảng con bất thường
    """
    model = IsolationForest(contamination=contamination, random_state=42)
    predictions = model.fit_predict(data)
    anomalies = [
        {
            "index": i,
            "item": data[i].tolist()
        } for i, pred in enumerate(predictions) if pred == -1
    ]
    return anomalies

def calculate_kl_divergence(data):
    """Calculates KL divergence between all pairs of distributions."""
    kl_divergences = {}
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            p = np.histogram(data[i], bins=10, density=True)[0] + 1e-10
            q = np.histogram(data[j], bins=10, density=True)[0] + 1e-10
            kl_divergence = stats.entropy(p, q)
            kl_divergences[f"Dist {i+1} vs Dist {j+1}"] = kl_divergence
    return kl_divergences

def calculate_wasserstein_distance(data):
    """Calculates Wasserstein distance (Earth Mover's Distance) between all pairs of distributions."""
    distances = {}
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            dist = wasserstein_distance(data[i], data[j])
            distances[f"Dist {i+1} vs Dist {j+1}"] = dist
    return distances

def find_most_different_distribution(data):
    """Finds the distribution that has the largest difference compared to others."""
    kl_divergences = calculate_kl_divergence(data)
    wasserstein_distances = calculate_wasserstein_distance(data)

    # Summarize KL divergence
    kl_scores = {i: 0 for i in range(len(data))}
    for pair, divergence in kl_divergences.items():
        i, j = map(int, pair.replace("Dist ", "").split(" vs "))
        kl_scores[i-1] += divergence
        kl_scores[j-1] += divergence

    # Summarize Wasserstein distances
    wasserstein_scores = {i: 0 for i in range(len(data))}
    for pair, dist in wasserstein_distances.items():
        i, j = map(int, pair.replace("Dist ", "").split(" vs "))
        wasserstein_scores[i-1] += dist
        wasserstein_scores[j-1] += dist

    # Normalize scores and combine
    print(kl_scores)
    print(wasserstein_scores)
    kl_max = max(kl_scores.values())
    wasserstein_max = max(wasserstein_scores.values())

    combined_scores = {
        i: (kl_scores[i] / kl_max + wasserstein_scores[i] / wasserstein_max)
        for i in range(len(data))
    }

    most_different = max(combined_scores, key=combined_scores.get)
    return most_different + 1, combined_scores

def perform_kmeans_clustering(data):
    """Performs KMeans clustering to group similar distributions and returns cluster assignments."""
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(data)
    anomaly_items = [data[i] for i in range(len(clusters)) if clusters[i] == np.argmin(clusters)]
    cluster_assignments = {f"Dist {i+1}": f"Cluster {clusters[i]}" for i in range(len(clusters))}
    print(cluster_assignments)
    print(anomaly_items)
    return anomaly_items


def perform_hierarchical_clustering(data, n_clusters=2):
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = hierarchical.fit_predict(data)
    anomaly_items = [data[i] for i in range(len(clusters)) if clusters[i] == np.argmin(clusters)]
    cluster_assignments = {f"Dist {i+1}": f"Cluster {clusters[i]}" for i in range(len(clusters))}
    # print(cluster_assignments)
    # print(anomaly_items)
    return anomaly_items
# # Identify the most different distribution
# most_different, scores = find_most_different_distribution(data)
# print(f"The most different distribution is Dist {most_different}.")
# print("Scores:")
# for dist, score in scores.items():
#     print(f"Dist {dist+1}: {score:.5f}")
#
# # Perform KMeans clustering
# clusters, cluster_assignments = perform_kmeans_clustering(data)
#
perform_hierarchical_clustering(data)