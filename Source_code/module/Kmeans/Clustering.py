def k_means_clustering(data, attribute_index, k,labels):
    from sklearn.cluster import KMeans
    import numpy as np
    X = np.array([row[attribute_index] for row in data]).reshape(-1, 1)
    try:
        kmeans = KMeans(n_clusters=k, random_state=0,n_init='auto').fit(X)
        centroids = kmeans.cluster_centers_.flatten().tolist()
        cluster_assignments = kmeans.labels_.tolist()
        cluster_assignments = [labels[i] for i in cluster_assignments]
        return cluster_assignments, centroids
    except:
        return X.T.tolist()[0], 0