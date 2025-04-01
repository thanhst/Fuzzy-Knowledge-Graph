def FIS(df,k_values,labels):
    import pandas as pd
    from module.Kmeans.Clustering import k_means_clustering
    y = df.iloc[:,-1:]
    data = df.values.tolist()
    clusters = []
    centroids = []
    for i in range(len(data[0])-1):
        cluster_assignments, centroid = k_means_clustering(data, i, k_values[i],labels[i])
        clusters.append(cluster_assignments)
        centroids.append(centroid)
    out_df = pd.DataFrame(clusters).T
    out_df = pd.concat([out_df,y],axis=1)
    out_df.columns = df.columns
    return out_df