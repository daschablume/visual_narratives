def cluster_connected_components(sim_matrix, threshold):
    n = len(sim_matrix)
    visited = [False] * n
    clusters = []

    def dfs(node, cluster):
        visited[node] = True
        cluster.append(node)
        for neighbor in range(n):
            if not visited[neighbor] and sim_matrix[node][neighbor] >= threshold:
                dfs(neighbor, cluster)

    for i in range(n):
        if not visited[i]:
            cluster = []
            dfs(i, cluster)
            clusters.append(cluster)

    return clusters