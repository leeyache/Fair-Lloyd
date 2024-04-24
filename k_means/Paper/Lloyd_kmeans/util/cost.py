import numpy as np



# 计算每个种群的代价
def caculate_cost(k,data,centroids,group_index,assign,name2ix,unique_values):
    cost = np.zeros(len(unique_values))
    for group in unique_values:
        for i in range(k) :
            group_in_cur_clustering = group_index[name2ix[group]] & (assign == i)
            cluster_data = data[group_in_cur_clustering]
            distances = np.sum((cluster_data - centroids[i])**2,axis=1)

            cost[name2ix[group]] += sum(distances)
        cost[name2ix[group]] /= sum(group_index[name2ix[group]])
    return cost