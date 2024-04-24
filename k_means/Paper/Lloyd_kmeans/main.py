import util.load
import numpy as np
import sys
import matplotlib.pyplot as plt
import util.load
from sklearn.decomposition import PCA
import ast
import util.dataprocess
from sklearn.cluster import KMeans
import util.cost
def main():
    # 数据集所在文件夹
    config_root = 'Resource'
    # 选用的数据集
    config_name = sys.argv[1]
    # 加载配置文件
    config_parser = util.load.load_configuration(config_root,config_name)
    dataRoot = config_parser.get(config_name,'dataRoot')
    dataName = config_parser.get(config_name,'dataName')
    # 加载数据
    df = util.load.load_data(dataRoot,dataName)
    # 是否属于PCA
    PCAUse = config_parser[config_name].getboolean('PCA')
    # 获取聚类使用的数目
    k_number_str = config_parser[config_name].get('k_number')
    k_number_min,k_number_max = ast.literal_eval(k_number_str)
    k_number = range(k_number_min,k_number_max)
    #对数据进行缺省值处理
    util.dataprocess.fill_na(df)
    fair_column = config_parser[config_name].get('fair_column')
    # 获取聚类中不会使用的列
    drop_columns_str = config_parser[config_name].get('drop_columns')
    if drop_columns_str== '':
        drop_columns = None
    else:
        drop_columns = [column.strip() for column in drop_columns_str.split(',')]
    # 删除聚类中不需要的列
    df_for_encoder = util.dataprocess.drop_columns(df, drop_columns)
    # 对除了保护列之外的所有数据进行独热编码
    df_encode = util.dataprocess.one_hot(df_for_encoder,fair_column)
    df_for_scale = util.dataprocess.drop_columns(df_encode, fair_column)
    # 对数据进行标准化处理
    df_scale = util.dataprocess.scaler(df_for_scale)
    df_fair_column = df[fair_column]
    # 用于存储结果的列表
    result_list = [np.zeros(len(df_fair_column.unique())) for _ in k_number ]
    result_list = np.array(result_list)
    #迭代次数
    enum_number = int(sys.argv[2])
    filename_result = 'Result\\'+config_name+'_'+fair_column+'_' + str(enum_number)+'_'+str(k_number_min)+'_'+str(k_number_max)+'_result.txt'
    fig_name = 'Result\\'+config_name+'_'+fair_column + '_' + str(enum_number) + '_' + str(k_number_min) + '_' + str(
        k_number_max) + 'Floyd_cost.png'
    if PCAUse:
        pca_number = config_parser[config_name].getint('PCA_number')
        pca = PCA(n_components=pca_number)
        data = pca.fit_transform(df_scale)
    else:
        data = df_scale
    unique_values = df_fair_column.unique()
    name2ix = util.dataprocess.name2index(unique_values)
    # 每个种群对应的坐标
    group_index = [0 for _ in range(len(unique_values))]
    for group in unique_values:
        group_index[name2ix[group]] = df_fair_column == group
    random_state = 40
    for _ in range(enum_number):
        for cluster_number in k_number:
            # 使用 k-means 聚类算法进行聚类
            kmeans = KMeans(n_clusters=cluster_number, random_state=random_state, init='k-means++', n_init=10)
            kmeans.fit(data)
            assign = kmeans.labels_
            centroids = kmeans.cluster_centers_
            function_cost = util.cost.caculate_cost(cluster_number,data,centroids,group_index,assign,name2ix,unique_values)
            result_list[cluster_number - k_number_min] += function_cost
            with open(filename_result, 'a', encoding='utf-8') as f:
                f.write("结果是：{}\n".format(function_cost))
                f.flush()
    for i in range(len(result_list)):
        result_list[i] /= enum_number
    for group in unique_values:

        plt.plot(k_number, result_list[:,name2ix[group]], label=group)
    plt.xlabel('k')
    plt.ylabel('cost')
    # 设置纵坐标轴范围为1到2，且严格限制在1到2之间
    # plt.ylim(1.6, 2.5)
    plt.legend()
    plt.savefig(fig_name)
    plt.show()

if __name__ == '__main__':
    main()

