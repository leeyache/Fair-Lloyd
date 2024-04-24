import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# 对数据进行缺省值处理
def fill_na(df):
    # 对数值型数据进行缺省处理
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for column in numeric_columns:
        df[column].fillna(df[column].mean(), inplace=True)

    # 对非数值型数据进行缺省处理
    non_numeric_columns = df.select_dtypes(include=['object']).columns
    for column in non_numeric_columns:
        df[column].fillna(df[column].mode()[0], inplace=True)


# 对除了进行公平性度量的其他列进行独热编码处理
def one_hot(df,fair_column):
    # 对除了race之外的非数值型列进行独热编码
    non_numeric_columns = df.select_dtypes(include=['object']).drop(columns=fair_column).columns
    df_encoded = pd.get_dummies(df, columns=non_numeric_columns)
    return df_encoded

# 去除聚类中不使用的列
def drop_columns(df, drop_columns):
    if drop_columns is not None:
        df_for_clustering = df.drop(columns=drop_columns)
    else:
        df_for_clustering = df
    return df_for_clustering


# 对数据进行正则化处理
def scaler(df_for_clustering):
    # 使用MinMaxScaler对数据进行正则化处理
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_for_clustering)
    return scaled_data

#名字到索引的映射
def name2index(unique_values):
    name2ix = {}
    for i, name in enumerate(unique_values):
        name2ix[name] = i
    return name2ix
