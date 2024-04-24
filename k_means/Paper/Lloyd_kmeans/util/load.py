import pandas as pd
import os
import configparser
import codecs
# 加载数据集
def load_data(root,name):
    path = os.path.join(root,name)
    df = pd.read_csv(path,header=0)
    return df

# 加载配置文件
def load_configuration(root, name):
    name = name + '.ini'
    path = os.path.join(root, name)
    config = configparser.ConfigParser()
    with codecs.open(path, 'r', encoding='utf-8') as f:
        config.read_file(f)
    return config
