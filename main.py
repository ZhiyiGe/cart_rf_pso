import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_excel('data.xlsx', index_col=0)
data = pd.DataFrame(data)
print(type(data))
print(data.head())
columns = data.columns
plt.scatter(data.index, data['硅含量'])
plt.show()


# 定义3σ法则识别异常值函数
def three_sigma(data):
    Ser = data.copy(deep=True)
    '''
    Ser1：表示传入DataFrame的某一列。
    '''
    for column in columns:
        Ser1 = Ser[column]
        mean = Ser1.mean()
        std = Ser1.std()
        rule = (mean - 3 * std > Ser1) | (mean + 3 * std < Ser1)
        index = np.arange(Ser1.shape[0])[rule]
        Ser1.iloc[index] = np.NAN
    return Ser


data_2 = three_sigma(data)
print([data_2[column].isnull().sum() for column in columns])


def pretreatment(data):
    """
    预处理数组
    后位填充，使连续非空值段的最小长度=3
    """
    Ser = data.copy(deep=True)
    for column in columns:
        ser1 = Ser[column]
        mean = ser1.mean()
        bool_ = ser1.isnull()  # 获取布尔数组，如果不是nan元素为True
        ser1[bool_] = mean
    return Ser


data_3 = pretreatment(data_2)
print([data_3[column].isnull().sum() for column in columns])

corr_value = data_3.corr()['硅含量']
print(corr_value)
print(type(corr_value))
Data1 = data_3[columns[abs(corr_value) > 0.25]]
Data2 = data_3[columns[abs(corr_value) > 0.33]]
corr_matrix = Data2.corr()
corr_value2 = Data2.corr()['硅含量']


def Couplingindex(matrix):
    c = matrix.columns
    matrix_value = abs(matrix.values)
    n = len(matrix_value)
    value = []
    for i in range(n):
        value += [(sum(matrix_value[i][:i]) + sum(matrix_value[i][i + 1:])) / (n - 1)]
    return pd.Series(value, index=c, name='耦合性指数')


tmp = Couplingindex(corr_matrix)
plt.scatter(tmp, abs(corr_value2))
plt.show()
print(tmp.sort_values().index)
Data3 = Data2.drop(['压差', '鼓风动能'], axis=1)
X = np.array(Data3.drop(['硅含量'], axis=1))
y = np.array(Data3['硅含量'])
x_train, x_test, y_train, y_test = train_test_split(X,y)
rf = RandomForestRegressor()
print(x_train.shape,y_train.shape)
rf.fit(x_train,y_train)
y_predict = rf.predict(x_test)
plt.plot(y_test)
plt.plot(y_predict)
plt.legend(['test', 'predict'])
plt.show()

# # 筛选所需数据
# data = data.loc[:, ['时间戳', '采集项名称', '采集项值']]
# data.columns = ['time', 'name', 'value']
# pd.value_counts(data['name'])
# # 以字典的形式存储数据
# grouped = data.groupby('name')
# data_grouped = {}
# for name, group in grouped:
#     data_grouped[name] = group.drop(columns='name').reset_index(drop=True)
#
#
# # 绘制数据分布图
# def paint(data, name, column):
#     mean = data_grouped[name][column].mean()
#     std = data_grouped[name][column].std()
#     length = len(data[name][column])
#     data[name].plot(figsize=(100, 20), style='bo--')
#     plt.hlines(mean + 3 * std, 0, length, color="red")  # 横线
#     plt.hlines(mean, 0, length, color="red")  # 横线
#     plt.hlines(mean - 3 * std, 0, length, color="red")  # 横线
#     plt.show()
#
#
# for name in data_grouped:
#     print(name)
#     paint(data_grouped, name, 'value')
#
#

#
#
# data_grouped_nan = {}
# for name in data_grouped:
#     data_grouped_nan[name] = three_sigma(data_grouped[name], 'value')
#
# for name in data_grouped_nan:
#     print(data_grouped_nan[name].isnull().sum())
#
# for name in data_grouped_nan:
#     print(name)
#     paint(data_grouped_nan, name, 'value')
#
#     from itertools import groupby
#
#
#     def pretreatment(ser):
#         """
#         预处理数组
#         后位填充，使连续非空值段的最小长度=3
#         """
#         bool_ = ser.notnull()  # 获取布尔数组，如果不是nan元素为True
#         index = np.arange(ser.shape[0])[bool_]
#         fun = lambda x: x[1] - x[0]
#         for k, g in groupby(enumerate(index), fun):
#             lst = [j for i, j in g]
#             length = max(lst) - min(lst)
#             if length == 1:  # 连续非空字段的长度为2
#                 ser[min(lst) - 1] = ser[min(lst)]
#             elif length == 0:  # 连续非空字段的长度为1
#                 ser[min(lst) - 1] = ser[min(lst) + 1] = ser[min(lst)]
#         return ser
#
#
#     def get_single_fill(ser):
#         """
#         单向的移动填充
#         同时返回nan处的权重
#         """
#         new_ser_left2_right = ser.copy()
#         weight = pd.Series(np.ones_like(ser))  # 权重初始化为1
#         bool_ = ser.isna()  # 获取布尔数组，如果是nan元素为True
#         count = 1  # nan连续性计数
#         for idx in range(len(ser)):
#             if idx >= 3 and bool_.iloc[idx]:
#                 weight[idx] = 1 / count
#                 count += 1
#                 # 当前的nan填充数值为前三元素的平均
#                 new_ser_left2_right.iloc[idx] = new_ser_left2_right.iloc[idx - 3:idx].mean()
#             else:
#                 count = 1
#         return new_ser_left2_right, weight
#
#
# # 双向均值填补
# def duplex_fill(df_pure, param):
#     """
#     先调用three_sigma函数剔除异常值，然后使用双向滑动平均法填充剔除后的缺失
#     :param df_pure: pd.DataFrame类型
#     :param param ：str 类，指标名
#     :return pd.DataFrame类型, 返回处理好的数据表
#     """
#     ser = df_pure[param]  # 待处理列的数据
#     ser = pretreatment(ser)
#     new_ser_left2_right, left_weight = get_single_fill(ser)
#     new_ser_right2_left, right_weight = get_single_fill(ser[::-1])  # 反过来计算缺失处的预测值和权重
#     new_ser_right2_left = new_ser_right2_left[::-1]  # 反转回去
#     right_weight = right_weight[::-1]  # 反转回去
#
#     # 计算归一化权重 ，同时注意到pd.serial 的索引是乱的，要以np.array类型计算
#     left_weight_normal = left_weight.values / (left_weight.values + right_weight.values)
#     right_weight_normal = right_weight.values / (left_weight.values + right_weight.values)
#
#     # 加权平均
#     ser_ans = left_weight_normal * new_ser_left2_right.values + right_weight_normal * new_ser_right2_left.values
#     df_pure[param] = ser_ans
#     return df_pure
#
#
# for name in data_grouped_nan:
#     data_grouped_nan[name] = duplex_fill(data_grouped_nan[name], 'value')
#
# for name in data_grouped_nan:
#     print(name)
#     paint(data_grouped_nan, name, 'value')
# for name in data_grouped_nan:
#     print(data_grouped_nan[name].notnull().sum())
#
# # 合并各表
# data_output = pd.DataFrame()
# len(data_output)
# for name in data_grouped_nan:
#     data_grouped_nan[name]['name'] = name
#     if len(data_output) == 0:
#         data_output = data_grouped_nan[name]
#         continue
#     data_output = pd.concat([data_output, data_grouped_nan[name]], axis=1)
# print(data_output.shape)
# # 排序并重置索引
# data_output.reset_index(drop=True).head()
# # 写入excel
# data_output.to_excel('data/xichang2#_solved.xlsx', index=False)
