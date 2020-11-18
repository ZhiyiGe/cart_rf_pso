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