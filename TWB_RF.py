import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# k、m、t、n、d 分别代表特征子集容量、叶节点包含最少样本数、森林规模、采样次数、最大层深
K = 5
M = 5
T = 30
N = 4
D = 10


class data_process:
    def __init__(self, path):
        self.data = pd.read_excel(path, index_col=0)  # 读取数据
        self.data_old = self.data.copy(deep=True)  # 原始数据
        self.columns = self.data.columns  # 参数名称

    # 两参数之间分布的散点图
    def scatter(self, xlabel, ylabel):
        plt.scatter(self.data_old[xlabel], self.data_old[ylabel])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    # 绘制参数分布直方图
    def hist(self, name):
        plt.hist(self.data_old[name], edgecolor="white")
        # 显示横轴标签
        plt.xlabel("区间")
        # 显示纵轴标签
        plt.ylabel("个数")
        # 显示图标题
        plt.title(name)
        plt.show()

    # 定义3σ法则识别异常值函数
    def three_sigma(self):
        index = np.arange(self.data.shape[0])
        # Ser1：表示传入DataFrame的某一列。
        drop = np.array([])  #
        for column in self.columns:
            Ser1 = self.data[column]
            mean = Ser1.mean()
            std = Ser1.std()
            rule = (mean - 3 * std > Ser1) | (mean + 3 * std < Ser1)
            drop = np.hstack((drop, index[rule]))
        self.data = self.data.drop(drop)  # 剔除所有异常数据样本

    # 均值填充缺失值
    def fill(self):
        Ser = self.data
        for column in self.columns:
            ser1 = Ser[column]
            mean = ser1.mean()
            bool_ = ser1.isnull()  # 获取布尔数组，如果不是nan元素为True
            ser1[bool_] = mean

    # 计算耦合性指数矩阵
    def Couplingindex(self, matrix):
        c = matrix.columns
        matrix_value = abs(matrix.values)
        n = len(matrix_value)
        value = []
        for i in range(n):
            value += [(sum(matrix_value[i][:i]) + sum(matrix_value[i][i + 1:])) / (n - 1)]
        return pd.Series(value, index=c, name='耦合性指数')

    # 处理数据
    def process(self):
        self.fill()  # 填充缺失值
        self.three_sigma()  # 3sigmod异常值检测
        # self.fill()  # 填充缺失值
        self.corr_value_data = self.data.corr()['硅含量']  # 计算硅含量与其他参数之间的相关系数
        Data1 = self.data[self.columns[abs(self.corr_value_data) > 0.25]]  # 相关系数大于0.25的组成Data1
        Data2 = self.data[self.columns[abs(self.corr_value_data) > 0.33]]  # 相关系数大于0.25的组成Data1
        self.corr_matrix_Data2 = Data2.corr()  # 计算Data2的相关系数矩阵
        self.corr_value_Data2 = Data2.corr()['硅含量'] # 计算Data2中其他参数与硅含量的相关系数
        self.coupleid = self.Couplingindex(self.corr_matrix_Data2) # 计算Data2参数间的耦合指数
        Data3 = Data2.drop(['压差', '鼓风动能'], axis=1)  # 剔除与其他特征耦合性最大的特征

        return Data1, Data2, Data3


class RF:
    def __init__(self,
                 n_estimators, # 深林规模
                 n_samples, # 每个决策树训练所需的数据
                 n_features=2, # 特征子集容量
                 max_depth=None, # 最大深度
                 min_samples_leaf=1,  # 最少叶节点样本数
                 min_samples_split=10):# 内部节点再划分所需最小样本数
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_samples = n_samples
        self.n_features = n_features
        self.min_samples_split = min_samples_split

        # 中间变量
        self.feature_index = {} # 每棵决策树随机抽取到的特征的索引
        self.num = n_estimators # 森林规模
        self.forest = self.buildforest() # 创建随机森林
        self.wight = np.zeros(self.num) # 决策树的权重值
        self.rmse_tree = np.zeros(self.num) # 决策树的均方根误差
        self.train_dataset = None # 随机森林的训练样本集

        # 模型性能指标
        self.mse = None # 随机森立的均方误差
        self.rmse = None # 随机森林的均方根误差

    # 创建随即森林，并以列表的形式返回
    def buildforest(self):
        ret = []
        for _ in range(self.num):
            ret += [DecisionTreeRegressor(max_depth=self.max_depth,
                                          min_samples_leaf=self.min_samples_leaf,
                                          min_samples_split=self.min_samples_split)]
        return ret

    # 为每棵决策树随机分配对应数量的特征
    def feature_choose(self):
        for index in range(self.num):
            n = self.train_dataset.shape[1] - 1
            self.feature_index[index] = []
            while len(self.feature_index[index]) < self.n_features:
                tmp = int(np.floor(np.random.random() * n))
                if tmp not in self.feature_index[index]:
                    self.feature_index[index].append(tmp)

    # 队训练数据集进行采样 返回特征集和目标值
    def booststrap(self, dataset, num_samples):

        bootstrapping = []
        n = len(dataset)
        while len(bootstrapping) < num_samples:
            tmp = int(np.floor(np.random.random() * n))
            # if tmp not in bootstrapping:
            bootstrapping.append(tmp)
        # 通过序号获得原始数据集中的数据
        D_1 = []
        for i in range(num_samples):
            D_1.append(dataset[bootstrapping[i]])
        D_1 = np.array(D_1)
        return D_1[:, :-1], D_1[:, -1]

    # 对随机森林模型纪念性训练
    def fit(self, x_train, y_train):
        y_train = np.reshape(y_train, (-1, 1))
        self.train_dataset = np.concatenate((x_train, y_train), axis=1)
        self.feature_choose() # 为决策树随机分配特征
        # 对每棵决策树进行训练
        for i in range(self.num):
            x, y = self.booststrap(self.train_dataset, self.n_samples)
            x = x[:, self.feature_index[i]]
            x_train, x_test, y_train, y_test = train_test_split(x, y)
            self.forest[i].fit(x_train, y_train)
            # print("  trained  " + str(i) + '  tree')
            r = np.sqrt(mean_squared_error(y_test, self.forest[i].predict(x_test)))
            self.rmse_tree[i] = r
            # print("  rmse  " + str(i) + '  tree')

        # 计算决策数的权重值
        for i in range(self.num):
            w = (self.rmse_tree.sum() - self.rmse_tree[i]) / ((self.num - 1) * self.rmse_tree.sum())
            self.wight[i] = w
        self.evaluatemodel()

    # 对随机森林模型进行评估，计算均方误差及均方根误差
    def evaluatemodel(self):
        x, y = self.train_dataset[:, :-1], self.train_dataset[:, -1]
        y_p = self.predict(x)
        self.mse = mean_squared_error(y, y_p)
        self.rmse = np.sqrt(self.mse)

    # 使用训练后的随机森林模型对数据进行预测
    def predict(self, x):
        cart_out = []
        for i in range(self.num):
            x_test = x[:, self.feature_index[i]]
            cart_out += [list(self.forest[i].predict(x_test))]
        cart_out = np.array(cart_out).T
        return cart_out.dot(self.wight)


# k、m、t、n、d 分别代表特征子集容量、叶节点包含最少样本数、森林规模、采样次数、最大层深
# PSO优化函数
def PSO_optimize_func(k, m, t, n, d):
    processer = data_process('data.xlsx')
    Data1, Data2, Data3 = processer.process()
    X = Data2.drop(['硅含量'], axis=1).values
    y = Data2['硅含量'].values

    x_train, x_test, y_train, y_test = train_test_split(X, y)
    rf = RF(t, n_samples=n * 50, n_features=k, min_samples_leaf=m, max_depth=d)
    rf.fit(x_train, y_train)
    return rf.rmse


if __name__ == '__main__':
    processer = data_process('data.xlsx')
    # processer.scatter('风压', '硅含量')
    # processer.hist('硅含量')
    Data1, Data2, Data3 = processer.process()
    X = Data3.drop(['硅含量'], axis=1).values
    y = Data3['硅含量'].values
    x_train, x_test, y_train, y_test = train_test_split(X, y)

    rf = RF(T, n_samples=N * 50, n_features=K, min_samples_leaf=M, max_depth=D)
    rf.fit(x_train, y_train)
    print('mse:', rf.mse, 'rmse:', rf.rmse)
    y_predict = rf.predict(x_test)
    plt.plot(y_test)
    plt.plot(y_predict)
    plt.legend(['test', 'predict'])
    plt.show()
