import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


class RF():
    def __init__(self, n):
        self.num = n
        self.forest = self.buildforest()
        self.wight = LinearRegression()

    def buildforest(self):
        ret = []
        for _ in range(self.num):
            ret += [DecisionTreeRegressor()]
        return ret

    def booststrap(self, dataset, num_samples):
        bootstrapping = []
        for i in range(num_samples):
            bootstrapping.append(np.floor(np.random.random() * len(X)))
        # 通过序号获得原始数据集中的数据
        D_1 = []
        for i in range(num_samples):
            D_1.append(dataset[int(bootstrapping[i])])
        D_1 = np.array(D_1)
        return D_1[:,1:], D_1[:,0]

    def fit(self, dataset):
        y = []
        data = []
        for cart in self.forest:
            x_train, y_train = self.booststrap(dataset, 20)
            cart.fit(x_train, y_train)
        for _ in range(100):
            x_test, y_test = self.booststrap(dataset, 1)
            cart_out = []
            y += [y_test]
            for cart in self.forest:
                cart_out += list(cart.predict(x_test))
            data += [cart_out]
        self.wight.fit(data, y)

    def predict(self, x):
        cart_out = []
        for cart in self.forest:
            cart_out += [list(cart.predict(x))]
        cart_out = np.array(cart_out).T
        return self.wight.predict(cart_out)


# 使用sin函数随机产生x，y数据
X = np.linspace(-3, 3, 50)
y = np.sin(X) + 0.1 * np.random.randn(X.shape[0])  # 产生数据y，增加随机噪声
data = np.array([[a,b]for a,b in zip(y, X)])

rf = RF(50)
rf.fit(data)

x_new = np.array([3, 1, 0.1]).reshape(-1,1)  # 自己随便写的一个样本数据
print(rf.predict(x_new))

