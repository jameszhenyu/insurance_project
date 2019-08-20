#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']

plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore")
reload(sys)
sys.setdefaultencoding('utf8')


def build_rfr(data2):
    X = data2.loc[:, ('Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat')]
    y = data2.loc[:, 'Sun']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    rfc = RandomForestClassifier(n_estimators=25, random_state=0)
    rfc.fit(X_train, y_train.astype('int'))
    y_pred = rfc.predict(X_test)
    # mean = np.mean(np.array(data1['Sun']))
    # std = np.std(np.array(data1['Sun']))
    # y_pred = y_pred * std + mean
    # y_test = y_test * std + mean
    plt.figure()
    plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
    plt.plot(range(len(y_pred)), y_test, 'r', label="test")
    plt.legend(loc="upper right")
    plt.xlabel("numbers")
    plt.ylabel('values')
    plt.show()
    ratio = abs(y_pred - y_test) / y_test
    ratio = pd.DataFrame(ratio)
    ratio.to_csv(
        u'C:\\Users\\Administrator\\Desktop\\data待处理\\rfc_ratio1.csv',
        index=False)
    data3 = (ratio[ratio['Sun'] < 0.26])
    l = len(data3)

    len_all = len(ratio)

    ratio2 = (float(l) / len_all) * 100
    print('随机森林比率小于0.26的占比是：{:.2f}%'.format(ratio2))


def Normalization(data1):
    mean = []

    for index in data1.index:
        mean.append(np.mean(
            data1.loc[index, ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat']]))  # 计算每一行的平均数
    for index1 in data1.index:

        data1.loc[index1, :] = data1.loc[index1, :] - \
            mean[index1]  # 减去平均数，生成新数据

    return mean, data1


def Normalization_1():
    sam = []
    a = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']
    for i in a:
        y = data1.loc[:, i]

        ys = list(preprocessing.scale(y))

        sam.append(ys)

    data2 = pd.DataFrame(
        {
            'Mon': np.array(
                sam[0]), 'Tue': np.array(
                sam[1]), 'Wed': np.array(
                    sam[2]), 'Thur': np.array(
                        sam[3]), 'Fri': np.array(
                            sam[4]), 'Sat': np.array(
                                sam[5]), 'Sun': np.array(
                                    sam[6])})
    return data2


def build_lr_1(data2):
    X = data2.loc[:, ('Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat')]
    y = data2.loc[:, 'Sun']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    linreg = LinearRegression()
    model = linreg.fit(X_train, y_train)

    y_pred = linreg.predict(X_test)

    mean = np.mean(np.array(data1['Sun']))
    std = np.std(np.array(data1['Sun']))
    y_pred = y_pred * std + mean
    y_test = y_test * std + mean

    plt.figure()
    plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
    plt.plot(range(len(y_pred)), y_test, 'r', label="test")
    plt.legend(loc="upper right")
    plt.xlabel("numbers")
    plt.ylabel('values')
    plt.show()
    sum_mean = 0
    for i in range(len(y_pred)):
        sum_mean += (y_pred[i] - y_test.values[i]) ** 2
    sum_erro = np.sqrt(sum_mean / len(y_pred))

    print("RMSE by hand:", sum_erro)
    ratio = abs(y_pred - y_test) / y_test
    ratio = pd.DataFrame(ratio)
    ratio.to_csv(
        u'C:\\Users\\Administrator\\Desktop\\data待处理\\ratio1.csv',
        index=False)
    data3 = (ratio[ratio['Sun'] < 0.26])
    l = len(data3)

    len_all = len(ratio)

    ratio2 = (float(l) / len_all) * 100
    print('比率小于0.26的占比是：{:.2f}%'.format(ratio2))


def build_svr(data2):
    X = data2.loc[:, ('Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat')]
    y = data2.loc[:, 'Sun']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    svr = SVR(kernel='rbf')
    model = svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    mean = np.mean(np.array(data1['Sun']))
    std = np.std(np.array(data1['Sun']))
    y_pred = y_pred * std + mean
    y_test = y_test * std + mean
    # print y_pred
    plt.figure()
    plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
    plt.plot(range(len(y_pred)), y_test, 'r', label="test")
    plt.legend(loc="upper right")
    plt.xlabel("numbers")
    plt.ylabel('values')
    plt.show()
    sum_mean = 0
    for i in range(len(y_pred)):
        sum_mean += (y_pred[i] - y_test.values[i]) ** 2
    sum_erro = np.sqrt(sum_mean / len(y_pred))

    print("RMSE by hand:", sum_erro)
    ratio = abs(y_pred - y_test) / y_test
    ratio = pd.DataFrame(ratio)
    ratio.to_csv(
        u'C:\\Users\\Administrator\\Desktop\\data待处理\\ratio1.csv',
        index=False)
    data3 = (ratio[ratio['Sun'] < 0.26])
    l = len(data3)

    len_all = len(ratio)

    ratio2 = (float(l) / len_all) * 100
    print('svr比率小于0.26的占比是：{:.2f}%'.format(ratio2))


def build_svr_extra(data2):
    X = data2.loc[:, ('Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat')]
    y = data2.loc[:, 'Sun']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    svr = SVR()
    model = svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    # mean = np.mean(np.array(data1['Sun']))
    # std = np.std(np.array(data1['Sun']))
    # y_pred = y_pred * std + mean
    # y_test = y_test * std + mean
    # print y_pred
    plt.figure()
    plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
    plt.plot(range(len(y_pred)), y_test, 'r', label="test")
    plt.legend(loc="upper right")
    plt.xlabel("numbers")
    plt.ylabel('values')
    plt.show()
    sum_mean = 0
    for i in range(len(y_pred)):
        sum_mean += (y_pred[i] - y_test.values[i]) ** 2
    sum_erro = np.sqrt(sum_mean / len(y_pred))

    print("RMSE by hand:", sum_erro)
    ratio = abs(y_pred - y_test) / y_test
    ratio = pd.DataFrame(ratio)
    ratio.to_csv(
        u'C:\\Users\\Administrator\\Desktop\\data待处理\\ratio1.csv',
        index=False)
    data3 = (ratio[ratio['Sun'] < 0.26])
    l = len(data3)

    len_all = len(ratio)

    ratio2 = (float(l) / len_all) * 100
    print('svr比率小于0.26的占比是：{:.2f}%'.format(ratio2))


def build_poly(data2):
    X = data2.loc[:, ('Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat')]
    y = data2.loc[:, 'Sun']
    poly_reg = PolynomialFeatures(degree=2)

    X_poly = poly_reg.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=0.3, random_state=0)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)

    plt.figure()
    plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
    plt.plot(range(len(y_pred)), y_test, 'r', label="test")
    plt.legend(loc="upper right")
    plt.xlabel("numbers")
    plt.ylabel('values')
    plt.show()
    ratio = abs(y_pred - y_test) / y_test
    ratio = pd.DataFrame(ratio)
    ratio.to_csv(
        u'C:\\Users\\Administrator\\Desktop\\data待处理\\poly_ratio1.csv',
        index=False)
    data3 = (ratio[ratio['Sun'] < 0.26])
    l = len(data3)

    len_all = len(ratio)

    ratio2 = (float(l) / len_all) * 100
    print('多项式比率小于0.26的占比是：{:.2f}%'.format(ratio2))

    # data2.loc[:, 'poly_result'] = y_pred
    # data2['poly_result'].plot()
    # data2['Sun'].plot()
    # plt.legend()
    # plt.show()
    # data2.loc[:, 'ratio'] = abs(data2['poly_result'] - data2['Sun']) / data2['Sun']
    # data3 = data2[data2['ratio'] < 0.1]
    # ratio = float(len(data3)) / len(data2)
    # print ('小于0.1占比是：{:.2f}'.format(ratio))
    #
    #
    # data2.to_csv(u'C:\\Users\\Administrator\\Desktop\\data待处理\\ratio_poly.csv',index=False)


def build_ridge(data2):
    X = data2.loc[:, ('Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat')]
    y = data2.loc[:, 'Sun']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    ridge = RidgeCV(alphas=[0.2, 0.5, 0.8], cv=5)  # 5折交叉验证
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    plt.figure()
    plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
    plt.plot(range(len(y_pred)), y_test, 'r', label="test")
    plt.legend(loc="upper right")
    plt.xlabel("numbers")
    plt.ylabel('values')
    plt.show()
    ratio = abs(y_pred - y_test) / y_test
    ratio = pd.DataFrame(ratio)
    ratio.to_csv(
        u'C:\\Users\\Administrator\\Desktop\\data待处理\\ridge_ratio1.csv',
        index=False)
    data3 = (ratio[ratio['Sun'] < 0.26])
    l = len(data3)

    len_all = len(ratio)

    ratio2 = (float(l) / len_all) * 100
    print('岭回归比率小于0.26的占比是：{:.2f}%'.format(ratio2))

    # data2.loc[:, 'ridge'] = r
    # print len(r)
    # data2['ridge'].plot()
    # y_test.plot()
    # bins = np.arange(0, 101, 10)
    # plt.hist(data2['collections'], bins)


if __name__ == '__main__':
    data = pd.read_csv(
        u'C:\\Users\\Administrator\\Desktop\\data待处理\\保费收入_newest.csv')
    Mon = data[data['date'] == '星期一']['collections']
    Tue = data[data['date'] == '星期二']['collections']
    Wed = data[data['date'] == '星期三']['collections']
    Thur = data[data['date'] == '星期四']['collections']
    Fri = data[data['date'] == '星期五']['collections']
    Sat = data[data['date'] == '星期六']['collections']
    Sun = data[data['date'] == '星期日']['collections']

    data1 = pd.DataFrame({'Mon': np.array(Mon),
                          'Tue': np.array(Tue),
                          'Wed': np.array(Wed),
                          'Thur': np.array(Thur),
                          'Fri': np.array(Fri),
                          'Sat': np.array(Sat),
                          'Sun': np.array(Sun)})

    # data1.hist(bins=50, figsize=(20, 15))
    # plt.show()

    # for col in data1.columns:
    #     plt.hist(data1[col])
    #     plt.legend({col})
    #     plt.show()

    data3 = Normalization_1()  # 列标准化
    # build_svr(data3)#SVR模型
    #
    # mean, data2 = Normalization(data1)#行标准化

    # build_lr_1(data3)
    build_svr(data3)

    build_ridge(data1)  # 岭回归
    build_poly(data1)  # 线性多项式回归

