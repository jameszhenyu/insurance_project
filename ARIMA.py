#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import AR
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings("ignore")
reload(sys)
sys.setdefaultencoding('utf8')

#中文乱码处理
plt.rcParams["font.sans-serif"]=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False



# 移动平均图
def draw_trend(timeseries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeseries.rolling(window=size).mean()
    # 对size个数据移动平均的方差
    rol_std = timeseries.rolling(window=size).std()

    timeseries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_std.plot(color='black', label='Rolling standard deviation')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()


def draw_ts(timeseries):
    f = plt.figure(facecolor='white')
    timeseries.plot(color='blue')
    plt.show()


# Dickey-Fuller test:
def teststationarity(ts):
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput


def diff(timeSeries):
    fig = plt.figure(facecolor='white', figsize=(12, 8))
    #
    ax1 = fig.add_subplot(221)
    date = timeSeries
    date_plot = date.plot()
    # diff_1
    ax2 = fig.add_subplot(222)
    diff_1 = timeSeries.diff(1)
    date_diff_1_plot = diff_1.plot()
    # diff_2
    ax3 = fig.add_subplot(223)
    diff_2 = timeSeries.diff(2)
    date_diff_2_plot = diff_2.plot()
    # diff_3
    ax4 = fig.add_subplot(224)
    diff_3 = timeSeries.diff(3)
    date_diff_3_plot = diff_3.plot()
    plt.show()

    return diff_1, diff_2, diff_3


# def decompose(timeseries):
#     # 返回包含三个部分 trend（趋势部分） ， seasonal（季节性部分） 和residual (残留部分)
#     decomposition = seasonal_decompose(timeseries,frep=1)
#
#     trend = decomposition.trend
#     seasonal = decomposition.seasonal
#     residual = decomposition.resid
#
#     plt.subplot(411)
#     plt.plot(data, label='Original')
#     plt.legend(loc='best')
#     plt.subplot(412)
#     plt.plot(trend, label='Trend')
#     plt.legend(loc='best')
#     plt.subplot(413)
#     plt.plot(seasonal, label='Seasonality')
#     plt.legend(loc='best')
#     plt.subplot(414)
#     plt.plot(residual, label='Residuals')
#     plt.legend(loc='best')
#     plt.tight_layout()
#     plt.show()
#     return trend, seasonal, residual
def draw_acf_pacf(timeSeries):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(timeSeries, lags=40, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(timeSeries, lags=40, ax=ax2)
    plt.show()

def plot_curve(true_data, predicted_data):
    plt.plot(true_data, label='True data')
    plt.plot(predicted_data, label='Predicted data')
    plt.legend()
    plt.savefig('result.png')
    plt.show()






if __name__ == '__main__':
    df = pd.read_csv(u'C:\\Users\\Administrator\\Desktop\\data待处理\\保费收入.csv', index_col='year')
    df.index = pd.to_datetime(df.index)
    data=df['collections']




    data1 = np.log(data)

    draw_trend(data,12)#查看原始数据的均值和方差
    print (teststationarity(data))
    diff(data)


    diff_1 = data.diff(1)
    diff_1.dropna(inplace=True)
    diff_1_1 = diff_1.diff(1)
    diff_1_1.dropna(inplace=True)
    print teststationarity(diff_1_1)
    rol_mean = data1.rolling(window=12).mean()
    rol_mean.dropna(inplace=True)
    ts_diff_1 = rol_mean.diff(1)
    ts_diff_1.dropna(inplace=True)
    draw_acf_pacf(ts_diff_1)

    # model = ARIMA(ts_diff_1, order=(1, 1, 11))
    model=AR(ts_diff_1, order=())


    result_arima = model.fit(disp=-1, method='css')
    predict_ts = result_arima.predict()
    print predict_ts

    # 一阶差分还原
    diff_shift_ts = ts_diff_1.shift(1)
    diff_recover_1 = predict_ts.add(diff_shift_ts)
    # 再次一阶差分还原
    rol_shift_ts = rol_mean.shift(1)
    diff_recover = diff_recover_1.add(rol_shift_ts)
    # 移动平均还原
    rol_sum = data1.rolling(window=11).sum()
    rol_recover = diff_recover * 12 - rol_sum.shift(1)
    # 对数还原
    log_recover = np.exp(rol_recover)
    log_recover.dropna(inplace=True)
    result=pd.DataFrame(log_recover.values)

    print result
    print ("----------------------------------")
    df1 = pd.read_csv(u'C:\\Users\\Administrator\\Desktop\\data待处理\\保费收入.csv')
    #print np.array(raw['collections'])-np.array(result[0])
    df1.drop(index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], inplace=True)
    print df1
    result['real'] = df1['collections'].values
   # print(result)
    ratio=abs(result['real'] - result[0]) / result['real']
    ratio=pd.DataFrame(ratio)

    r=float(len(ratio[ratio[0]<0.25]))/len(ratio)
    print r
    ratio.to_csv(u'C:\\Users\\Administrator\\Desktop\\data待处理\\1111ratio.csv',index=False)





