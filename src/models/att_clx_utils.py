import numpy as np
import pandas as pd
from sklearn import metrics
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import matplotlib.pyplot as plt


def adf_test(temp):
    """p-value > 0.562 or Critical Value(1%) > -3.44 implies non-stationary."""
    t = adfuller(temp)
    output = pd.DataFrame(index=['Test Statistic Value', 'p-value', 'Lags Used',
                                 'Number of Observations Used', 'Critical Value(1%)',
                                 'Critical Value(5%)', 'Critical Value(10%)'], columns=['value'])
    output['value']['Test Statistic Value'] = t[0]
    output['value']['p-value'] = t[1]
    output['value']['Lags Used'] = t[2]
    output['value']['Number of Observations Used'] = t[3]
    output['value']['Critical Value(1%)'] = t[4]['1%']
    output['value']['Critical Value(5%)'] = t[4]['5%']
    output['value']['Critical Value(10%)'] = t[4]['10%']
    print(output)


def acf_pacf_plot(seq, acf_lags=20, pacf_lags=20):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(seq, lags=acf_lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(seq, lags=pacf_lags, ax=ax2)
    plt.show()


def create_dataset(dataset, look_back=20):
    """
    Exactly as per reference: generates sliding windows.
    Returns: TrainX (samples, time_steps, features), Train_Y (samples, features)
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    TrainX = np.array(dataX)
    Train_Y = np.array(dataY)
    return TrainX, Train_Y


def evaluation_metric(y_test, y_hat):
    MSE = metrics.mean_squared_error(y_test, y_hat)
    RMSE = MSE ** 0.5
    MAE = metrics.mean_absolute_error(y_test, y_hat)
    R2 = metrics.r2_score(y_test, y_hat)
    print('MSE: %.5f' % MSE)
    print('RMSE: %.5f' % RMSE)
    print('MAE: %.5f' % MAE)
    print('R2: %.5f' % R2)


def NormalizeMult(data):
    """
    Percentile-based normalization (0 to 100 percentile).
    Returns normalized data and the normalization parameters (mins/maxes).
    """
    data = np.array(data)
    normalize = np.arange(2 * data.shape[1], dtype='float64')
    normalize = normalize.reshape(data.shape[1], 2)

    for i in range(0, data.shape[1]):
        col_data = data[:, i]
        listlow, listhigh = np.percentile(col_data, [0, 100])
        normalize[i, 0] = listlow
        normalize[i, 1] = listhigh
        delta = listhigh - listlow
        if delta != 0:
            data[:, i] = (data[:, i] - listlow) / delta

    return data, normalize


def NormalizeMultUseData(data, normalize):
    """Applies existing normalization parameters to new data."""
    data = np.array(data)
    for i in range(0, data.shape[1]):
        listlow = normalize[i, 0]
        listhigh = normalize[i, 1]
        delta = listhigh - listlow
        if delta != 0:
            data[:, i] = (data[:, i] - listlow) / delta
    return data


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """Converts time series to a supervised learning format."""
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def prepare_data(series, n_test, n_in, n_out):
    """Slices the supervised data into training and testing sets based on hardcoded indices."""
    values = series.values
    supervised_data = series_to_supervised(values, n_in, n_out)
    # The reference uses fixed indices 3499/3500 for the split
    train = supervised_data.iloc[:3500, :]
    test = supervised_data.iloc[3500:, :]
    return train, test