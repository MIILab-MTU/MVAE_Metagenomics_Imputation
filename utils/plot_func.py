import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib

from matplotlib.ticker import FuncFormatter
from matplotlib import rc
from sklearn import metrics
from glob import glob
from sklearn.linear_model import LinearRegression
from scipy import stats



def plot_residual_vs_gt(y_true, y_pred, title, save_path, x_interval=20000, y_interval=1000, fontsize=20):
    def formatnum(x, pos):
        return '$%0.1f$x$10^{5}$' % (x / 100000)

    reg = LinearRegression().fit(np.array(y_true).reshape(-1, 1), y_pred)
    print(reg.coef_)
    print(reg.intercept_)

    residuals = []
    for i in range(len(y_true)):
        residuals.append(y_pred[i] - (y_true[i]*reg.coef_[0] + reg.intercept_))

    f, ax = plt.subplots(1, 1)
    f.set_size_inches(8, 8)
    plt.title(title, fontdict={"fontsize": fontsize})
    formatter = FuncFormatter(formatnum)

    max_value = max(np.max(y_true), np.max(y_pred))
    min_value = max(np.min(y_true), np.min(y_pred))

    ax.set_xlim(min_value - x_interval, max_value + x_interval)
    ax.set_ylim(np.mean(residuals) - np.std(residuals) - y_interval,
                np.mean(residuals) + np.std(residuals) + y_interval)
    ax.xaxis.set_major_formatter(formatter)
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start + 20000, end, 40000))

    ax.scatter(y_true, residuals, s=10)
    ax.tick_params(axis='both', which='major', labelsize=fontsize * 0.75)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=90)

    plt.xlabel(r"ground truth ($N$)", fontdict={"fontsize": fontsize})
    plt.ylabel(r"volume residuals ($N$)", fontdict={"fontsize": fontsize})
    plt.savefig(save_path)
    plt.close()


def p_vals_per_coef(lm, y, x):
    import pandas as pd
    x = np.array(x).reshape(-1, 1)
    params = np.append(lm.intercept_, lm.coef_)
    predictions = lm.predict(x)

    newX = pd.DataFrame({"Constant": np.ones(len(x))}).join(pd.DataFrame(x))
    MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX.columns))

    # Note if you don't want to use a DataFrame replace the two lines above with
    # newX = np.append(np.ones((len(X),1)), X, axis=1)
    # MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

    var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b

    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]

    sd_b = np.round(sd_b, 3)
    ts_b = np.round(ts_b, 3)
    p_values = np.round(p_values, 3)
    params = np.round(params, 4)

    myDF3 = pd.DataFrame()
    myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3["Probabilities"] = [params, sd_b, ts_b,
                                                                                                  p_values]
    print(myDF3)


def plot_points(y_true, y_pred, title, save_path, interval=100, fontsize=20):
    def formatnum(x, pos):
        return '$%0.1f$x$10^{5}$' % (x / 100000)

    f, ax = plt.subplots(1, 1)
    f.set_size_inches(8.5, 8)
    plt.title(title, fontdict={"fontsize": fontsize})
    formatter = FuncFormatter(formatnum)

    ax.scatter(y_true, y_pred, s=fontsize)
    max_value = max(np.max(y_true), np.max(y_pred))
    min_value = max(np.min(y_true), np.min(y_pred))

    # ax.set_xlim(min_value - interval, max_value + interval)
    # ax.set_ylim(min_value - interval, max_value + interval)
    # ax.xaxis.set_major_formatter(formatter)
    start, end = ax.get_xlim()
    # ax.xaxis.set_ticks(np.arange(start, end, interval))
    # ax.yaxis.set_ticks(np.arange(start, end, interval))

    #ax.xaxis.label.set_size(fontsize*2)
    ax.tick_params(axis='both', which='major', labelsize=fontsize*0.75)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=90)
    # ax.yaxis.set_major_formatter(formatter)

    # y lines
    gt_lines = np.linspace(min_value - interval, max_value + interval, interval)
    plt.plot(gt_lines, gt_lines, linestyle='--', color='r', label="GT")

    # bin lines
    reg = LinearRegression().fit(np.array(y_true).reshape(-1, 1), y_pred)
    print(reg.coef_)
    print(reg.intercept_)
    print(reg)
    plt.plot(gt_lines, reg.coef_[0]*gt_lines+reg.intercept_, linestyle='--', color='g', label="Prediction")

    if reg.intercept_ > 0:
        plt.text(0.50*(max_value+min_value), 0.48*(max_value+min_value),
                 "Y=%.4fX+%4.4f" % (reg.coef_[0], reg.intercept_),
                 fontdict={"fontsize": fontsize})
    else:
        plt.text(0.50 * (max_value + min_value), 0.48 * (max_value + min_value),
                 "Y=%.4fX - %4.4f" % (reg.coef_[0], abs(reg.intercept_)),
                 fontdict={"fontsize": fontsize})

    plt.legend(loc=2, prop={'size': fontsize})
    from matplotlib import rc
    plt.xlabel(r"ground truth", fontdict={"fontsize": fontsize})
    plt.ylabel(r"model prediction", fontdict={"fontsize": fontsize})
    # plt.close()
    plt.savefig(save_path)
    plt.close()


def plot_relative_error(y_true, y_pred, title, save_path, x_interval=0, y_interval=2.0, fontsize=20):
    def formatnum(x, pos):
        return '$%0.1f$x$10^{5}$' % (x / 100000)

    relative_error = []
    for i in range(len(y_true)):
        relative_error.append(100 * (y_pred[i] - y_true[i]) / y_true[i])
        print("RE {} : {}".format(i, 100 * (y_pred[i] - y_true[i]) / y_true[i]))

    # print("RE MIN = {}, MAX = {}".format(min(relative_error), max(relative_error)))

    f, ax = plt.subplots(1, 1)
    f.set_size_inches(8, 8)
    plt.title(title, fontdict={"fontsize": fontsize})
    formatter = FuncFormatter(formatnum)

    max_value = max(np.max(y_true), np.max(y_pred))
    min_value = max(np.min(y_true), np.min(y_pred))

    ax.set_xlim(min_value - x_interval, max_value + x_interval)
    ax.set_ylim(-3 - y_interval, 3 + y_interval)
    ax.xaxis.set_major_formatter(formatter)
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end))
    ax.tick_params(axis='both', which='major', labelsize=fontsize * 0.75)

    x = np.linspace(min_value - x_interval, max_value + x_interval, x_interval)

    ax.scatter(np.array(y_true), np.array(relative_error),
               marker="*", s=160, c=np.arange(len(np.array(y_true))), cmap="hsv")
    ax.scatter(np.array(y_true), np.array(relative_error), 
               marker="o", s=160, c=np.arange(len(np.array(y_true))), cmap="hsv")

    plt.plot(x, [np.mean(relative_error)] * x_interval, linestyle='--', color='r', label="mean = %0.4f" % np.mean(relative_error))
    plt.plot(x, [np.mean(relative_error) + np.std(relative_error)] * x_interval, linestyle=':', color='r',
             label="mean+std = %0.4f" % (np.mean(relative_error)+np.std(relative_error)))
    plt.plot(x, [np.mean(relative_error) - np.std(relative_error)] * x_interval, linestyle=':', color='r',
             label="mean-std = %0.4f" % (np.mean(relative_error)-np.std(relative_error)))

    plt.legend(loc="lower right", prop={'size': fontsize*0.75})
    plt.xlabel(r"ground truth ($N$)", fontdict={"fontsize": fontsize})
    plt.ylabel(r"relative error (%)", fontdict={"fontsize": fontsize})
    plt.savefig(save_path)
    plt.close()

