from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import cm
import itertools
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from math import sqrt

if __name__ == '__main__':
    feature_names = ["Log GDP per capita", "Social support", "Healthy life expectancy at birth", "Freedom to make life choices", "Perceptions of corruption"]
    df = pd.read_csv("world-happiness-report.csv")
    dt = pd.DataFrame(df, columns=feature_names)
    y = pd.DataFrame(df, columns=["Life Ladder"])

    # Data Interpolation
    # Method 1 - using fillna (ffill/pad or bfill)
    dt_copy_1 = dt.copy()
    dt_copy_1 = dt_copy_1.fillna(method="ffill")

    # Method 2 - using Pandas interpolate
    dt_copy_2 = dt.copy()
    dt_copy_2 = dt_copy_2.interpolate()

    # Method 3 - KNNImputer
    dt_copy_3 = dt.copy()
    imputer = KNNImputer(n_neighbors=3)
    imputed = imputer.fit_transform(dt_copy_3)
    dt_copy_3 = pd.DataFrame(imputed, columns=dt.columns)

    combinations = []
    # Itertools
    for length in range(1, len(feature_names) + 1):
        for subset in itertools.combinations(feature_names, length):
            combinations.append(list(subset))
    print('\n\n\n\n\n')
    print(combinations)
    print(len(combinations))
    # print("FEATURES ---------")
    # print(dt)
    #
    # print("MEASUREMENT ---------")
    # print(y)

    # X_train, X_test, y_train, y_test = train_test_split(dt_copy_3, y, test_size=0.2)

    # # print("MEASUREMENT ---------")
    # # print(y_test["Life Ladder"])

    # rows = y["Life Ladder"].count()
    # max_k = int(rows / 10)

    # for combination in combinations:
    #     train_accuracies = []
    #     test_accuracies = []
    #     rmse_values = []

    #     for i in range(1, max_k):  # ensure this reaches 1949 total instead of 1948
    #         knr = KNeighborsRegressor(i)

    #         x_train = X_train[combination]

    #         knr.fit(x_train, y_train["Life Ladder"])

    #         x_test = X_test[combination]

    #         train_accuracies.append(knr.score(x_train, y_train["Life Ladder"]))
    #         test_accuracies.append(knr.score(x_test, y_test["Life Ladder"]))

    #         y_predict = knr.predict(x_test)
    #         error = sqrt(mean_squared_error(y_test, y_predict))
    #         rmse_values.append(error)
    #         # y_predict = knr.predict(x_test)

    #     plt.figure()
    #     plt.subplot(1, 3, 1)
    #     plt.plot(range(1, max_k), train_accuracies, label="Training Accuracy")

    #     plt.subplot(1, 3, 2)
    #     plt.plot(range(1, max_k), test_accuracies, label="Test Accuracy")

    #     plt.subplot(1, 3, 3)
    #     plt.plot(range(1, max_k), rmse_values, label="Test accuracy 2")

    #     plt.show()
    #     break


    # print(test_accuracies.index(max(test_accuracies)))


    # for i in range(1, rows + 1):
    #     knr = KNeighborsRegressor(i)
    #
    #     x_train = X_train[[feature_names[0], feature_names[1], feature_names[2]]]
    #
    #     knr.fit(x_train, y_train["Life Ladder"])
    #
    #     x_test = X_test[[feature_names[0], feature_names[1], feature_names[2]]]
    #     y_predict = knr.predict(x_test)
    #
    #     print(pd.DataFrame(list(zip(y_test["Life Ladder"], y_predict)), columns=['target', 'predicted']))
    #     print(f'Accuracy: {knr.score(x_test, y_test["Life Ladder"]):.4f}')

    # knr = KNeighborsRegressor(15)
    #
    # x_train = X_train[[feature_names[0], feature_names[1], feature_names[2]]]
    # # print(x_train)
    # # y_train = y_train
    # knr.fit(x_train, y_train["Life Ladder"])
    #
    # x_test = X_test[[feature_names[0], feature_names[1], feature_names[2]]]
    # y_predict = knr.predict(x_test)

    # print("MEASUREMENT ---------")
    # print(y_predict)

    # print(pd.DataFrame(list(zip(y_test["Life Ladder"], y_predict)), columns=['target', 'predicted']))
    # print(f'Accuracy: {knr.score(x_test,y_test["Life Ladder"]):.4f}')
    #
    # dia_cm = cm.get_cmap('Reds')
    #
    # x_min = X_train[feature_names[0]].min()
    # x_max = X_train[feature_names[0]].max()
    # x_range = x_max - x_min
    # x_min = x_min - 0.1 * x_range
    # x_max = x_max + 0.1 * x_range
    # y_min = X_train[feature_names[-1]].min()
    # y_max = X_train[feature_names[-1]].max()
    # y_range = y_max - y_min
    # y_min = y_min - 0.1 * y_range
    # y_max = y_max + 0.1 * y_range
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, .01 * x_range),
    #                      np.arange(y_min, y_max, .01 * y_range))
    # z = knr.predict(list(zip(xx.ravel(), yy.ravel())))
    # z = z.reshape(xx.shape)
    #
    # plt.figure()
    # plt.pcolormesh(xx, yy, z, cmap=dia_cm, shading='auto')
    #
    # plt.scatter(x_train[feature_names[0]], x_train[feature_names[-1]],
    #             c=y_train["Life Ladder"], label='Training data', cmap=dia_cm,
    #             edgecolor='black', linewidth=1, s=150)
    # plt.scatter(x_test[feature_names[0]], x_test[feature_names[-1]],
    #             c=y_test["Life Ladder"], marker='*', label='Testing data', cmap=dia_cm,
    #             edgecolor='black', linewidth=1, s=150)
    # plt.xlabel(feature_names[0])
    # plt.ylabel(feature_names[-1])
    # plt.legend()
    # plt.colorbar()
    #
    # plt.show()
