import os
import sys

import csv
from collections import namedtuple

import numpy as np

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib.axes import Axes


class Lab1:
    def __init__(self, path=None):
        self.path = None

        self.x = None
        self.y = None

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.pred_y_train = None
        self.pred_y_test = None
        self.pred_y = None

        if path is not None:
            if not os.path.isabs(path):
                path = os.path.normpath(os.path.join(sys.path[0], path))

            self.read(path)

        self.path = path
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3)

    def read(self, path: str):
        if not os.path.isabs(path):
            path = os.path.normpath(os.path.join(sys.path[0], path))

        with open(path, 'r') as data_file:
            reader = csv.reader(data_file, delimiter=',')
            next(reader)
            data = np.array([list(map(float, row)) for row in reader]).reshape(-1, 2)
            # data = np.array(data).reshape(-1, 2)[data[:, 0].argsort()]
            x, y = map(lambda k:  data[:, k].reshape(-1, 1), [0, 1])
        self.x, self.y = map(lambda k: (k - np.mean(k)) / np.std(k), [x, y])

    def clean(self):
        self.pred_y_train = None
        self.pred_y_test = None
        self.pred_y = None

    def show_model(self, x: np.ndarray, y: np.ndarray, title: str, on_scatter=True):
        data = np.column_stack((x, y))
        data = data[data[:, 0].argsort()]
        x_, y_ = data[:, 0], data[:, 1]

        _, ax = plt.subplots()
        ax.grid()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        if on_scatter:
            ax.scatter(self.x, self.y)
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.2))
        ax.plot(x_, y_, 'r')

        plt.show()

    def _model_regression(self, model: LinearRegression or Ridge, regressors_count: int):
        regressors = PolynomialFeatures(degree=regressors_count, include_bias=False)

        new_x_train = regressors.fit_transform(self.x_train)
        new_x_test = regressors.fit_transform(self.x_test)
        model.fit(new_x_train, self.y_train)

        self.pred_y_train = model.predict(new_x_train)
        self.pred_y_test = model.predict(new_x_test)
        self.pred_y = model.predict(regressors.fit_transform(self.x))

    def linear_regression(self, regressors_count: int):
        model = LinearRegression()
        self._model_regression(model, regressors_count)

    def ridge_regression(self, regressors_count: int, lmbd: float):
        model = Ridge(lmbd)
        self._model_regression(model, regressors_count)


def data_representation(model: Lab1):
    _, ax = plt.subplots()
    ax.grid()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.scatter(model.x, model.y)
    ax.set_title('Диаграмма рассеяния')

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))

    plt.show()


def task_1(model: Lab1, max_regressors: int, lmbd_b: float, lmbd_c: float):
    def draw_plots(title: str, data: tuple):
        _, ax = plt.subplots()
        ax.grid()
        ax.set_title(title)
        ax.set_xlabel('Число регрессоров модели')
        ax.set_ylabel('MSE')

        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.set_xlim(0, len(data[0].x) + 1)

        lines = []
        for k in data:
            lines.append(ax.plot(k.x, k.y, color=k.color, marker=k.marker, label=k.label)[0])
        plt.legend(handles=lines)

        plt.show()

    def task_1a():
        mse_train = []
        mse_test = []
        for m in range(1, max_regressors):
            model.linear_regression(m)
            mse_train.append(mse(model.y_train, model.pred_y_train))
            mse_test.append(mse(model.y_test, model.pred_y_test))

        mse_tpl = namedtuple('mse_data', ('x', 'y', 'color', 'marker', 'label'))

        draw_plots(title='Линейная регрессия', data=(
            mse_tpl(np.array(range(1, max_regressors)), mse_train, 'b', '.', 'Обучающая выборка'),
            mse_tpl(np.array(range(1, max_regressors)), mse_test, 'r', 'o', 'Тестовая выборка')))
        model.clean()

    def task_1b():
        mse_train = []
        mse_test = []
        for m in range(1, max_regressors):
            model.ridge_regression(m, lmbd_b)
            mse_train.append(mse(model.y_train, model.pred_y_train))
            mse_test.append(mse(model.y_test, model.pred_y_test))

        mse_tpl = namedtuple('mse_data', ('x', 'y', 'color', 'marker', 'label'))

        draw_plots(title='Гребневая регрессия при ' + r'$\lambda \approx 0$', data=(
            mse_tpl(np.array(range(1, max_regressors)), mse_train, 'b', '.', 'Обучающая выборка'),
            mse_tpl(np.array(range(1, max_regressors)), mse_test, 'r', 'o', 'Тестовая выборка')))
        model.clean()

    def task_1c():
        mse_train = []
        mse_test = []
        for m in range(1, max_regressors):
            model.ridge_regression(m, lmbd_c)
            mse_train.append(mse(model.y_train, model.pred_y_train))
            mse_test.append(mse(model.y_test, model.pred_y_test))

        mse_tpl = namedtuple('mse_data', ('x', 'y', 'color', 'marker', 'label'))

        draw_plots(title='Гребневая регрессия при ' + r'$\lambda \gg 0$', data=(
            mse_tpl(np.array(range(1, max_regressors)), mse_train, 'b', '.', 'Обучающая выборка'),
            mse_tpl(np.array(range(1, max_regressors)), mse_test, 'r', 'o', 'Тестовая выборка')))
        model.clean()

    task_1a()
    task_1b()
    task_1c()


def task_2(model: Lab1, m: int):
    model.linear_regression(m)
    model.show_model(model.x_test, model.pred_y_test, title='Линейная регрессия')

    model.ridge_regression(m, 0.01)
    model.show_model(model.x_test, model.pred_y_test, title='Гребневая регрессия при ' + r'$\lambda \approx 0$')

    model.ridge_regression(m, 1)
    model.show_model(model.x_test, model.pred_y_test, title='Гребневая регрессия при ' + r'$\lambda \gg 0$')


test = Lab1('data_v1-04.csv')
test.linear_regression(1)
data_representation(test)
task_1(test, 46, 0.01, 5)
task_2(test, 5)
