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
from matplotlib.ticker import MultipleLocator

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
            x, y = map(lambda k:  data[:, k].reshape(-1, 1), [0, 1])
        self.x, self.y = map(lambda k: (k - np.mean(k)) / np.std(k), [x, y])

    def new_split(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3)

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

        return model

    def ridge_regression(self, regressors_count: int, lmbd: float):
        model = Ridge(lmbd)
        self._model_regression(model, regressors_count)

        return model


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

        # ax.set_yscale('log')

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

        mse_tpl = namedtuple('data', ('x', 'y', 'color', 'marker', 'label'))

        draw_plots(title='Линейная регрессия', data=(
            mse_tpl(np.array(range(1, max_regressors)), mse_train, 'b', '.', 'Обучающая выборка'),
            mse_tpl(np.array(range(1, max_regressors)), mse_test, 'r', '.', 'Тестовая выборка')))
        draw_plots(title='Линейная регрессия', data=(
            mse_tpl(np.array(range(1, max_regressors // 2)), mse_train[:len(mse_train) // 2], 'b', '.',
                    'Обучающая выборка'),
            mse_tpl(np.array(range(1, max_regressors // 2)), mse_test[:len(mse_train) // 2], 'r', '.',
                    'Тестовая выборка')))
        model.clean()

    def task_1b():
        mse_train = []
        mse_test = []
        for m in range(1, max_regressors):
            model.ridge_regression(m, lmbd_b)
            mse_train.append(mse(model.y_train, model.pred_y_train))
            mse_test.append(mse(model.y_test, model.pred_y_test))

        mse_tpl = namedtuple('data', ('x', 'y', 'color', 'marker', 'label'))

        draw_plots(title='Гребневая регрессия при ' + r'$\lambda \approx 0$', data=(
            mse_tpl(np.array(range(1, max_regressors)), mse_train, 'b', '.', 'Обучающая выборка'),
            mse_tpl(np.array(range(1, max_regressors)), mse_test, 'r', '.', 'Тестовая выборка')))
        draw_plots(title='Гребневая регрессия при ' + r'$\lambda \approx 0$', data=(
            mse_tpl(np.array(range(1, max_regressors // 2)), mse_train[:len(mse_train) // 2], 'b', '.',
                    'Обучающая выборка'),
            mse_tpl(np.array(range(1, max_regressors // 2)), mse_test[:len(mse_train) // 2], 'r', '.',
                    'Тестовая выборка')))
        model.clean()

    def task_1c():
        mse_train = []
        mse_test = []
        for m in range(1, max_regressors):
            model.ridge_regression(m, lmbd_c)
            mse_train.append(mse(model.y_train, model.pred_y_train))
            mse_test.append(mse(model.y_test, model.pred_y_test))

        mse_tpl = namedtuple('data', ('x', 'y', 'color', 'marker', 'label'))

        draw_plots(title='Гребневая регрессия при ' + r'$\lambda \gg 0$', data=(
            mse_tpl(np.array(range(1, max_regressors)), mse_train, 'b', '.', 'Обучающая выборка'),
            mse_tpl(np.array(range(1, max_regressors)), mse_test, 'r', '.', 'Тестовая выборка')))
        draw_plots(title='Гребневая регрессия при ' + r'$\lambda \gg 0$', data=(
            mse_tpl(np.array(range(1, max_regressors // 2)), mse_train[:len(mse_train) // 2], 'b', '.',
                    'Обучающая выборка'),
            mse_tpl(np.array(range(1, max_regressors // 2)), mse_test[:len(mse_train) // 2], 'r', '.',
                    'Тестовая выборка')))
        model.clean()

    task_1a()
    task_1b()
    task_1c()


def task_2(model: Lab1, m: int, lmbd_b: float, lmbd_c: float):
    model.linear_regression(m)
    model.show_model(model.x_test, model.pred_y_test, title='Линейная регрессия')
    model.clean()

    model.ridge_regression(m, lmbd_b)
    model.show_model(model.x_test, model.pred_y_test, title='Гребневая регрессия при ' + r'$\lambda \approx 0$')
    model.clean()

    model.ridge_regression(m, lmbd_c)
    model.show_model(model.x_test, model.pred_y_test, title='Гребневая регрессия при ' + r'$\lambda \gg 0$')
    model.clean()


def task_3(model: Lab1, m: int, lmbd_min: float, lmbd_max: float, lmbd_step: float):
    ls = np.arange(lmbd_min, lmbd_max, lmbd_step)
    coef_matrix = np.empty((len(ls), m + 1))
    curr_row = 0

    for l in ls:
        rm = model.ridge_regression(m, l)
        coef_matrix[curr_row, 0] = rm.intercept_
        coef_matrix[curr_row, 1:] = rm.coef_
        curr_row += 1

    _, ax = plt.subplots()
    ax.grid()
    ax.set_title('Зависимость коэффициентов гребневой регрессии от ' + r'$\lambda$ ' + r'$(m = {})$'.format(m))
    ax.set_xlabel(r'$\lambda$')
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_ylabel('Значения коэффициентов')

    lines = []
    for k in range(m + 1):
        lines.append(ax.plot(ls, coef_matrix[:, k], label=r'$\beta_{}$'.format(k))[0])
    plt.legend(handles=lines)
    plt.show()

    model.clean()


def task_4(model: Lab1, m: int, lmbd_b: float, lmbd_c: float):
    def draw_plots(title: str, data: tuple):
        _, ax = plt.subplots()
        ax.grid()
        ax.set_title(title)
        ax.set_ylabel('Частота значения')
        ax.set_xlabel('Значение дисперсии')

        ax.xaxis.set_major_locator(MultipleLocator(0.01))
        ax.xaxis.set_minor_locator(MultipleLocator(0.001))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(MultipleLocator(2))

        ax.hist(data.x, color=data.color, label=data.label, bins=data.bins, ec='k')
        plt.legend()

        plt.show()

    def task_4a():
        unique_x = np.unique(model.x)
        train = {k: [] for k in unique_x}
        test = {k: [] for k in unique_x}
        for i in range(100):
            model.new_split()
            model.linear_regression(m)
            for j in range(len(model.x_train)):
                train[model.x_train[j][0]].append(model.pred_y_train[j][0])
            for j in range(len(model.x_test)):
                test[model.x_test[j][0]].append(model.pred_y_test[j][0])
        train_var = []
        test_var = []
        for i in unique_x:
            train_var.append(np.var(train[i]))
            test_var.append(np.var(test[i]))

        tpl = namedtuple('data', ('x', 'color', 'label', 'bins'))

        draw_plots(title='Распределения дисперсий линейной регрессии', data=(
            tpl(train_var, 'b', 'Обучающая выборка', 50)))
        draw_plots(title='Распределения дисперсий линейной регрессии', data=(
            tpl(test_var, 'r', 'Тестовая выборка', 50)))
        model.clean()

    def task_4b():
        unique_x = np.unique(model.x)
        train = {k: [] for k in unique_x}
        test = {k: [] for k in unique_x}
        for i in range(100):
            model.new_split()
            model.ridge_regression(m, lmbd_b)
            for j in range(len(model.x_train)):
                train[model.x_train[j][0]].append(model.pred_y_train[j][0])
            for j in range(len(model.x_test)):
                test[model.x_test[j][0]].append(model.pred_y_test[j][0])
        train_var = []
        test_var = []
        for i in unique_x:
            train_var.append(np.var(train[i]))
            test_var.append(np.var(test[i]))

        tpl = namedtuple('data', ('x', 'color', 'label', 'bins'))

        draw_plots(title='Распределения дисперсий гребневой регрессии при ' + r'$\lambda \approx 0$', data=(
            tpl(train_var, 'b', 'Обучающая выборка', 50)))
        draw_plots(title='Распределения дисперсий гребневой регрессии при ' + r'$\lambda \approx 0$', data=(
            tpl(test_var, 'r', 'Тестовая выборка', 50)))
        model.clean()

    def task_4c():
        unique_x = np.unique(model.x)
        train = {k: [] for k in unique_x}
        test = {k: [] for k in unique_x}
        for i in range(100):
            model.new_split()
            model.ridge_regression(m, lmbd_c)
            for j in range(len(model.x_train)):
                train[model.x_train[j][0]].append(model.pred_y_train[j][0])
            for j in range(len(model.x_test)):
                test[model.x_test[j][0]].append(model.pred_y_test[j][0])
        train_var = []
        test_var = []
        for i in unique_x:
            train_var.append(np.var(train[i]))
            test_var.append(np.var(test[i]))

        tpl = namedtuple('data', ('x', 'color', 'label', 'bins'))

        draw_plots(title='Распределения дисперсий гребневой регрессии при ' + r'$\lambda \gg 0$', data=(
            tpl(train_var, 'b', 'Обучающая выборка', 50)))
        draw_plots(title='Распределения дисперсий гребневой регрессии при ' + r'$\lambda \gg 0$', data=(
            tpl(test_var, 'r', 'Тестовая выборка', 50)))
        model.clean()

    task_4a()
    task_4b()
    task_4c()


def task_5(model: Lab1, m: int, lmbd_b: float, lmbd_c: float):
    def draw_plots(title: str, data: tuple, xlabel: str):
        _, ax = plt.subplots()
        ax.grid()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Остатки')

        lines = []
        for k in data:
            lines.append(ax.scatter(k.x, k.y, color=k.color, marker=k.marker, label=k.label))
        plt.legend(handles=lines)
        plt.show()

    def task_5a():
        model.linear_regression(m)
        diff_test = model.y_test - model.pred_y_test
        diff_train = model.y_train - model.pred_y_train

        tpl = namedtuple('data', ('x', 'y', 'color', 'marker', 'label'))

        draw_plots(title='Остатки линейной регрессии', data=(
            tpl(model.x_train, diff_train, 'b', '.', 'Обучающая выборка'),
            tpl(model.x_test, diff_test, 'r', '.', 'Тестовая выборка')),
                   xlabel='x')
        draw_plots(title='Остатки линейной регрессии', data=(
            tpl(model.y_train, diff_train, 'b', '.', 'Обучающая выборка'),
            tpl(model.y_test, diff_test, 'r', '.', 'Тестовая выборка')),
                   xlabel='y')
        print(diff_test.mean())
        model.clean()

    def task_5b():
        model.ridge_regression(m, lmbd_b)
        diff_test = model.y_test - model.pred_y_test
        diff_train = model.y_train - model.pred_y_train

        tpl = namedtuple('data', ('x', 'y', 'color', 'marker', 'label'))

        draw_plots(title='Остатки гребневой регрессии при ' + r'$\lambda \approx 0$', data=(
            tpl(model.x_train, diff_train, 'b', '.', 'Обучающая выборка'),
            tpl(model.x_test, diff_test, 'r', '.', 'Тестовая выборка')),
                   xlabel='x')
        draw_plots(title='Остатки гребневой регрессии при ' + r'$\lambda \approx 0$', data=(
            tpl(model.y_train, diff_train, 'b', '.', 'Обучающая выборка'),
            tpl(model.y_test, diff_test, 'r', '.', 'Тестовая выборка')),
                   xlabel='y')
        print(diff_test.mean())
        model.clean()

    def task_5c():
        model.ridge_regression(m, lmbd_c)
        diff_test = model.y_test - model.pred_y_test
        diff_train = model.y_train - model.pred_y_train

        tpl = namedtuple('data', ('x', 'y', 'color', 'marker', 'label'))

        draw_plots(title='Остатки гребневой регрессии  при ' + r'$\lambda \gg 0$', data=(
            tpl(model.x_train, diff_train, 'b', '.', 'Обучающая выборка'),
            tpl(model.x_test, diff_test, 'r', '.', 'Тестовая выборка')),
                   xlabel='x')
        draw_plots(title='Остатки гребневой регрессии  при ' + r'$\lambda \gg 0$', data=(
            tpl(model.y_train, diff_train, 'b', '.', 'Обучающая выборка'),
            tpl(model.y_test, diff_test, 'r', '.', 'Тестовая выборка')),
                   xlabel='y')
        print(diff_test.mean())
        model.clean()

    task_5a()
    task_5b()
    task_5c()


test = Lab1('data_v1-04.csv')
data_representation(test)
task_1(test, 50, 0.001, 100)
task_2(test, 5, 0.001, 100)
task_3(test, 5, 0.001, 100, 0.5)
task_5(test, 5, 0.001, 100)
task_4(test, 5, 0.001, 100)
