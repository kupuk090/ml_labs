import os
import sys

import csv
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import MultipleLocator


class Lab2:
    def __init__(self, path=None):
        self.path = None
        self.orig_data = None

        self.data = {}
        self.train = {}
        self.test = {}

        if path is not None:
            if not os.path.isabs(path):
                path = os.path.normpath(os.path.join(sys.path[0], path))

            self.read(path)

        self.path = path
        for k, v in self.data.items():
            self.train[k], self.test[k] = train_test_split(v, test_size=0.3)

    def read(self, path: str):
        if not os.path.isabs(path):
            path = os.path.normpath(os.path.join(sys.path[0], path))

        with open(path, 'r') as data_file:
            reader = csv.reader(data_file, delimiter=',')
            next(reader)
            data = np.array([list(map(float, row)) for row in reader])
            self.orig_data = data

        for l in map(int, np.unique(data[:, -1])):
            self.data[l] = data[data[:, -1] == l][:, :-1]
            self.train[l], self.test[l] = train_test_split(self.data[l], test_size=0.3)

    def calc_shared_scalar_cov(self):
        s = []
        for v in self.train.values():
            s.append(np.mean(np.var(v, axis=0)))
        return np.diag((1, 1)) * np.mean(s), 0

    def calc_shared_diagonal_cov(self):
        s = []
        for v in self.train.values():
            s.append(np.var(v, axis=0))
        matr = np.zeros((2, 2))
        matr[0, 0], matr[1, 1] = np.mean(s, axis=0)
        return matr, 90 if matr[0, 0] < matr[1, 1] else 0

    def calc_scalar_cov(self, key):
        matr = np.diag((1, 1)) * np.mean(np.var(self.train[key], axis=0))
        return matr, 90 if matr[0, 0] < matr[1, 1] else 0

    def calc_diagonal_cov(self, key):
        matr = np.cov(self.train[key][:, 0], self.train[key][:, 1])
        matr[0, 1] = matr[1, 0] = 0
        return matr, 90 if matr[0, 0] < matr[1, 1] else 0

    def calc_shared_cov(self):
        s = []
        for v in self.train.values():
            s.append(np.cov(v[:, 0], v[:, 1]))
        m = np.mean(s, axis=0)

        # Find and sort eigenvalues and eigenvectors into descending order
        eigvals, eigvecs = np.linalg.eigh(m)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        # The anti-clockwise angle to rotate our ellipse by
        vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
        angle = np.arctan2(vy, vx) * 180 / np.pi

        return m, angle

    def calc_cov(self, key):
        m = np.cov(self.train[key][:, 0], self.train[key][:, 1])

        # Find and sort eigenvalues and eigenvectors into descending order
        eigvals, eigvecs = np.linalg.eigh(m)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        # The anti-clockwise angle to rotate our ellipse by
        vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
        angle = np.arctan2(vy, vx) * 180 / np.pi

        return m, angle


def data_representation(model: Lab2):
    _, ax = plt.subplots()
    ax.grid()
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Диаграмма рассеяния')

    g = []
    for k, v in model.data.items():
        g.append(ax.scatter(v[:, 0], v[:, 1], marker='.', label=f'Class {k}'))
    plt.legend(handles=g)

    # ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))

    plt.show()


def task_1(model: Lab2):
    def draw_plots(title: str, data: list):
        _, ax = plt.subplots()
        ax.grid()
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(title)

        mini = np.min(model.orig_data[:, :-1])
        maxi = np.max(model.orig_data[:, :-1])
        ax.set_xlim(mini,  maxi)
        ax.set_ylim(mini,  maxi)

        g = []
        colors = []
        for k, v in model.train.items():
            a = ax.scatter(v[:, 0], v[:, 1], marker='.', label=f'Class {k}')
            c = a.get_facecolor()[0]
            g.append(a)
            colors.append(c)

            g.append(ax.scatter(model.test[k][:, 0], model.test[k][:, 1], marker='x', s=20, color=c))

        k = 0
        for el in data:
            w = 2 * el.m[0, 0]
            h = 2 * el.m[1, 1]
            for j in range(1, 5):
                ell = Ellipse(xy=(el.x, el.y),
                              width=w, height=h,
                              angle=el.a,
                              fill=False,
                              color=colors[k])
                ax.add_patch(ell)

                w *= 0.8
                h *= 0.8
            k += 1
        plt.legend(handles=g)

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))

        plt.show()

    def a():
        matr, angle = model.calc_shared_scalar_cov()
        data = []
        for k, v in model.train.items():
            x, y = np.mean(v, axis=0)
            data.append(matr_tpl(x, y, matr, angle, f'Class {k}'))
        draw_plots(title='Shared & Scalar', data=data)

    def b():
        matr, angle = model.calc_shared_diagonal_cov()
        data = []
        for k, v in model.train.items():
            x, y = np.mean(v, axis=0)
            data.append(matr_tpl(x, y, matr, angle, f'Class {k}'))
        draw_plots(title='Shared & Diagonal', data=data)

    def c():
        data = []
        for k, v in model.train.items():
            x, y = np.mean(v, axis=0)
            matr, angle = model.calc_scalar_cov(k)
            data.append(matr_tpl(x, y, matr, angle, f'Class {k}'))
        draw_plots(title='Not shared & Scalar', data=data)

    def d():
        data = []
        for k, v in model.train.items():
            x, y = np.mean(v, axis=0)
            matr, angle = model.calc_diagonal_cov(k)
            data.append(matr_tpl(x, y, matr, angle, f'Class {k}'))
        draw_plots(title='Not shared & Diagonal', data=data)

    def e():
        matr, angle = model.calc_shared_cov()
        data = []
        for k, v in model.train.items():
            x, y = np.mean(v, axis=0)
            data.append(matr_tpl(x, y, matr, angle, f'Class {k}'))
        draw_plots(title='Shared', data=data)

    def f():
        data = []
        for k, v in model.train.items():
            x, y = np.mean(v, axis=0)
            matr, angle = model.calc_cov(k)
            data.append(matr_tpl(x, y, matr, angle, f'Class {k}'))
        draw_plots(title='No assumptions', data=data)

    matr_tpl = namedtuple('data', ('x', 'y', 'm', 'a', 'l'))

    a()
    b()
    c()
    d()
    e()
    f()


def task_2(model: Lab2):
    def abf_discr(kind: str, lst: list):

        Ak, bk, ck = {}, {}, {}
        for x in lst:
            key, cov_matr = x
            inv_cov_matr = np.linalg.inv(cov_matr)
            m = np.mean(model.train[key], axis=0).T

            Ak[key] = -0.5 * inv_cov_matr
            bk[key] = inv_cov_matr.dot(m)
            ck[key] = -0.5 * m.T.dot(inv_cov_matr).dot(m) - 0.5 * np.log(np.linalg.det(cov_matr)) + np.log(1 / len(model.train.keys()))

        if kind == 'train':
            dct = model.train
        else:
            dct = model.test

        dk = {k: [] for k in dct.keys()}
        for k, v in dct.items():
            for item in v:
                v_ = []
                for k_ in dct.keys():
                    v_.append(np.exp(item.dot(Ak[k_]).dot(item.T) + bk[k_].T.dot(item.T) + ck[k_]))
                dk[k].append(v_)

        return {k: np.array(v) for k, v in dk.items()}

    def cde_discr(kind: str, lst: list):

        bk, ck = {}, {}
        for x in lst:
            key, cov_matr = x
            inv_cov_matr = np.linalg.inv(cov_matr)
            m = np.mean(model.train[key], axis=0).T

            bk[key] = inv_cov_matr.dot(m)
            ck[key] = -0.5 * m.T.dot(inv_cov_matr).dot(m) + np.log(1 / len(model.train.keys()))

        if kind == 'train':
            dct = model.train
        else:
            dct = model.test

        dk = {k: [] for k in dct.keys()}
        for k, v in dct.items():
            for item in v:
                v_ = []
                for k_ in dct.keys():
                    v_.append(np.exp(bk[k_].T.dot(item.T) + ck[k_]))
                dk[k].append(v_)

        return {k: np.array(v) for k, v in dk.items()}

    def calc_accuracy(dct: dict):
        acc = 0
        count = 0
        for k, v in dct.items():
            acc += np.sum((np.argmax(v, axis=1) + 1) == k)
            count += len(v)
        return acc / count

    def a():
        matr, _ = model.calc_shared_scalar_cov()
        data = []
        for k in model.train.keys():
            data.append((k, matr))
        score_train = abf_discr('train', data)
        for k in model.test.keys():
            data.append((k, matr))
        score_test = abf_discr('test', data)

        return dict(zip(acc_df.columns, ['Shared & Scalar', calc_accuracy(score_train), calc_accuracy(score_test)]))

    def b():
        matr, _ = model.calc_shared_diagonal_cov()
        data = []
        for k in model.train.keys():
            data.append((k, matr))
        score_train = abf_discr('train', data)
        for k in model.test.keys():
            data.append((k, matr))
        score_test = abf_discr('test', data)

        return dict(zip(acc_df.columns, ['Shared & Diagonal', calc_accuracy(score_train), calc_accuracy(score_test)]))

    def c():
        data = []
        for k, v in model.train.items():
            matr, _ = model.calc_scalar_cov(k)
            data.append((k, matr))
        score_train = cde_discr('train', data)
        for k in model.test.keys():
            matr, _ = model.calc_scalar_cov(k)
            data.append((k, matr))
        score_test = cde_discr('test', data)

        return dict(zip(acc_df.columns, ['Not shared & Scalar', calc_accuracy(score_train), calc_accuracy(score_test)]))

    def d():
        data = []
        for k, v in model.train.items():
            matr, _ = model.calc_diagonal_cov(k)
            data.append((k, matr))
        score_train = cde_discr('train', data)
        for k in model.test.keys():
            matr, _ = model.calc_diagonal_cov(k)
            data.append((k, matr))
        score_test = cde_discr('test', data)

        return dict(zip(acc_df.columns, ['Not shared & Diagonal', calc_accuracy(score_train), calc_accuracy(score_test)]))

    def e():
        matr, _ = model.calc_shared_cov()
        data = []
        for k in model.train.keys():
            data.append((k, matr))
        score_train = abf_discr('train', data)
        for k in model.test.keys():
            data.append((k, matr))
        score_test = abf_discr('test', data)

        return dict(zip(acc_df.columns, ['Shared', calc_accuracy(score_train), calc_accuracy(score_test)]))

    def f():
        data = []
        for k in model.train.keys():
            matr, _ = model.calc_cov(k)
            data.append((k, matr))
        score_train = abf_discr('train', data)
        for k in model.test.keys():
            matr, _ = model.calc_cov(k)
            data.append((k, matr))
        score_test = abf_discr('test', data)

        return dict(zip(acc_df.columns, ['No assumptions', calc_accuracy(score_train), calc_accuracy(score_test)]))

    acc_df = pd.DataFrame(columns=['Cov matrix', 'Train', 'Test'])
    acc_df = acc_df.append(a(), ignore_index=True)
    acc_df = acc_df.append(b(), ignore_index=True)
    acc_df = acc_df.append(c(), ignore_index=True)
    acc_df = acc_df.append(d(), ignore_index=True)
    acc_df = acc_df.append(e(), ignore_index=True)
    acc_df = acc_df.append(f(), ignore_index=True)
    print(acc_df.to_string(index=False))


test = Lab2("data_v2-04.csv")
# data_representation(test)
# task_1(test)
task_2(test)
