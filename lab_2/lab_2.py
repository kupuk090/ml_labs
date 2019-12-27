import os
import sys

import csv
from collections import namedtuple
from itertools import product

import numpy as np
import pandas as pd

from copy import copy

from scipy import interp

from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import MultipleLocator

import matplotlib
matplotlib.use('macosx')


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
        c = 0
        for v in self.train.values():
            s.append(np.mean(np.var(v, axis=0) * (len(v) - 1)))
            c += len(v) - 1
        return np.eye(2) * np.sum(s) / c, 0

    def calc_shared_diagonal_cov(self):
        s = []
        c = 0
        for v in self.train.values():
            s.append(np.var(v, axis=0) * (len(v) - 1))
            c += len(v) - 1
        matr = np.zeros((2, 2))
        matr[0, 0], matr[1, 1] = np.sum(s, axis=0) / c
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
        c = 0
        for v in self.train.values():
            s.append(np.cov(v[:, 0], v[:, 1]) * (len(v) - 1))
            c += len(v) - 1
        m = np.sum(s, axis=0) / c

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
        koef = [1, 0.95, 0.9, 0.75, 0.5]
        for el in data:
            w = 2 * el.m[0, 0]
            h = 2 * el.m[1, 1]
            for j in range(5):
                ell = Ellipse(xy=(el.x, el.y),
                              width=w * koef[j], height=h * koef[j],
                              angle=el.a,
                              fill=False,
                              color=colors[k])
                ax.add_patch(ell)
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


def task_2_4_5(model: Lab2):
    def cdf_discr(kind: str, lst: list):
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

    def abe_discr(kind, lst: list):
        bk, ck = {}, {}
        for x in lst:
            key, cov_matr = x
            inv_cov_matr = np.linalg.inv(cov_matr)
            m = np.mean(model.train[key], axis=0).T

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

    def calc_pr(score_data: dict, title: str):
        pr_y_true = np.concatenate([np.repeat(k, len(v), axis=0) for k, v in score_data.items()], axis=0).reshape((-1, 1))
        pr_y_test = np.concatenate([v / np.sum(v, axis=1).reshape((-1, 1)) for k, v in score_data.items()], axis=0).reshape((-1, 4))

        y_true = label_binarize(pr_y_true, classes=list(score_data.keys()))
        n_classes = y_true.shape[1]

        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = metrics.precision_recall_curve(y_true[:, i],
                                                                        pr_y_test[:, i])
            average_precision[i] = metrics.average_precision_score(y_true[:, i],
                                                                   pr_y_test[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(y_true.ravel(),
                                                                                pr_y_test.ravel())
        average_precision["micro"] = metrics.average_precision_score(y_true, pr_y_test, average="micro")

        # A "macro-average": quantifying score on all classes jointly
        average_precision["macro"] = metrics.average_precision_score(y_true, pr_y_test, average="macro")

        # drawing
        _, ax = plt.subplots()
        lines = []
        labels = []

        l, = ax.plot(recall["micro"], precision["micro"], linestyle=':', lw=3)
        lines.append(l)
        labels.append('micro-average Precision-recall (area = {0:0.2f})'
                      ''.format(average_precision["micro"]))

        for i in range(n_classes):
            l, = ax.plot(recall[i], precision[i])
            lines.append(l)
            labels.append('Precision-recall for Class {0} (area = {1:0.2f})'
                          ''.format(i + 1, average_precision[i]))

        ax.grid()
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.02))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.02))

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall curves for {} cov matrix'.format(title))
        plt.legend(lines, labels)

        plt.show()

        return average_precision["micro"], average_precision["macro"]

    def calc_roc(score_data: dict, title: str):
        roc_y_true = np.concatenate([np.repeat(k, len(v), axis=0) for k, v in score_data.items()], axis=0).reshape((-1, 1))
        roc_y_test = np.concatenate([v / np.sum(v, axis=1).reshape((-1, 1)) for k, v in score_data.items()], axis=0).reshape((-1, 4))

        y_true = label_binarize(roc_y_true, classes=list(score_data.keys()))
        n_classes = y_true.shape[1]

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], roc_y_test[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true.ravel(), roc_y_test.ravel())
        roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

        # drawing
        _, ax = plt.subplots()
        lines = []
        labels = []

        l, = ax.plot(fpr["micro"], tpr["micro"], linestyle=':', lw=3)
        lines.append(l)
        labels.append('micro-average ROC (area = {0:0.2f})'
                      ''.format(roc_auc["micro"]))
        l, = ax.plot(fpr["macro"], tpr["macro"], linestyle=':', lw=3)
        lines.append(l)
        labels.append('macro-average ROC (area = {0:0.2f})'
                      ''.format(roc_auc["macro"]))

        for i in range(n_classes):
            l, = ax.plot(fpr[i], tpr[i])
            lines.append(l)
            labels.append('ROC for Class {0} (area = {1:0.2f})'
                          ''.format(i + 1, roc_auc[i]))

        plt.plot([0, 1], [0, 1], color='black', linestyle='--')

        ax.grid()
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.02))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.02))

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver operating characteristic curves for {} cov matrix'.format(title))
        plt.legend(lines, labels)

        plt.show()

        return roc_auc["micro"], roc_auc["macro"]

    def add_to_axis_pr(score_data: dict, title: str, ax):
        pr_y_true = np.concatenate([np.repeat(k, len(v), axis=0) for k, v in score_data.items()], axis=0).reshape(
            (-1, 1))
        pr_y_test = np.concatenate([v / np.sum(v, axis=1).reshape((-1, 1)) for k, v in score_data.items()],
                                   axis=0).reshape((-1, 4))

        y_true = label_binarize(pr_y_true, classes=list(score_data.keys()))

        precision = dict()
        recall = dict()
        average_precision = dict()

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(y_true.ravel(),
                                                                                pr_y_test.ravel())
        average_precision["micro"] = metrics.average_precision_score(y_true, pr_y_test, average="micro")

        # drawing
        return ax.plot(recall["micro"], precision["micro"], label='{0} (area = {1:0.2f})'
                       .format(title, average_precision["micro"]))[0]

    def add_to_axis_roc(score_data: dict, title: str, ax_micro, ax_macro):
        roc_y_true = np.concatenate([np.repeat(k, len(v), axis=0) for k, v in score_data.items()], axis=0).reshape(
            (-1, 1))
        roc_y_test = np.concatenate([v / np.sum(v, axis=1).reshape((-1, 1)) for k, v in score_data.items()],
                                    axis=0).reshape((-1, 4))

        y_true = label_binarize(roc_y_true, classes=list(score_data.keys()))
        n_classes = y_true.shape[1]

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], roc_y_test[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true.ravel(), roc_y_test.ravel())
        roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

        return ax_micro.plot(fpr["micro"], tpr["micro"], label='{0} (area = {1:0.2f})'
                             .format(title, roc_auc["micro"]))[0],\
               ax_macro.plot(fpr["macro"], tpr["macro"], label='{0} (area = {1:0.2f})'
                             .format(title, roc_auc["macro"]))[0]

    def a():
        matr, _ = model.calc_shared_scalar_cov()
        data = []
        for k in model.train.keys():
            data.append((k, matr))
        score_train = abe_discr('train', data)

        for k in model.test.keys():
            data.append((k, matr))
        score_test = abe_discr('test', data)

        tmp = add_to_axis_roc(score_train, 'Shared & Scalar', ax_roc_train_micro, ax_roc_train_macro)
        ax_roc_train_micro_lst.append(tmp[0])
        ax_roc_train_macro_lst.append(tmp[1])
        tmp = add_to_axis_roc(score_test, 'Shared & Scalar', ax_roc_test_micro, ax_roc_test_macro)
        ax_roc_test_micro_lst.append(tmp[0])
        ax_roc_test_macro_lst.append(tmp[1])
        ax_pr_train_micro_lst.append(add_to_axis_pr(score_train, 'Shared & Scalar', ax_pr_train_micro))
        ax_pr_test_micro_lst.append(add_to_axis_pr(score_test, 'Shared & Scalar', ax_pr_test_micro))

        return dict(zip(acc_df.columns, ['Shared & Scalar', calc_accuracy(score_train), calc_accuracy(score_test)])), \
               dict(zip(pr_df.columns, ('Shared & Scalar',) + calc_pr(score_train, 'Shared & Scalar') + calc_pr(score_test, 'Shared & Scalar'))), \
               dict(zip(roc_df.columns, ('Shared & Scalar',) + calc_roc(score_train, 'Shared & Scalar') + calc_roc(score_test, 'Shared & Scalar')))

    def b():
        matr, _ = model.calc_shared_diagonal_cov()
        data = []
        for k in model.train.keys():
            data.append((k, matr))
        score_train = abe_discr('train', data)

        for k in model.test.keys():
            data.append((k, matr))
        score_test = abe_discr('test', data)

        tmp =  add_to_axis_roc(score_train, 'Shared & Diagonal', ax_roc_train_micro, ax_roc_train_macro)
        ax_roc_train_micro_lst.append(tmp[0])
        ax_roc_train_macro_lst.append(tmp[1])
        tmp =  add_to_axis_roc(score_test, 'Shared & Diagonal', ax_roc_test_micro, ax_roc_test_macro)
        ax_roc_test_micro_lst.append(tmp[0])
        ax_roc_test_macro_lst.append(tmp[1])
        ax_pr_train_micro_lst.append(add_to_axis_pr(score_train, 'Shared & Diagonal', ax_pr_train_micro))
        ax_pr_test_micro_lst.append(add_to_axis_pr(score_test, 'Shared & Diagonal', ax_pr_test_micro))

        return dict(zip(acc_df.columns, ['Shared & Diagonal', calc_accuracy(score_train), calc_accuracy(score_test)])),\
               dict(zip(pr_df.columns, ('Shared & Diagonal',) + calc_pr(score_train, 'Shared & Diagonal') + calc_pr(score_test, 'Shared & Diagonal'))), \
               dict(zip(roc_df.columns, ('Shared & Diagonal',) + calc_roc(score_train, 'Shared & Diagonal') + calc_roc(score_test, 'Shared & Diagonal')))

    def c():
        data = []
        for k, v in model.train.items():
            matr, _ = model.calc_scalar_cov(k)
            data.append((k, matr))
        score_train = cdf_discr('train', data)

        for k in model.test.keys():
            matr, _ = model.calc_scalar_cov(k)
            data.append((k, matr))
        score_test = cdf_discr('test', data)

        tmp =  add_to_axis_roc(score_train, 'Not shared & Scalar', ax_roc_train_micro, ax_roc_train_macro)
        ax_roc_train_micro_lst.append(tmp[0])
        ax_roc_train_macro_lst.append(tmp[1])
        tmp =  add_to_axis_roc(score_test, 'Not shared & Scalar', ax_roc_test_micro, ax_roc_test_macro)
        ax_roc_test_micro_lst.append(tmp[0])
        ax_roc_test_macro_lst.append(tmp[1])
        ax_pr_train_micro_lst.append(add_to_axis_pr(score_train, 'Not shared & Scalar', ax_pr_train_micro))
        ax_pr_test_micro_lst.append(add_to_axis_pr(score_test, 'Not shared & Scalar', ax_pr_test_micro))

        return dict(zip(acc_df.columns, ['Not shared & Scalar', calc_accuracy(score_train), calc_accuracy(score_test)])),\
               dict(zip(pr_df.columns, ('Not shared & Scalar',) + calc_pr(score_train, 'Not shared & Scalar') + calc_pr(score_test, 'Not shared & Scalar'))), \
               dict(zip(roc_df.columns, ('Not shared & Scalar',) + calc_roc(score_train, 'Not shared & Scalar') + calc_roc(score_test, 'Not shared & Scalar')))

    def d():
        data = []
        for k, v in model.train.items():
            matr, _ = model.calc_diagonal_cov(k)
            data.append((k, matr))
        score_train = cdf_discr('train', data)

        for k in model.test.keys():
            matr, _ = model.calc_diagonal_cov(k)
            data.append((k, matr))
        score_test = cdf_discr('test', data)

        tmp =  add_to_axis_roc(score_train, 'Not shared & Diagonal', ax_roc_train_micro, ax_roc_train_macro)
        ax_roc_train_micro_lst.append(tmp[0])
        ax_roc_train_macro_lst.append(tmp[1])
        tmp =  add_to_axis_roc(score_test, 'Not shared & Diagonal', ax_roc_test_micro, ax_roc_test_macro)
        ax_roc_test_micro_lst.append(tmp[0])
        ax_roc_test_macro_lst.append(tmp[1])
        ax_pr_train_micro_lst.append(add_to_axis_pr(score_train, 'Not shared & Diagonal', ax_pr_train_micro))
        ax_pr_test_micro_lst.append(add_to_axis_pr(score_test, 'Not shared & Diagonal', ax_pr_test_micro))

        return dict(zip(acc_df.columns, ['Not shared & Diagonal', calc_accuracy(score_train), calc_accuracy(score_test)])), \
               dict(zip(pr_df.columns, ('Not shared & Diagonal',) + calc_pr(score_train, 'Not shared & Diagonal') + calc_pr(score_test, 'Not shared & Diagonal'))), \
               dict(zip(roc_df.columns, ('Not shared & Diagonal',) + calc_roc(score_train, 'Not shared & Diagonal') + calc_roc(score_test, 'Not shared & Diagonal')))

    def e():
        matr, _ = model.calc_shared_cov()
        data = []
        for k in model.train.keys():
            data.append((k, matr))
        score_train = abe_discr('train', data)

        for k in model.test.keys():
            data.append((k, matr))
        score_test = abe_discr('test', data)

        tmp =  add_to_axis_roc(score_train, 'Shared', ax_roc_train_micro, ax_roc_train_macro)
        ax_roc_train_micro_lst.append(tmp[0])
        ax_roc_train_macro_lst.append(tmp[1])
        tmp =  add_to_axis_roc(score_test, 'Shared', ax_roc_test_micro, ax_roc_test_macro)
        ax_roc_test_micro_lst.append(tmp[0])
        ax_roc_test_macro_lst.append(tmp[1])
        ax_pr_train_micro_lst.append(add_to_axis_pr(score_train, 'Shared', ax_pr_train_micro))
        ax_pr_test_micro_lst.append(add_to_axis_pr(score_test, 'Shared', ax_pr_test_micro))

        return dict(zip(acc_df.columns, ['Shared', calc_accuracy(score_train), calc_accuracy(score_test)])), \
               dict(zip(pr_df.columns, ('Shared',) + calc_pr(score_train, 'Shared') + calc_pr(score_test, 'Shared'))), \
               dict(zip(roc_df.columns, ('Shared',) + calc_roc(score_train, 'Shared') + calc_roc(score_test, 'Shared')))

    def f():
        data = []
        for k in model.train.keys():
            matr, _ = model.calc_cov(k)
            data.append((k, matr))
        score_train = cdf_discr('train', data)

        for k in model.test.keys():
            matr, _ = model.calc_cov(k)
            data.append((k, matr))
        score_test = cdf_discr('test', data)

        tmp =  add_to_axis_roc(score_train, 'No assumptions', ax_roc_train_micro, ax_roc_train_macro)
        ax_roc_train_micro_lst.append(tmp[0])
        ax_roc_train_macro_lst.append(tmp[1])
        tmp =  add_to_axis_roc(score_test, 'No assumptions', ax_roc_test_micro, ax_roc_test_macro)
        ax_roc_test_micro_lst.append(tmp[0])
        ax_roc_test_macro_lst.append(tmp[1])
        ax_pr_train_micro_lst.append(add_to_axis_pr(score_train, 'No assumptions', ax_pr_train_micro))
        ax_pr_test_micro_lst.append(add_to_axis_pr(score_test, 'No assumptions', ax_pr_test_micro))

        return dict(zip(acc_df.columns, ['No assumptions', calc_accuracy(score_train), calc_accuracy(score_test)])), \
               dict(zip(pr_df.columns, ('No assumptions',) + calc_pr(score_train, 'No assumptions') + calc_pr(score_test, 'No assumptions'))), \
               dict(zip(roc_df.columns, ('No assumptions',) + calc_roc(score_train, 'No assumptions') + calc_roc(score_test, 'No assumptions')))

    # task 4 & 5
    roc_df = pd.DataFrame(columns=['Cov matrix', 'Train Micro', 'Train Macro', 'Test Micro', 'Test Macro'])
    pr_df = pd.DataFrame(columns=['Cov matrix', 'Train Micro', 'Train Macro', 'Test Micro', 'Test Macro'])
    acc_df = pd.DataFrame(columns=['Cov matrix', 'Train', 'Test'])

    _, ax_roc_train_micro = plt.subplots()
    _, ax_roc_test_micro = plt.subplots()
    _, ax_roc_train_macro = plt.subplots()
    _, ax_roc_test_macro = plt.subplots()
    _, ax_pr_train_micro = plt.subplots()
    _, ax_pr_test_micro = plt.subplots()

    ax_roc_train_micro_lst = []
    ax_roc_test_micro_lst = []
    ax_roc_train_macro_lst = []
    ax_roc_test_macro_lst = []
    ax_pr_train_micro_lst = []
    ax_pr_test_micro_lst = []

    t = a()
    acc_df = acc_df.append(t[0], ignore_index=True)
    pr_df = pr_df.append(t[1], ignore_index=True)
    roc_df = roc_df.append(t[2], ignore_index=True)

    t = b()
    acc_df = acc_df.append(t[0], ignore_index=True)
    pr_df = pr_df.append(t[1], ignore_index=True)
    roc_df = roc_df.append(t[2], ignore_index=True)

    t = c()
    acc_df = acc_df.append(t[0], ignore_index=True)
    pr_df = pr_df.append(t[1], ignore_index=True)
    roc_df = roc_df.append(t[2], ignore_index=True)

    t = d()
    acc_df = acc_df.append(t[0], ignore_index=True)
    pr_df = pr_df.append(t[1], ignore_index=True)
    roc_df = roc_df.append(t[2], ignore_index=True)

    t = e()
    acc_df = acc_df.append(t[0], ignore_index=True)
    pr_df = pr_df.append(t[1], ignore_index=True)
    roc_df = roc_df.append(t[2], ignore_index=True)

    t = f()
    acc_df = acc_df.append(t[0], ignore_index=True)
    pr_df = pr_df.append(t[1], ignore_index=True)
    roc_df = roc_df.append(t[2], ignore_index=True)

    print(acc_df.to_string(index=False))
    print()
    print(pr_df.to_string(index=False))
    print()
    print(roc_df.to_string(index=False))

    ax = acc_df.plot.bar(x='Cov matrix', rot=90)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))
    plt.show()

    for _, lst in [(ax_roc_test_micro, ax_roc_test_micro_lst), (ax_roc_train_micro, ax_roc_train_micro_lst)]:
        _, ax = plt.subplots()

        for line in lst:
            lcp = copy(line)
            lcp.axes = None
            lcp.figure = None
            lcp.set_transform(ax.transData)
            ax.add_line(lcp)
        ax.plot([0, 1], [0, 1], color='black', linestyle='--')

        ax.grid()
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.02))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.02))

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Micro-averaged ROC curves')

        plt.legend(lst)
        plt.show()

    for _, lst in [(ax_roc_test_macro, ax_roc_test_macro_lst), (ax_roc_train_macro, ax_roc_train_macro_lst)]:
        _, ax = plt.subplots()

        for line in lst:
            lcp = copy(line)
            lcp.axes = None
            lcp.figure = None
            lcp.set_transform(ax.transData)
            ax.add_line(lcp)
        ax.plot([0, 1], [0, 1], color='black', linestyle='--')

        ax.grid()
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.02))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.02))

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Macro-averaged ROC curves')

        plt.legend(lst)
        plt.show()

    for _, lst in [(ax_pr_test_micro, ax_pr_test_micro_lst), (ax_pr_train_micro, ax_pr_train_micro_lst)]:
        _, ax = plt.subplots()

        for line in lst:
            lcp = copy(line)
            lcp.axes = None
            lcp.figure = None
            lcp.set_transform(ax.transData)
            ax.add_line(lcp)

        ax.grid()
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.02))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.02))

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Micro-averaged PR curves')

        plt.legend(lst)
        plt.show()


def task_3(model: Lab2):
    def cdf_discr(_lst: list):
        Ak, bk, ck = {}, {}, {}
        for x in _lst:
            key, cov_matr = x
            inv_cov_matr = np.linalg.inv(cov_matr)
            m = np.mean(model.train[key], axis=0).T

            Ak[key] = -0.5 * inv_cov_matr
            bk[key] = inv_cov_matr.dot(m).reshape((-1, 1))
            ck[key] = -0.5 * m.T.dot(inv_cov_matr).dot(m) - 0.5 * np.log(np.linalg.det(cov_matr)) + np.log(1 / len(model.train.keys()))

        res = []
        classes = model.data.keys()
        for k in classes:
            res.append(np.exp(np.sum(lst.dot(Ak[k]) * lst, axis=1).reshape((-1, 1)) + lst.dot(bk[k]) + ck[k]))

        return np.concatenate(res, axis=1)

    def abe_discr(_lst: list):
        bk, ck = {}, {}
        for x in _lst:
            key, cov_matr = x
            inv_cov_matr = np.linalg.inv(cov_matr)
            m = np.mean(model.train[key], axis=0).T

            bk[key] = inv_cov_matr.dot(m).reshape((-1, 1))
            ck[key] = -0.5 * m.T.dot(inv_cov_matr).dot(m) + np.log(1 / len(model.train.keys()))

        res = []
        classes = model.data.keys()
        for k in classes:
            res.append(np.exp(lst.dot(bk[k]) + ck[k]))

        return np.concatenate(res, axis=1)

    def calc_points(points: np.array):
        df = pd.DataFrame(columns=['x1', 'x2', '1-2', '1-3', '1-4', '2-3', '2-4', '3-4'])
        df['x1'], df['x2'] = lst[:, 0], lst[:, 1]
        df['1-2'] = np.isclose(points[:, 0], points[:, 1], atol=1e-4)
        df['1-3'] = np.isclose(points[:, 0], points[:, 2], atol=1e-4)
        df['1-4'] = np.isclose(points[:, 0], points[:, 3], atol=1e-4)
        df['2-3'] = np.isclose(points[:, 1], points[:, 2], atol=1e-4)
        df['2-4'] = np.isclose(points[:, 1], points[:, 3], atol=1e-4)
        df['3-4'] = np.isclose(points[:, 2], points[:, 3], atol=1e-4)

        return df

    def draw_plots(title: str, data: pd.DataFrame, leg=True):
        _, ax = plt.subplots()
        ax.grid()
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(title)

        g = []
        colors = []
        for k, v in model.train.items():
            a = ax.scatter(v[:, 0], v[:, 1], marker='.', label=f'Class {k}')
            c = a.get_facecolor()[0]
            g.append(a)
            colors.append(c)

            g.append(ax.scatter(model.test[k][:, 0], model.test[k][:, 1], marker='x', s=20, color=c))

        for k in data.columns[2:]:
            v = data[data[k] == 1]
            g.append(ax.scatter(v.x1, v.x2, marker='.', s=5, label=f'Separator {k}'))

        if leg:
            plt.legend(handles=g)

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))

        plt.show()

    def a():
        matr, _ = model.calc_shared_scalar_cov()
        data = []
        for k in model.data.keys():
            data.append((k, matr))

        draw_plots('Shared & Scalar', calc_points(abe_discr(data)))

    def b():
        matr, _ = model.calc_shared_diagonal_cov()
        data = []
        for k in model.data.keys():
            data.append((k, matr))

        draw_plots('Shared & Diagonal', calc_points(abe_discr(data)))

    def c():
        data = []
        for k, v in model.train.items():
            matr, _ = model.calc_scalar_cov(k)
            data.append((k, matr))

        draw_plots('Not shared & Scalar', calc_points(cdf_discr(data)), False)

    def d():
        data = []
        for k, v in model.train.items():
            matr, _ = model.calc_diagonal_cov(k)
            data.append((k, matr))

        draw_plots('Not shared & Diagonal', calc_points(cdf_discr(data)), False)

    def e():
        matr, _ = model.calc_shared_cov()
        data = []
        for k in model.train.keys():
            data.append((k, matr))

        draw_plots('Shared', calc_points(abe_discr(data)))

    def f():
        data = []
        for k in model.data.keys():
            matr, _ = model.calc_cov(k)
            data.append((k, matr))
        draw_plots('No assumptions', calc_points(cdf_discr(data)), False)

    lst = product(np.arange(model.orig_data[:, 0].min(), model.orig_data[:, 0].max(), 0.005),
                  np.arange(model.orig_data[:, 1].min(), model.orig_data[:, 1].max(), 0.005))
    # lst = product(np.arange(model.orig_data[:, 0].min(), model.orig_data[:, 0].max(), 0.01),
    #               np.arange(model.orig_data[:, 1].min(), model.orig_data[:, 1].max(), 0.01))
    lst = np.array(list(lst))

    a()
    b()
    c()
    d()
    e()
    f()


test = Lab2("data_v2-04.csv")
# data_representation(test)
# task_1(test)
# task_2_4_5(test)
# task_3(test)
