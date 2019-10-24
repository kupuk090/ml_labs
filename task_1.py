import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib.axes import Axes
import csv


class RegressionModel(object):

    def __init__(self, path):

        with open(path, 'r') as fp:
            reader = csv.reader(fp, delimiter=',', quotechar='"')
            next(reader, None)  # skip the headers
            data_reader = [list(map(float, row)) for row in reader]
            data_reader = np.array(sorted(data_reader, key=lambda y: y[0]))
        x = data_reader[:,0].reshape(-1, 1)
        y = data_reader[:,1].reshape(-1, 1)
        self.x = (x-np.mean(x)) / np.std(x)
        self.y = (y - np.mean(y)) / np.std(y)

    def show_rmodel(self, x_pred, y_pred, save_path = None):
        y_pred = [y_pred for _, y_pred in sorted(zip(x_pred, y_pred))]
        x_pred = sorted(x_pred)
        plt.scatter(self.x, self.y)
        plt.plot(x_pred, y_pred, 'r');
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        plt.clf()
        return

    def new_train_test_split(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3)
        return

    def get_mse(self,y,y_pred):
        return mean_squared_error(y,y_pred)

    def linear_regression(self, poly_degree):
        lin_model = LinearRegression()
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        x_new_train = poly.fit_transform(self.x_train)
        x_new_test = poly.fit_transform(self.x_test)
        lin_model.fit(x_new_train, self.y_train)
        self.y_pred_train = lin_model.predict(x_new_train)
        self.y_pred_test = lin_model.predict(x_new_test)
        self.y_pred_all = lin_model.predict( poly.fit_transform(self.x))
        #self.y_pred_lin = model.predict(X_new[self.fit_size:])
        return lin_model

    def lasso_regression(self, poly_degree, alpha):
        lasso_model = Lasso(alpha)
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        x_new_train = poly.fit_transform(self.x_train)
        x_new_test = poly.fit_transform(self.x_test)
        lasso_model.fit(x_new_train, self.y_train)
        self.y_pred_train = lasso_model.predict(x_new_train)
        self.y_pred_test = lasso_model.predict(x_new_test)
        self.y_pred_all = lasso_model.predict(poly.fit_transform(self.x))
        # self.y_pred_lin = model.predict(X_new[self.fit_size:])
        return lasso_model
def task_1(path):
    model = RegressionModel(path)
    model.new_train_test_split()

    task_1a(model)
    task_1bc(model,0.01)
    task_1bc(model, 1)



def task_1a(model):
    mse_arr_test = []
    mse_arr_train = []
    #model.new_train_test_split()
    for m in range(1,11):
        model.linear_regression(m)
        #model.show_rmodel(model.x_train, model.y_pred_train)
        #model.show_rmodel(model.x_test, model.y_pred_test)
        mse_arr_test.append(model.get_mse(model.y_test,model.y_pred_test))
        mse_arr_train.append(model.get_mse(model.y_train, model.y_pred_train))

    fig, ax = plt.subplots()
    ax.grid()
    ax.set_xlabel('Степень многочлена регрессионной модели')
    ax.set_ylabel('Среднеквадратичная ошибка')
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlim(0,12)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.set_ylim(0.5, 1.1)
    line_up,=ax.plot(np.array(range(1, 11)), mse_arr_train, "r", marker=".", label = 'Обучающая выборка');
    line_down,=ax.plot(np.array(range(1, 11)), mse_arr_test, "b", marker="s", label = 'Тестовая выборка');
    plt.legend(handles=[line_up, line_down])

    plt.show()

    """
    plt.savefig('task1/a_train.png')
    plt.clf()
    plt.plot(np.array(range(1,10)), mse_arr_test, 'r');
    plt.show()
    #plt.savefig('task1/a_test.png')
    #plt.clf()
 """
def task_1bc(model,alpha):
    mse_arr_test = []
    mse_arr_train = []
    # model.new_train_test_split()
    for m in range(1, 11):

        model.lasso_regression(m,alpha)
       # model.show_rmodel(model.x_train, model.y_pred_train)
        #model.show_rmodel(model.x_test, model.y_pred_test)
        mse_arr_test.append(model.get_mse(model.y_test, model.y_pred_test))
        mse_arr_train.append(model.get_mse(model.y_train, model.y_pred_train))

    fig, ax = plt.subplots()
    ax.grid()
    ax.set_xlabel('Степень многочлена регрессионной модели')
    ax.set_ylabel('Среднеквадратичная ошибка')
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlim(0, 12)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.set_ylim(0.5, 1.1)
    line_up, = ax.plot(np.array(range(1, 11)), mse_arr_train, "r", marker=".", label='Обучающая выборка');
    line_down, = ax.plot(np.array(range(1, 11)), mse_arr_test, "b", marker="s", label='Тестовая выборка');
    plt.legend(handles=[line_up, line_down])

    plt.show()


def task_2(path):
    model = RegressionModel(path)
    model.new_train_test_split()
    m=3

    model.linear_regression(m)
    model.show_rmodel(model.x_test,model.y_pred_test,'task2_a_' + str(m) + '_test.png')


    model.lasso_regression(m, 0.1)
    model.show_rmodel(model.x_test, model.y_pred_test,'task2_bc_' + str(0.1) + '_test.png')

    model.lasso_regression(m, 0.001)
    model.show_rmodel(model.x_test, model.y_pred_test, 'task2_bc_' + str(0.001) + '_test.png')

def task3(path):
    model = RegressionModel(path)
    model.new_train_test_split()
    m=3
    w0,w1,w2,w3 = [],[],[],[]
    digits = np.arange(0, 10, 0.1)
    for alpha in digits:
        lm=model.lasso_regression(m, alpha)
        #model.show_rmodel(model.x_test, model.y_pred_test)
        w0.append(lm.intercept_)
        c = lm.coef_
        w1.append(c[0])
        w2.append(c[1])
        w3.append(c[2])
    for g in [w0,w1,w2,w3]:
        plt.plot(digits,g)
    plt.savefig('task3/trace_plot_'+str(m)+'.png')
    plt.show()
    plt.clf()

def task_4a(model):
    attempt_number = 100
    m=3
    train = {}
    test = {}
    model_x = set([ y for x in model.x for y in x])
    for i in model_x:
        train[i] = []
        test[i] = []
    x_train = [ y for x in model.x for y in x]
    for i in range(100):
        model.new_train_test_split()
        model.linear_regression(m)
        for j in range(210):
            train[model.x_train[j][0]].append(model.y_pred_train[j][0])
        for j in range(90):
            test[model.x_test[j][0]].append(model.y_pred_test[j][0])
    train_var=[]
    test_var=[]
    for i in model_x:
        train_var.append(np.var(train[i]))
        test_var.append(np.var(test[i]))
    hist, bins = np.histogram(train_var, bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()
def task4(path):
    model = RegressionModel(path)
    task_4a(model)

def task_5a(model):
    m=9
    model.new_train_test_split()
    model.linear_regression(m)
    dif_test =  model.y_test - model.y_pred_test
    dif_train =  model.y_train - model.y_pred_train

    y_pred = [dif_test  for _, dif_test  in sorted(zip(model.x_test, dif_test ))]
    x_pred = sorted(model.y_pred_test)
    plt.scatter(x_pred, y_pred)
    plt.savefig('task5/lin_test.png')
    plt.show()
    plt.clf()

    y_pred = [dif_train for _, dif_train in sorted(zip(model.x_train, dif_train))]
    x_pred = sorted(model.y_pred_train)
    plt.scatter(x_pred, y_pred)
    plt.savefig('task5/lin_train.png')
    plt.show()
    plt.clf()

def task5(path):
    model = RegressionModel(path)
    task_5a(model)

def main():
    #task_2('data_v1-05.csv')

    path = 'data_v1-05.csv'
    task_2(path)
'''
    model = RegressionModel(path)
    model.new_train_test_split()
    lm=model.lasso_regression(9,0.5)
    model.show_rmodel(model.x_test, model.y_pred_test)
    model.show_rmodel(model.x_train, model.y_pred_train)
'''


main()