import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
import csv
import matplotlib.patches
import matplotlib.lines
import matplotlib.path
import seaborn as sb
import pandas as pd
import  sklearn.metrics as metr
import math



class ByesClassificator(object):

    def __init__(self, path):
        self.dict = {1: [[],[]], 2: [[],[]], 3: [[],[]], 4:[[],[]]}
        with open(path, 'r') as fp:
            reader = csv.reader(fp, delimiter=',', quotechar='"')
            next(reader, None)  # skip the headers
            data_reader = [list(map(float, row)) for row in reader]
            data_reader = np.array(data_reader)
        x = data_reader[:, 0]
        y = data_reader[:, 1]
        label = data_reader[:, 2]

        self.data = [x,y,label]
        self.dict_train = {}
        self.dict_test = {}
        self.a={}
        self.b={}
        self.c={}
        self.s = {}
        self.roc_auc_mat = []
        self.mat_a_b_acc_class={}
        self.decision_func = []
        for i in range(0, label.size):
            self.dict[label[i]][0].append(x[i])
            self.dict[label[i]][1].append(y[i])
        #print(self.dict)
        #self.dict_train = self.dict
        #self.dict_test = self.dict

    def new_train_test_split(self):
        d = self.dict
        for i in self.dict.keys():
            d_train0, d_test0, d_train1, d_test1  = train_test_split(d[i][0],d[i][1], test_size=0.3)
            self.dict_train[i]=[d_train0,d_train1]
            self.dict_test[i] = [d_test0,d_test1]
        return

    def get_cov_matrix_naive_shared(self):
        sum = 0
        k=0
        di = self.dict_train
        for i in di.keys():
            sum +=np.var(di[i][0])+np.var(di[i][1])
            k+=2
        return np.eye(2, dtype=float)*(sum/k)

    def get_cov_matrix_diagonal_shared(self):
        sum0 = 0
        sum1 = 0
        k = 0
        di = self.dict_train
        for i in di.keys():
            sum0 += np.var(di[i][0])
            sum1 += np.var(di[i][1])
            k += 1
        sum0 /=k
        sum1 /= k
        m=np.eye(2, dtype=float)
        m[0][0]*= sum0
        m[1][1] *= sum1
        return  m

    def get_cov_matrix_scalar(self):
        m = np.eye(4, dtype=float)
        di = self.dict_train
        for i in di.keys():
            m[i-1][i-1] = (np.var(di[i][0])+np.var(di[i][1]))/2
        return  m



    def show_data(self):

        fig, ax = plt.subplots()

        ax.set_xlabel('X1')
        ax.set_ylabel('X2')

        x_min = np.min(self.data[0])
        x_max = np.max(self.data[0])

        y_min = np.min(self.data[1])
        y_max = np.max(self.data[1])

        mini = np.min([x_min, y_min])
        maxi = np.max([x_max, y_max])
        ax.set_xlim(mini, maxi)
        ax.set_ylim(mini, maxi)
        ax.grid()

        colors = ['red', 'green', 'blue', 'm']
        leg = []
        for i,v in self.dict.items():
            leg.append(ax.scatter(v[0], v[1], s=2))
        plt.legend((leg[0], leg[1], leg[2], leg[3]),
                       ('Класс 1', 'Класс 2', 'Класс 3', 'Класс 4'))
        plt.show()



    def show_matrix(self, circles):
        fig, ax = plt.subplots()

        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        x_min = np.min(self.data[0])
        x_max = np.max(self.data[0])

        y_min = np.min(self.data[1])
        y_max = np.max(self.data[1])

        mini = np.min([x_min,y_min])
        maxi = np.max([x_max, y_max])
        ax.set_xlim(mini,  maxi)
        ax.set_ylim(mini,  maxi)
        ax.grid()
        k=0
        ang=0
        colors = ['red', 'green', 'blue','m']
        leg=[]
        for i in circles:
            x_c= i[0]
            y_c = i[1]
            w=2*i[2]
            h=2*i[3]
            if len(i)==5:
                ang = i[4]
            delta_w=w*0.25
            delta_h = h * 0.25
            while w>0 and h>0:
                c = matplotlib.patches.Ellipse((x_c,y_c),
                                                width=w,
                                                height=h,
                                                angle=ang,
                                                fill=False,
                                                color=colors[k])
                w -= delta_w
                h -= delta_h
                ax.add_patch(c)

            k+=1
            leg.append(ax.scatter(self.dict_train[k][0], self.dict_train[k][1], s=2, color=colors[k-1]))
        plt.legend((leg[0],leg[1], leg[2],  leg[3]), ('РљР»Р°СЃСЃ 1', 'РљР»Р°СЃСЃ 2', 'РљР»Р°СЃСЃ 3','РљР»Р°СЃСЃ 4'))
        plt.show()

    def show_matrix_12(self, m):
        c = []
        for i in range(1,5):
            mx = np.mean(self.dict_train[i][0])
            my = np.mean(self.dict_train[i][1])
            c.append([mx,my,m[0][0],m[1][1]])
        self.show_matrix(c)

    def show_matrix_3(self):
        c = []

        for i in range(1, 5):
            m, a = self.get_m3(i)
            mx = np.mean(self.dict_train[i][0])
            my = np.mean(self.dict_train[i][1])
            c.append([mx, my, m[0][0], m[1][1],a])
        self.show_matrix(c)

    def show_all_matrix(self):
        m = self.get_cov_matrix_naive_shared()
        self.show_matrix_12(m)
        m = self.get_cov_matrix_diagonal_shared()
        self.show_matrix_12(m)

        self.show_matrix_3()

    def get_priors(self):
        priors = []
        k=0
        for i in range(1,5):

            n=len(self.dict_train[i][0])
            k+=n
            priors.append(n)
        priors = np.asarray(priors)/k
        #priors = [0.25,0.25,0.25,0.25]
        return priors

    def get_m3(self,k):
        x=self.dict_train[k][0]
        y=self.dict_train[k][1]
        cov_mat = np.stack((x, y), axis=0)
        cov=np.cov(cov_mat)
        vx = np.var(x)
        vy = np.var(y)
        angel = math.atan(2*cov[0][1]/(vx-vy))/2

        return np.cov(cov_mat),angel*180/math.pi


    def get_m2(self):
        arr=[]
        for k in range(1,5):
            cov_mat = np.stack((self.dict_train[k][0], self.dict_train[k][1]), axis=0)
            arr.append(np.cov(cov_mat))
        av = np.mean(arr,axis=0)
        return av
    def sigma(self, alpha, betha,k):
        m1 = self.get_cov_matrix_naive_shared()
        #m2=self.get_m2()
        m2 = self.get_cov_matrix_diagonal_shared()
       # m3 = self.get_cov_matrix_scalar()
        m3,a=self.get_m3(k)
      # m4=np.eye(2, dtype=float)*m3[k-1][k-1]
        res = alpha*m1 + betha*m2 + (1-alpha-betha)*m3
        return res

    def culc_coeff(self, alpha, betha):
        for k in self.dict.keys():
            s=self.sigma(alpha, betha,k)
            self.s[k]=s
            self.a[k]=self.A(s)
            self.b[k] = self.B(s,k)
            self.c[k] = self.C(s, k)
        return



    def A(self,sigma):
        res = np.linalg.inv(sigma)
        return -0.5 * res

    def B(self,sigma,k):
        m1 = np.mean(self.dict_train[k][0])
        m2 = np.mean(self.dict_train[k][1])

        return np.dot(np.linalg.inv(sigma),[[m1],[m2]])

    def C(self,sigma,k):
        m1 = np.mean(self.dict_train[k][0])
        m2 = np.mean(self.dict_train[k][1])
        x_m = [[m1],[m2]]
        res1=np.linalg.inv(sigma)
        res2=-0.5*np.transpose(x_m)[0]
        res3=np.dot(res2,res1)
        res4=np.dot(res3,x_m)
        res5=np.log(np.linalg.det(sigma))
        res6=np.log(0.25)
        res = res4[0]  - 0.5* res5+ res6
        return res

    def discriminant_func(self,x,k,alpha,betha):
        #S = self.sigma(alpha, betha,k)
        #p=self.get_priors()
        xt=np.transpose(x)[0]
        A=self.a[k]
        res1 = np.dot(xt,A)
        res1 = np.dot(res1,x)[0]
        res2 = np.dot(np.transpose(self.b[k])[0],x)[0]
        res3=self.c[k]
        res =  res1 + res2 + res3
        return res

    def get_class(self, x, alpha, betha):
        score = -100000000
        arr=[0,0,0,0]
        cl=None
        for i in range(1,5):
            d=self.discriminant_func(x, i, alpha, betha)
            if score < d:
                cl=i
                score=d
            arr[i-1]=math.exp(d)

        return cl, arr

    # РїСЂРёРјРµРЅСЏРµРј РєР»Р°СЃСЃРёС„РёРєР°С‚РѕСЂ РєРѕ РІСЃРµР№ РІС‹Р±РѕСЂРєРµ
    def get_accuracy(self, alpha, betha, d,size):

        self.decision_func.clear()
        good=0
        all=0
        arr = {1:[],2:[],3:[],4:[]}

        for i,v in d.items():
            x, y = v
            k=0
            for j in range(0, len(x)):
                #if all > 150:
                   # break
                cl,f  = self.get_class([[x[j]],[y[j]]], alpha, betha)
                fsum = np.sum(f)
                if cl==i:
                    arr[i].append([x[j],y[j]])
                    good+=1
                af = np.concatenate(([i],f/fsum))
                self.decision_func.append(af) #Р·Р°РїРѕР»РЅСЏРµРј С„СѓРЅРєС†РёСЋ СЂРµС€РµРЅРёСЏ РґР»СЏ СЂР°СЃС‡РµС‚Р° roc_auc
                all+=1
                k+=1
        acc = good/all

        df = np.array(self.decision_func)
        y_test = df[:, 0]
        des_func = np.delete(df, 0, 1)
        auc = metr.roc_auc_score(y_true=y_test, y_score=des_func, average='macro', multi_class='ovo' )

        self.mat_a_b_acc_class[(alpha, betha)] = [acc,arr,auc] # РґР»СЏ РІС‹РїРѕР»РЅРµРЅРёСЏ С‚СЂРµС‚СЊРµРіРѕ Р·Р°РґР°РЅРёСЏ

        return acc, auc


    # СЂР°СЃС‡РµС‚ С‚РѕС‡РЅРѕСЃС‚Рё РґР»СЏ РІСЃРµС… Р°Р»СЊС„Р° Рё Р±РµС‚Р°
    def get_statistics(self, d, size=500, arr=[0, 0.01, 0.001, 0.1, 0.2, 0.3, 0.5, 1]):  # ,0.8,0.9,1]):
        self.mat_a_b_acc_class.clear()
        # arr = [0.1,0.01,0.001,0.2,0.02,0.002,0.3,0.03,0.003,0.5,0.05,0.005]

        l = len(arr)
        auc_mat = np.zeros((l, l), dtype=float)
        mat = np.zeros((l, l), dtype=float)
        max_acc = 0
        max_auc = 0
        for i in arr:
            for j in arr:
                if j + i > 1:
                    break
                self.culc_coeff(i, j)
                acc, auc = self.get_accuracy(i, j, d, size)

                if acc > max_acc:
                    max_acc = acc

                if auc > max_auc:
                    max_auc = auc

                mat[arr.index(i)][arr.index(j)] = acc
                auc_mat[arr.index(i)][arr.index(j)] = auc

                print([i, j, acc])
        return mat, max_acc, auc_mat, max_auc



    def draw_heat_map_matrix(self,m):
        arr = [0, 0.01, 0.001, 0.1, 0.2, 0.3, 0.5, 1]  # , 0.8, 0.9, 1]
        df = pd.DataFrame(m, index=arr, columns=arr)
        heat_map = sb.heatmap(df)
        plt.show()

    def task_3ab(self, max, mode):
        fig, ax = plt.subplots()

        ax.grid()
        x,y = [],[]
        di={}
        if mode=='test':
            di=self.dict_test
        else:
            di = self.dict_train
        for i,v in di.items():
            x+=v[0]
            y+=v[1]
        ax.scatter(x,y, s=2, color='k')
        k = 0
        c = ['r','g','b','m']
        for i,v in self.mat_a_b_acc_class.items():
            if v[0]==max:
                for j,a in v[1].items():
                    arr=np.array(a, dtype='float')
                    ax.scatter(arr[:,0],arr[:,1], s=2, color=c[j-1])
        plt.show()

    def task_5(self, max, mode):
        fig, ax = plt.subplots()

        ax.grid()
        x,y = [],[]
        di={}
        if mode=='test':
            di=self.dict_test
        else:
            di = self.dict_train
        for i,v in di.items():
            x+=v[0]
            y+=v[1]
        ax.scatter(x,y, s=2, color='gray')
        k = 0
        c = ['r','g','b','m']
        for i,v in self.mat_a_b_acc_class.items():
            if v[2]==max:
                for j,a in v[1].items():
                    arr=np.array(a, dtype='float')
                    ax.scatter(arr[:,0],arr[:,1], s=2, color=c[j-1])
        xx,yy,Z = self.show_class_boundaries()
        plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
        plt.show()




    def show_class_boundaries(self):
        #fig, ax = plt.subplots()
        C = 1.0  # SVM regularization parameter
        clf = svm.SVC(kernel='rbf', gamma=0.7, C=C)
        X=np.array(self.data)[0,:]
        X=X.reshape(-1, 1)
        Y = np.array(self.data)[1,:]
        Y=Y.reshape(-1, 1)
        X=np.hstack((X,Y))
        Y = np.array(self.data)[2, :]
        clf.fit(X, Y)

        h = .02  # step size in the mesh
        # create a mesh to plot in
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
       # dis = clf.decision_function(X)

        return xx,yy,Z
        #ax.scatter(np.array(self.data)[0,:], np.array(self.data)[1,:], s=2, color='k')

        #plt.show()

    def task_3vgd(self,mode):
        fig, ax = plt.subplots()

        ax.grid()
        x, y = [], []
        di = {}
        if mode == 'test':
            di = self.dict_test
        else:
            di = self.dict_train

        for i, v in di.items():
            x += v[0]
            y += v[1]
        ax.scatter(x, y, s=2, color='gray')

        for i, v in self.mat_a_b_acc_class.items():
            if i != (0,0):
                continue

            for j, a in v[1].items():
                arr = np.array(a, dtype='float')
                ax.scatter(arr[:, 0], arr[:, 1], s=2)


        xx,yy,Z = self.show_class_boundaries()
        plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
        plt.show()

        fig, ax = plt.subplots()
        ax.scatter(x, y, s=2, color='gray')
        for i, v in self.mat_a_b_acc_class.items():
            if i != (0,1):
                continue

            for j, a in v[1].items():
                arr = np.array(a, dtype='float')
                ax.scatter(arr[:, 0], arr[:, 1], s=2)


        plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

        plt.show()

        fig, ax = plt.subplots()
        ax.scatter(x, y, s=2, color='gray')
        for i, v in self.mat_a_b_acc_class.items():
            if i != (1, 0):
                continue

            for j, a in v[1].items():
                arr = np.array(a, dtype='float')
                ax.scatter(arr[:, 0], arr[:, 1], s=2)

        plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

        plt.show()



def main():
    cl = ByesClassificator("data_v2-05.csv")
    cl. new_train_test_split()
    cl.show_data()


    # Задание 1
    cl.show_all_matrix()


    print("TRAIN")

    # Задание 2
    m_tr, max_tr, auc_tr, max_auc_tr = cl.get_statistics(cl.dict_train)
    cl.draw_heat_map_matrix(m_tr)

    # Задание 3
    cl.task_3ab(max_tr,'train')
    cl.task_3vgd('train')

    # Задание 4
    cl.draw_heat_map_matrix(auc_tr)

    # Задание 5
    cl.task_5(max_auc_tr, 'train')



    print("TEST")

    # Задание 2
    m_ts, max_ts, auc_ts, max_auc_ts = cl.get_statistics(cl.dict_test)
    cl.draw_heat_map_matrix(m_ts)

    # Задание 3
    cl.task_3ab(max_ts, 'test')
    cl.task_3vgd('test')

    # Задание 4
    cl.draw_heat_map_matrix(auc_ts)

    # Задание 5
    cl.task_5(max_auc_ts, 'test')

main()