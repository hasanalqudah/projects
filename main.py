import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import persptron
import math
import ada
import sgd_algorithum
from sklearn import preprocessing


"""
this fuction used to print the success and errors

input predicted data and orginal data and the number of instances
"""

def print_error( predictarray, orginarray ,num):
    count1=0
    for i in range(len(orginarray)):
        if predictarray[i] != orginarray[i]:
          count1 = count1+1
    errorrate1 =(count1/num)*100/100
    print('success Rate=   '+ str(round(1-errorrate1,2))+'%')
    print('error Rate=     '+ str(round(errorrate1,2))+'%')



def calc_error_multiclass( predictarray, orginarray ,num,cc):
    ct2=0
    ct1=0
    add=[]
    if num==1:
        for i in range(len(predictarray)):
            if predictarray[i] == orginarray[i] and predictarray[i]==-1:
                ct1 = ct1+1
    if num==2:
        for i in range(len(predictarray)):
            if predictarray[i] == orginarray[i]and predictarray[i]==-1:
                ct1 = ct1+1
    if num==3:
        for i in range(len(predictarray)):
           if predictarray[i] == orginarray[i]and predictarray[i]==-1:
                ct1 = ct1+1
    for j in range(len(orginarray)):
        if orginarray[j]==-1:
            ct2=ct2+1

    print(' # of instances in class  '+str(num)+' that classified correctly are  '+str(ct1)+" from  "+str(ct2)+ '  with percentage of '+str((ct1/ct2)*100/100)+'%')



#*************************************************************************************8

"""
this is the multiclass function which generate models according to different clases
"""
def multiclass(file):

    if file=='seedtrain.csv':
        cl1=1
        cl2=2
        cl3=3
        testfile='seedtest.csv'
    else:
        cl1='Iris-setosa'
        cl2='Iris-versicolor'
        cl3='Iris-virginica'
        testfile='iristest.csv'
    eta=float(input("please input eta"))
    niter=int(input("please number of iteration"))

    dfm = pd.read_csv(file,header=None)
    nr = int(dfm.shape[0]) #obtain number of rows from whole dataset
    ncl = int(dfm.shape[1])# obtain the number of colomun from dataset
    xtrainm = dfm.iloc[0:nr, 0:ncl-1].values
     #divide y to 3 different classes
    ytrainc1 = dfm.iloc[0:nr, ncl-1].values
    ytrainc1 = np.where(ytrainc1 == cl1, -1,1)

    ytrainc2 = dfm.iloc[0:nr, ncl-1].values
    ytrainc2 = np.where(ytrainc2 == cl2, -1,1)

    ytrainc3 = dfm.iloc[0:nr, ncl-1].values
    ytrainc3 = np.where(ytrainc3 == cl3, -1,1)

    Tobject1=sgd_algorithum.SGD(eta,niter,random_state=1)
    Tobject2=sgd_algorithum.SGD(eta,niter,random_state=1)
    Tobject3=sgd_algorithum.SGD(eta,niter,random_state=1)

    Tobject1.fit(xtrainm,ytrainc1)
    Tobject2.fit(xtrainm,ytrainc2)
    Tobject3.fit(xtrainm,ytrainc3)


    df11 = pd.read_csv(testfile,header=None)

    xtesting=df11.iloc[0:nr, 0:ncl-1].values


    class1 = df11.iloc[0:nr, ncl-1].values
    class1 = np.where(class1 == cl1, -1,1)

    class2 = df11.iloc[0:nr, ncl-1].values
    class2 = np.where(class2 == cl2, -1,1)

    class3 = df11.iloc[0:nr, ncl-1].values
    class3 = np.where(class3 == cl3, -1,1)

    a1=Tobject1.predict(xtesting)
    a2=Tobject2.predict(xtesting)
    a3=Tobject3.predict(xtesting)

    calc_error_multiclass(a1,class1,1,a1.shape[0])
    calc_error_multiclass(a2,class2,2,a2.shape[0])
    calc_error_multiclass(a3,class3,3,a3.shape[0])
    print("class 1 statistics:")
    print_error(a1,class1,a1.shape[0])
    print("class 2 statistics:")
    print_error(a1,class1,a1.shape[0])
    print("class 3 statistics:")
    print_error(a1,class1,a1.shape[0])


#multiclass



"""
this function used to call every classifier according to user request
"""

def call_classifier(classifierName,file):

    if file=='seedtrain':
        class1=1
        testfile='seedtest'
    else:
        class1='Iris-setosa'
        testfile='iristest.csv'


    if classifierName=='perceptron':
        eta=float(input("please input eta"))
        niter=int(input("please number of iteration"))

        df = pd.read_csv(file, header=None)
        nr = int(df.shape[0]) #obtain number of rows from whole dataset
        ncl = int(df.shape[1])# obtain the number of colomun from dataset
        xtrain = df.iloc[0:nr, 0:ncl-1].values
        ytrain = df.iloc[0:nr, ncl-1].values
        ytrain = np.where(ytrain == class1, -1,1)
        df1 = pd.read_csv(testfile,header=None)
        nr1 = int(df1.shape[0]) #obtain number of rows from whole dataset
        ncl1 = int(df1.shape[1])# obtain the number of colomun from dataset
        xtest = df1.iloc[0:nr1, 0:ncl1-1].values
        ytest = df1.iloc[0:nr1, ncl1-1].values
        ytest = np.where(ytest == class1, -1, 1)
        ob=persptron.Perceptron(eta,niter)
        ob.fit(xtrain,ytrain)
        arr=ob.predict(xtest)
        print_error(arr,ytest,nr1)
        plt.plot(range(1,len(ob.errors_)+1),ob.errors_)
        plt.xlabel('attempts')
        plt.title('Perceptron')
        plt.ylabel('number of misclassification')
        plt.show()


    if classifierName=='addline':
        eta=float(input("please input eta"))
        niter=int(input("please number of iteration"))

        df = pd.read_csv(file,header=None)
        nr = int(df.shape[0]) #obtain number of rows from whole dataset
        ncl = int(df.shape[1])# obtain the number of colomun from dataset
        xtrain = df.iloc[0:nr, 0:ncl-1].values
        #preprocessing.normalize(xtrain,'l2')
        ytrain = df.iloc[0:nr, ncl-1].values
        ytrain = np.where(ytrain == class1, -1,1)
        df1 = pd.read_csv(testfile,header=None)
        nr1 = int(df1.shape[0]) #obtain number of rows from whole dataset
        ncl1 = int(df1.shape[1])# obtain the number of colomun from dataset
        xtest = df1.iloc[0:nr1, 0:ncl1-1].values
        #preprocessing.normalize(xtest,'l2')
        ytest = df1.iloc[0:nr1, ncl1-1].values
        ytest = np.where(ytest == class1, -1, 1)
        ob1=ada.Adaline(eta,niter)
        ob1.fit(xtrain,ytrain)
        arr=ob1.predict(xtest)
        print_error(arr,ytest,nr1)
        plt.plot(range(1,len(ob1.cost_)+1),ob1.cost_)
        plt.title('Addline')
        plt.xlabel('attempts')
        plt.ylabel('number of misclassification')
        plt.show()

    if classifierName=='sgd':
        eta=float(input("please input eta"))
        niter=int(input("please number of iteration"))

        df = pd.read_csv(file,header=None)
        nr = int(df.shape[0]) #obtain number of rows from whole dataset
        ncl = int(df.shape[1])# obtain the number of colomun from dataset
        xtrain = df.iloc[0:nr, 0:ncl-1].values
        preprocessing.normalize(xtrain,'l2')
        ytrain = df.iloc[0:nr, ncl-1].values
        ytrain = np.where(ytrain == class1, -1,1)
        df1 = pd.read_csv(testfile,header=None)
        nr1 = int(df1.shape[0]) #obtain number of rows from whole dataset
        ncl1 = int(df1.shape[1])# obtain the number of colomun from dataset
        xtest = df1.iloc[0:nr1, 0:ncl1-1].values
        #preprocessing.normalize(xtest,'l2')
        ytest = df1.iloc[0:nr1, ncl1-1].values
        ytest = np.where(ytest == class1, -1, 1)
        ob2=sgd_algorithum.SGD(eta,niter)
        ob2.fit(xtrain,ytrain)
        arr=ob2.predict(xtest)


        print_error(arr,ytest,nr1)
        plt.plot(range(1,len(ob2.cost_)+1),ob2.cost_)
        plt.title('stochastic gradient descent')
        plt.xlabel('attempts')
        plt.ylabel('number of misclassification')
        plt.show()

#************************************************************************************************
op = input("PLEASE ENTER 1 FOR CLASSIFICATION 2 FOR MULTI INSTANCE CLASSIFICATION 0 TO EXIT)\n ")
while op!='0':
    if op=='1':

        DATAFILE = input("PLEASE ENTER DATA file:(type: Iriss or seed)").lower()
        if DATAFILE == 'iriss':
            DATAFILE='iristrain.csv'

        elif DATAFILE == 'seed':
            DATAFILE='iristrain.csv'
        else:
            print("file input error")
        classifier = input("PLEASE ENTER CLASSIFIER NAME:(Perceptron,Addline,SGD)\n ").lower()
        if classifier == 'perceptron' or classifier == 'addline' or classifier == 'sgd':
            if classifier == 'perceptron':
                call_classifier(classifier , DATAFILE)
            elif classifier == 'addline':

                call_classifier(classifier , DATAFILE)
            elif classifier == 'sgd':
                call_classifier(classifier , DATAFILE)
            else:
                print("classifier name error")
    if op=='2':
        DATAFILE = input("PLEASE ENTER DATA file:(type: Iriss  or seed) ").lower()
        if DATAFILE == 'iriss':
            DATAFILE='iristrain.csv'

        if DATAFILE == 'seed':
            DATAFILE='seedtrain.csv'

        multiclass(DATAFILE)



    op = input("PLEASE ENTER 1 FOR CLASSIFICATION 2 FOR MULTI INSTANCE CLASSIFICATION 0 TO EXIT)\n ")
print("done!!!!!")










