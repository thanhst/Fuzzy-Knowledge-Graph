import pandas as pd
import numpy as np 
def combination(k, n):
    if k == 0 or k == n:
        return 1
    if k == 1:
        return n
    return combination(k - 1, n - 1) + combination(k, n - 1)


def caculateA(base):
    colum = len(base[0])
    row = len(base)
    A = np.zeros((row, combination(4, colum - 1)))

    for r1 in range(row):
        k = [0] * combination(4, colum - 1)
        temp = 0
        for a in range(0, colum - 4):
            for b in range(a + 1, colum - 3):
                for c in range(b + 1, colum - 2):
                    for d in range(c + 1, colum - 1):
                        for r2 in range(row):
                            if base[r1][a] == base[r2][a] and base[r1][b] == base[r2][b] and base[r1][c] == base[r2][c] and base[r1][d] == base[r2][d]:
                                k[temp] += 1

                        A[r1][temp] = k[temp] / row
                        temp += 1
    print("done A")
    return A


def caculateM(base):
    colum = len(base[0])
    row = len(base)
    M = np.zeros((row, colum - 1))
    for t1 in range(row):
        k = [0] * (colum - 1)
        temp = 0
        for i in range(colum - 1):
            for t2 in range(row):
                if base[t1][i] == base[t2][i] and base[t1][colum - 1] == base[t2][colum - 1]:
                    k[temp] += 1
            M[t1][temp] = k[temp] / row
            temp += 1

    return M


def caculateB(base, A, M):
    colum = len(base[0])
    row = len(base)
    B = np.zeros((row, combination(3, colum - 1)))

    for r in range(row):
        temp = 0
        for a in range(0, colum - 3):
            for b in range(a + 1, colum - 2):
                for c in range(b + 1, colum - 1):
                    B[r][temp] = sum(A[r]) * min(M[r][a], M[r][b], M[r][c])
                    temp += 1
    print("done B")
    return B


def caculateC(base, B):
    colum = len(base[0])
    row = len(base)
    cols = 2 * combination(3, colum - 1)
    C = np.zeros((row, cols))

    for r1 in range(row):
        temp = 0
        for i in range(2):
            for a in range(0, (colum - 3)):
                for b in range(a + 1, (colum - 2)):
                    for c in range(b + 1, (colum - 1)):
                        for r2 in range(row):
                            if base[r1][a] == base[r2][a] and base[r1][b] == base[r2][b] and base[r1][c] == base[r2][c] and base[r2][colum - 1] == i:
                                C[r1][temp] += B[r2][temp % combination(3, colum - 1)]
                        # print(temp,":",temp//combination(3,colum-1))
                        temp += 1
    print("done C")
    return C
def FISA(base, C, list):
    colum = len(base[0])
    row = len(base)

    cols = combination(3, (colum - 1))
    C0 = [0] * cols
    C1 = [0] * cols

    t = 0
    for a in range(0, colum - 3):
        for b in range(a + 1, colum - 2):
            for c in range(b + 1, colum - 1):
                for r in range(row-1):
                    if base[r][a] == list[a] and base[r][b] == list[b] and base[r][c] == list[c] and base[r][colum-1] == 0:
                        C0[t] = C[r][t + 0 * cols]
                        # break
                    if base[r][a] == list[a] and base[r][b] == list[b] and base[r][c] == list[c] and base[r][colum-1] == 1:
                        C1[t] = C[r][t + 1 * cols]
                        # break
                t += 1
    #print(t)

    D0 = max(C0) + min(C0)
    D1 = max(C1) + min(C1)

    #print(D0, max(C0), min(C0))
    #print(D1, max(C1), min(C1))
    #print(D2, max(C2), min(C2))
    if D0 > 9*D1:
        return 0, D0/(D0+D1)
    else:
        return 1, D1/(D0+D1)
def Acc(A,B):
    result = 0
    for i in range(len(A)):
        if int(A[i]) - int(float(B[i])) == 0:
            result += 1

    return round(result*100/len(A), 2)


def Tprecision(Pre,Act):
    result = 0
    TP = 0
    FP = 0

    for i in range(len(Pre)):
        if int(Pre[i]) == 1 and int(float(Act[i])) == 1:
            TP +=1
        if int(Pre[i]) == 1 and int(float(Act[i])) == 0:
            FP +=1

    #return str(TP) +  " : " + str(TP+FP)

    if TP:
        return round(100*TP/(TP+FP),2)
    else:
        if FP:
            return 0
        else:
            return None


def Trecall(Pre,Act):
    result = 0
    TP = 0
    FN = 0

    for i in range(len(Pre)):
        if int(Pre[i]) == 1 and int(float(Act[i])) == 1:
            TP +=1
        if int(Pre[i]) == 0 and int(float(Act[i])) == 1:
            FN +=1

    #return str(TP) +  " : " + str(TP+FN)

    if TP:
        return round(100*TP/(TP+FN),2)
    else:
        if FN:
            return 0
        else:
            return None

listAcc = []
listPre = []
listRe = []
timeTest = []
timeUpdate = []
listRank = []
res = []

def testAccuracy(base,Te,C):
    test = Te
    X = np.zeros(len(test))
    ddd = np.zeros(len(test))
    X_test = np.array(test).T[-1]
    for i in range(len(test)):
        X[i], ddd[i] = FISA(base, C, test[i])
        listRank.append(ddd[i])
        #print(test[i])
    # for dd in ddd:
    #     print(dd)
    # listRank.append(ddd)

    #print(X)
    #print(X_test)
    res.append(X)
    listAcc.append(Acc(X,X_test))
    listPre.append(Tprecision(X,X_test))
    listRe.append(Trecall(X,X_test))
from sklearn.model_selection import train_test_split
df = pd.read_csv('../../data/FIS/Rule_List_Language.csv')
# Y = df.iloc[-1]
traindf, testdf = train_test_split(df,test_size=0.30, random_state=17)
# traindf = pd.read_csv('../../data/FIS/Rule_List_Language.csv')
# testdf = pd.read_csv('../../data/Kmeans/Result_kmeans.csv')
base = traindf.values.tolist()
test = testdf.values.tolist()
print(test)
print(len(base))
print(base)


import time
start = time.time()
A = caculateA(base)
M = caculateM(base)
B = caculateB(base,A,M)
C = caculateC(base,B)
totalTime = time.time() - start
print(totalTime)

listAcc = []
listPre = []
listRe = []
timeTest = []
timeUpdate = []
listRank = []
res = []
testAccuracy(base,test,C)
print("List acc: ",listAcc)
print("Res value: ",pd.DataFrame(res[0]).value_counts())
print("Count train: ",traindf.iloc[-1].value_counts())
print("Count test: ",testdf.iloc[-1].value_counts())
print("Count List rank: ",pd.DataFrame(listRank).value_counts())
print("List rank: ",len(listRank))