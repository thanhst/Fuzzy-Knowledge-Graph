import pandas as pd
import numpy as np
import csv
import json
import fisa_module as fs
import os
import random
import time

class FKGS:
    print("FKGS is running")
    def __init__(self):
        self.listAcc = []
        self.listPre = []
        self.listRe = []
        self.timeTest = []
        self.timeUpdate = []
        self.listRank = []
        self.res = []
    
    def combination(self,k, n):
        if k == 0 or k == n:
            return 1
        if k == 1:
            return n
        return self.combination(k - 1, n - 1) + self.combination(k, n - 1)

    def diff(self,Rule1, Rule2):
        if Rule1[-1] != Rule2[-1]:
            return -1
        m = len(Rule1)
        count = 0
        for i in range(m-1):
            if Rule1[i] == Rule2[i]:
                count += 1
        return count/m
    
    def sampling(self,base, ran, e, k = 2):
        num = len(base)
        R = []
        list_index = []
        while (len(R)<num*ran/100):
            index = random.randrange(num)
            while (index in list_index):
                index = random.randrange(num)
            T = []
            T.append(index)
            for i in range(index-k,index+k+1):
                if i < num:
                    if i in list_index:
                        continue
                    else:
                        if self.diff(base[i],base[index]) < 1 - e:
                            T.append(i)
            for i in T:
                temp = 0
                for j in range(len(R)):
                    if self.diff(base[i],R[j]) < 1 - e:
                        continue
                    else:
                        temp = 1
                if temp:
                    T.remove(i)
            for i in T:
                R.append(base[i])
                list_index.append(i)

        return R


    # def calculateA(self,base,k):
    #     colum = len(base[0])
    #     row = len(base)
    #     A = np.zeros((row, self.combination(k, colum - 1)))

    #     for r1 in range(row):
    #         k = [0] * self.combination(k, colum - 1)
    #         temp = 0
    #         for a in range(0, colum - 4):
    #             for b in range(a + 1, colum - 3):
    #                 for c in range(b + 1, colum - 2):
    #                     for d in range(c + 1, colum - 1):
    #                         for r2 in range(row):
    #                             if base[r1][a] == base[r2][a] and base[r1][b] == base[r2][b] and base[r1][c] == base[r2][c] and base[r1][d] == base[r2][d]:
    #                                 k[temp] += 1

    #                         A[r1][temp] = k[temp] / row
    #                         temp += 1
    #     print("done A")
    #     return A
    
    def calculateA(self,base):
        colum = len(base[0])
        row = len(base)
        A = np.zeros((row, self.combination(4, colum - 1)))

        for r1 in range(row):
            k = [0] * self.combination(4, colum - 1)
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

    def calculateM(self,base):
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


    def calculateB(self,base, A, M):
        colum = len(base[0])
        row = len(base)
        B = np.zeros((row, self.combination(3, colum - 1)))

        for r in range(row):
            temp = 0
            for a in range(0, colum - 3):
                for b in range(a + 1, colum - 2):
                    for c in range(b + 1, colum - 1):
                        B[r][temp] = sum(A[r]) * min(M[r][a], M[r][b], M[r][c])
                        temp += 1
        print("done B")
        return B


    def calculateC(self,base, B):
        colum = len(base[0])
        row = len(base)
        cols = 6 * self.combination(3, colum - 1)
        C = np.zeros((row, cols))

        for r1 in range(row):
            temp = 0
            for i in range(1,7):
                for a in range(0, (colum - 3)):
                    for b in range(a + 1, (colum - 2)):
                        for c in range(b + 1, (colum - 1)):
                            for r2 in range(row):
                                if base[r1][a] == base[r2][a] and base[r1][b] == base[r2][b] and base[r1][c] == base[r2][c] and base[r2][colum - 1] == i:
                                    C[r1][temp] += B[r2][temp % self.combination(3, colum - 1)]
                            temp += 1
        print("done C")
        return C
    
    def FISA(self,base, C, list):
        colum = len(base[0])
        row = len(base)

        cols = self.combination(3, (colum - 1))
        C0 = [0] * cols
        C1 = [0] * cols
        C2 = [0] * cols
        C3 = [0] * cols
        C4 = [0] * cols
        C5 = [0] * cols

        t = 0
        for a in range(0, colum - 3):
            for b in range(a + 1, colum - 2):
                for c in range(b + 1, colum - 1):
                    for r in range(row-1):
                        if base[r][a] == list[a] and base[r][b] == list[b] and base[r][c] == list[c] and base[r][colum-1] == 1:
                            C0[t] = C[r][t + 0 * cols]
                        if base[r][a] == list[a] and base[r][b] == list[b] and base[r][c] == list[c] and base[r][colum-1] == 2:
                            C1[t] = C[r][t + 1 * cols]
                        if base[r][a] == list[a] and base[r][b] == list[b] and base[r][c] == list[c] and base[r][colum-1] == 3:
                            C2[t] = C[r][t + 2 * cols]
                        if base[r][a] == list[a] and base[r][b] == list[b] and base[r][c] == list[c] and base[r][colum-1] == 4:
                            C3[t] = C[r][t + 3 * cols]
                        if base[r][a] == list[a] and base[r][b] == list[b] and base[r][c] == list[c] and base[r][colum-1] == 5:
                            C4[t] = C[r][t + 4 * cols]
                        if base[r][a] == list[a] and base[r][b] == list[b] and base[r][c] == list[c] and base[r][colum-1] == 6:
                            C5[t] = C[r][t + 5 * cols]
                    t += 1

        D0 = max(C0) + min(C0)
        D1 = max(C1) + min(C1)
        D2 = max(C2) + min(C2)
        D3 = max(C3) + min(C3)
        D4 = max(C4) + min(C4)
        D5 = max(C5) + min(C5)
        
        DARR = [D0,D1,D2,D3,D4,D5]
        
        return DARR.index(max(DARR)) + 1, max(DARR)/sum(DARR)

        
    def Acc(self,A,B):
        result = 0
        for i in range(len(A)):
            if int(A[i]) == int(float(B[i])):
                result += 1
        return round(result*100/len(A), 2)


    def Tprecision(self, Pre, Act):
        TP = {}
        FP = {}
        unique_labels = set(Act) | set(Pre)
        for label in unique_labels:
            TP[label] = 0
            FP[label] = 0

        for i in range(len(Pre)):
            pred_label = int(Pre[i])
            true_label = int(Act[i])

            if pred_label == true_label:
                TP[pred_label] += 1
            else:
                FP[pred_label] += 1

        label_precision = {}
        for label in unique_labels:
            if TP[label] + FP[label] > 0:
                label_precision[label] = round(100 * TP[label] / (TP[label] + FP[label]), 2)
            else:
                label_precision[label] = 0
        return label_precision


    def Trecall(self, Pre, Act):
        TP = {}
        FN = {}
        unique_labels = set(Act) | set(Pre)

        for label in unique_labels:
            TP[label] = 0
            FN[label] = 0

        for i in range(len(Pre)):
            pred_label = int(Pre[i])
            true_label = int(Act[i])

            if pred_label == true_label:
                TP[true_label] += 1
            else:
                FN[true_label] += 1

        label_recall = {}
        for label in unique_labels:
            if TP[label] + FN[label] > 0:
                label_recall[label] = round(100 * TP[label] / (TP[label] + FN[label]), 2)
            else:
                label_recall[label] = 0

        return label_recall

    

    def testAccuracy(self,base,Te,C,n_classes):
        print("Bắt đầu test")
        test = Te
        X = np.zeros(len(test))
        ddd = np.zeros(len(test))
        X_test = np.array(test).T[-1]
        print("Bắt đầu tính toán FISA")
        for i in range(len(test)):
            try:
                X[i], ddd[i] = fs.FISA(base, C, test[i],n_classes)
                self.listRank.append(ddd[i])
            except RuntimeError as e:
                print("Exception: ",e)
        self.res.append(X)
        # print(X)
        # print(X_test)
        self.listAcc.append(self.Acc(X,X_test))
        self.listPre = list(self.Tprecision(X, X_test).values())
        self.listRe = list(self.Trecall(X, X_test).values())
        
    def FKGS(self,df,testdf,Turn = None,Modality = None,ran = None, e = None):
        basedf = df.values.tolist()
        traindf = [row[:] for row in basedf]
        test = testdf
        sampling_time = []
        train_time = []
        test_time = []
        accuracy = []
        train_labels = df.iloc[:, -1]
        test_labels = pd.DataFrame(testdf).iloc[:, -1]
        all_labels = pd.concat([train_labels, test_labels], axis=0)
        unique_labels = sorted(all_labels.unique())
        
        n_classes = len(unique_labels)
        
        for i in range(1,6):
            print(f"--------------------------------------Turn {i}---------------------------------")
            start = time.time()
            base = self.sampling(ran=ran, base=traindf, e=e)
            totalTime = time.time() - start
            sampling_time.append(totalTime)
            print(f'len base {i}:', len(base))
            
            print("\n---Start---")
            start = time.time()
            A = fs.calculateA(base)
            M = fs.calculateM(base)
            B = fs.calculateB(base,A,M)
            C = fs.calculateC(base,B,n_classes)
            C_normal = min_max_normalize(C)
            totalTime = time.time() - start
            train_time.append(totalTime)


            start = time.time()
            self.testAccuracy(base,test,C_normal,n_classes)
            totalTime = time.time() - start
            test_time.append(totalTime)
            print("---Finish---\n")
        print(":"*100)
        print('sampling_time:', sum(sampling_time) / len(sampling_time) if sampling_time else 0)
        print('train_time:', sum(train_time) / len(train_time) if train_time else 0)
        print('test_time:', sum(test_time) / len(test_time) if test_time else 0)
        print('accuracy:', sum(self.listAcc) / len(self.listAcc) if self.listAcc else 0)
        print('precision:',sum(self.listPre) / len(self.listPre) if self.listPre else 0)
        print('recall:',sum(self.listRe) / len(self.listRe) if self.listRe else 0)

        
    
    def scenario_random(self,df,Turn = None,Modality = None,ran = None):

        from sklearn.model_selection import train_test_split
        traindf, testdf = train_test_split(df,test_size=0.30, random_state=None)
        base = df.values.tolist()
        test = testdf.values.tolist()

        sampling_time = []
        train_time = []
        test_time = []
        accuracy = []
        for i in range(10):
            
            start = time.time()
            train = traindf.sample(n=int(len(traindf)*ran/100))
            base = train.values.tolist()
            totalTime = time.time() - start
            sampling_time.append(totalTime)
            print(f'len base {i}:', len(base))


            start = time.time()
            A = fs.caculateA(base)
            M = fs.caculateM(base)
            B = fs.caculateB(base,A,M)
            C = fs.caculateC(base,B)
            totalTime = time.time() - start
            train_time.append(totalTime)


            start = time.time()
            acc = self.testAccuracy(base,test,C)
            totalTime = time.time() - start
            accuracy.append(acc)
            test_time.append(totalTime)

        print('sampling_time:', sampling_time)
        print('train_time:', train_time)
        print('test_time:', test_time)
        print('accuracy:', accuracy)

    
def gaussian_normalize(C):
    C_mean = np.mean(C, axis=0)
    C_std = np.std(C, axis=0)
    C_normalized = (C - C_mean) / C_std
    return C_normalized
def min_max_normalize(C):
    C_min = np.min(C, axis=0)
    C_max = np.max(C, axis=0)
    C_normalized = (C - C_min) / (C_max - C_min)
    return C_normalized