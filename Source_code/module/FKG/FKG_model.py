import pandas as pd
import numpy as np
import csv
import json
from module.Module_CPP import fisa_module as fs
import os

class FKG:
    print("FKG LQT run")
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
        print(label_precision)
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

    

    def testAccuracy(self,base,Te,C):
        print("Bắt đầu test")
        test = Te
        X = np.zeros(len(test))
        ddd = np.zeros(len(test))
        X_test = np.array(test).T[-1]
        print("Bắt đầu tính toán FISA")
        for i in range(len(test)):
            try:
                X[i], ddd[i] =fs.FISA(base, C, test[i])
                self.listRank.append(ddd[i])
            except RuntimeError as e:
                print("Exception: ",e)
        self.res.append(X)
        print(X)
        print(X_test)
        self.listAcc.append(self.Acc(X,X_test))
        self.listPre = list(self.Tprecision(X, X_test).values())
        self.listRe = list(self.Trecall(X, X_test).values())
        
    def FKG(self,df,testdf,Turn = None,Modality = None):
        from sklearn.model_selection import train_test_split
        base = df.values.tolist()
        test = testdf
        labels_col = df.shape[1] - 1
        import time
        start = time.time()
        A = fs.calculateA(base)
        M = fs.calculateM(base)
        B = fs.calculateB(base,A,M)
        C = fs.calculateC(base,B)
        C_norm = min_max_normalize(C)
        totalTime = time.time() - start
        print("FKG train finish: ", totalTime)

        start = time.time()
        self.testAccuracy(base,test,C_norm)
        totalTimeTest = time.time() - start
        print("FKG test finish: ", totalTimeTest)
        results = {
            "Train Time": [totalTime],
            "Test Time" : [totalTimeTest],
            "Total Time" : [totalTime + totalTimeTest],
            "Test Accuracy": [self.listAcc],
            "Test Precision": [sum(self.listPre) / len(self.listPre) if self.listPre else 0],
            "Test Recall": [sum(self.listRe) / len(self.listRe) if self.listPre else 0],
            "Count List Rank": [pd.DataFrame( self.listRank).value_counts().to_dict()],
            "List Rank Length": [len( self.listRank)],
            "Label": self.res,
        }
        base_dir = os.getcwd()
        input_dir = os.path.join(base_dir,f"data/FKG/{Modality}/")
        
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
            
        AData = pd.DataFrame(A)
        AData.to_csv(os.path.join(input_dir,"A.csv"))
        BData = pd.DataFrame(B)
        BData.to_csv(os.path.join(input_dir,"B.csv"))
        CData = pd.DataFrame(C)
        CData.to_csv(os.path.join(input_dir,"C.csv"))
        MData = pd.DataFrame(M)
        MData.to_csv(os.path.join(input_dir,"M.csv"))
        
        dfData = pd.DataFrame(results)
        
        
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
        dfData.to_csv(os.path.join(input_dir,f"Results_FKG.csv"), index=False)
        
        
        csv_file = os.path.join(input_dir,f"Test/acc.csv")
        if(Turn!=None):
            with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)

                if file.tell() == 0:
                    writer.writerow(["id","Modality","Model","Accuracy", "F1 Score", "Recall"])

                writer.writerow([Turn,Modality,"FKG",f"{ self.listAcc[0]/100:.2%}",json.dumps([int(x) for x in self.listPre]),json.dumps([int(x) for x in self.listRe])])
        
        print("="*30)
        print("| {:<15} | {:>10} |".format("Name", "Value"))
        print("="*30)
        print("| {:<15} | {:>10.2f} s |".format("Train Time", totalTime))
        print("| {:<15} | {:>10.2f} s |".format("Test Time", totalTimeTest))
        print("| {:<15} | {:>10.2f} s |".format("Total Time", totalTime+totalTimeTest))
        print("="*30)
        print("| {:<15} | {:>10.2f} % |".format("Accuracy", self.listAcc[0]))
        print("| {:<15} | {:>10.2f} % |".format("Precision", sum(self.listPre) / len(self.listPre) if self.listPre else 0))
        print("| {:<15} | {:>10.2f} % |".format("Recall", sum(self.listRe) / len(self.listRe) if self.listRe else 0))
        print("="*30)
        
    def FKG_test(self,train,test,Turn = None,Modality = None):
        from sklearn.model_selection import train_test_split
        base = np.array(train)
        test = np.array(test)
        import time
        start = time.time()
        A = fs.calculateA(base)
        M = fs.calculateM(base)
        B = fs.calculateB(base,A,M)
        C = fs.calculateC(base,B)
        totalTime = time.time() - start
        print("FKG train finish: ", totalTime)

        start = time.time()
        self.testAccuracy(base,test,C)
        totalTimeTest = time.time() - start
        print("FKG test finish: ", totalTimeTest)
        results = {
            "Train Time": [totalTime],
            "Test Time" : [totalTimeTest],
            "Total Time" : [totalTime + totalTimeTest],
            "Test Accuracy": [self.listAcc],
            "Test Precision": [sum(self.listPre) / len(self.listPre) if self.listPre else 0],
            "Test Recall": [sum(self.listRe) / len(self.listRe) if self.listPre else 0],
            # "Count Train": [base.iloc[-1].value_counts().to_dict()],
            # "Count Test": [test.iloc[-1].value_counts().to_dict()],
            "Count List Rank": [pd.DataFrame( self.listRank).value_counts().to_dict()],
            "List Rank Length": [len( self.listRank)],
            "Label": self.res,
        }
        base_dir = os.getcwd()
        input_dir = os.path.join(base_dir,f"data/FKG/{Modality}/")
        
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
            
        AData = pd.DataFrame(A)
        AData.to_csv(os.path.join(input_dir,"A.csv"))
        BData = pd.DataFrame(B)
        BData.to_csv(os.path.join(input_dir,"B.csv"))
        CData = pd.DataFrame(C)
        CData.to_csv(os.path.join(input_dir,"C.csv"))
        MData = pd.DataFrame(M)
        MData.to_csv(os.path.join(input_dir,"M.csv"))
        
        dfData = pd.DataFrame(results)
        
        
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
        dfData.to_csv(os.path.join(input_dir,f"Results_FKG.csv"), index=False)
        
        
        csv_file = os.path.join(input_dir,f"Test/acc.csv")
        if(Turn!=None):
            with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)

                if file.tell() == 0:
                    writer.writerow(["id","Modality","Model","Accuracy", "F1 Score", "Recall"])

                writer.writerow([Turn,Modality,"FKG",f"{ self.listAcc[0]/100:.2%}",json.dumps([int(x) for x in self.listPre]),json.dumps([int(x) for x in self.listRe])])
        
        print("List acc: ", self.listAcc)
        

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