import pandas as pd
import numpy as np
from module.Module_CPP import fisa_module as fs
import csv
import os
import json

class FKG:
    
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
        cols = 2 * self.combination(3, colum - 1)
        C = np.zeros((row, cols))

        for r1 in range(row):
            temp = 0
            for i in range(2):
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

        t = 0
        for a in range(0, colum - 3):
            for b in range(a + 1, colum - 2):
                for c in range(b + 1, colum - 1):
                    for r in range(row-1):
                        if base[r][a] == list[a] and base[r][b] == list[b] and base[r][c] == list[c] and base[r][colum-1] == 0:
                            C0[t] = C[r][t + 0 * cols]
                        if base[r][a] == list[a] and base[r][b] == list[b] and base[r][c] == list[c] and base[r][colum-1] == 1:
                            C1[t] = C[r][t + 1 * cols]
                    t += 1

        D0 = max(C0) + min(C0)
        D1 = max(C1) + min(C1)

        if D0 > 9*D1:
            return 0, D0/(D0+D1)
        else:
            return 1, D1/(D0+D1)
        
    def Acc(self,A,B):
        result = 0
        for i in range(len(A)):
            if int(A[i]) - int(float(B[i])) == 0:
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
            # print("Bắt đầu lần chạy thứ ", i)
            try:
                X[i], ddd[i] = self.computeFISA(base, C, test[i],6)
                self.listRank.append(ddd[i])
            except RuntimeError as e:
                print("Exception: ",e)
        self.res.append(X)
        self.listAcc.append(self.Acc(X,X_test))
        self.listPre = list(self.Tprecision(X, X_test).values())
        self.listRe = list(self.Trecall(X, X_test).values())
        
    def FKG(self,df,Turn = None,Modality = None):
        from sklearn.model_selection import train_test_split
        traindf, testdf = train_test_split(df,test_size=0.30, random_state=17)
        base = traindf.values.tolist()
        test = testdf.values.tolist()
        labels_col = df.shape[1] - 1
        import time
        start = time.time()
        A = self.calculateA(base)
        M = self.calculateM(base,labels_col)
        B = self.calculateB(base,M)
        C = self.calculateC(base,B,labels_col)
        totalTime = time.time() - start
        print("FKG train finish: ", totalTime)

        start = time.time()
        self.testAccuracy(base,test,C)
        totalTimeTest = time.time() - start
        print("FKG test finish: ", totalTimeTest)
        results = {
            "Total Time": [totalTime],
            "Test Accuracy": [self.listAcc],
            "Test Precision": [self.listPre],
            "Test Recall": [self.listRe],
            "Count Train": [traindf.iloc[-1].value_counts().to_dict()],
            "Count Test": [testdf.iloc[-1].value_counts().to_dict()],
            "Count List Rank": [pd.DataFrame( self.listRank).value_counts().to_dict()],
            "List Rank Length": [len( self.listRank)]
        }
        AData = pd.DataFrame(A)
        AData.to_csv("./data/FKG/A.csv")
        BData = pd.DataFrame(B)
        BData.to_csv("./data/FKG/B.csv")
        CData = pd.DataFrame(C)
        CData.to_csv("./data/FKG/C.csv")
        MData = pd.DataFrame(M)
        MData.to_csv("./data/FKG/M.csv")
        
        dfData = pd.DataFrame(results)
        dfData.to_csv("./data/FKG/Results_FKG.csv", index=False)
        
        
        csv_file = "./data/Test/acc.csv"
        if(Turn!=None):
            with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)

                if file.tell() == 0:
                    writer.writerow(["id","Modality","Model","Accuracy", "F1 Score", "Recall"])

                writer.writerow([Turn,Modality,"FKG",f"{ self.listAcc[0]/100:.2%}",json.dumps([int(x) for x in self.listPre]),json.dumps([int(x) for x in self.listRe])])
        
        print("List acc: ", self.listAcc)
        # print("List pre: ", self.listPre)
        # print("List re: ", self.listRe)
        # print("Res value: ",pd.DataFrame ( self.res[0]).value_counts())
        # print("Count train: ",traindf.iloc[-1].value_counts())
        # print("Count test: ",testdf.iloc[-1].value_counts())
        # print("Count List rank: ",pd.DataFrame( self.listRank).value_counts())
        # print("List rank: ",len( self.listRank))
        
    def FKG_test(self,train,test,Turn = None,Modality = None):
        from sklearn.model_selection import train_test_split
        base = np.array(train)
        test = np.array(test)
        import time
        start = time.time()
        A = self.calculateA(base)
        M = self.calculateM(base)
        B = self.calculateB(base,A,M)
        C = self.calculateC(base,B)
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
            "Count Train": [base.iloc[-1].value_counts().to_dict()],
            "Count Test": [test.iloc[-1].value_counts().to_dict()],
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
