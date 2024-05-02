# Author: Zhenfei Lu
# Created Date: 4/27/2024
# Version: 1.0
# Email contact: luzhenfei_2017@163.com, zhenfeil@usc.edu

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random

class NLPClassifyModel(nn.Module):
    def __init__(self, vocabDict, vectorDim, sentenceLength, output_size):
        super().__init__()
        self.vocabDict = vocabDict
        self.embedding = nn.Embedding(len(vocabDict), vectorDim)  # embedding层
        self.avgpool = nn.AvgPool1d(sentenceLength)  # 池化层
        self.linear1 = nn.Linear(vectorDim, 15)
        self.activation = torch.sigmoid
        self.linear2 = nn.Linear(15, 15)
        # self.linear3 = nn.Linear(15, 15)
        self.output = nn.Linear(15, output_size)
        self.loss = nn.functional.cross_entropy
        self.softmax = torch.softmax

    def forward(self, x):
        x = self.embedding(x)
        x = self.avgpool(x.transpose(1, 2)).squeeze(2)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        # x = self.activation(self.linear3(x))
        x = self.output(x)
        x = self.softmax(x, 1)
        return x

    def calLoss(self, y_predict, y):
        return self.loss(y_predict, y)

    def data_iter(self, batch_size, features, labels):
        num_examples = len(features)
        indices = list(range(num_examples))
        random.shuffle(indices)
        for i in range(0, num_examples, batch_size):
            batch_indices = torch.tensor(
                indices[i: min(i + batch_size, num_examples)])
            yield features[batch_indices], labels[batch_indices]

    def fit(self, x, y, epoch, learning_rate, batch_size):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        metric = dict()
        metric["avg_loss"] = list()
        metric["accuracy"] = list()
        for i in range(1, epoch+1):
            # self.train()
            for X, Y in self.data_iter(batch_size, x, y):
                optimizer.zero_grad()
                loss = self.calLoss(self.forward(X), Y)
                loss.backward()
                optimizer.step()
            avg_loss = self.calLoss(self.forward(x), y)
            acc = self.calAccuracy(x, y)
            metric["avg_loss"].append(avg_loss)
            metric["accuracy"].append(acc)
            if i % 10 == 0 or i == epoch:
                print(f"Epoch: {i}, AvgLoss: {avg_loss}, Accuracy: {acc}")
        return metric

    def calAccuracy(self, x, y, showEachResult:bool=False, sentenceTypeDict:dict=None):
        def find_keys_by_values(dict, targetValues):
            str = ""
            for targetValue in targetValues:
                for k, v in dict.items():
                    if v == targetValue:
                        str = str + k
            return str
        acc = 0
        correct = 0
        N = y.shape[0]
        with torch.no_grad():  # not calculate auto-grad when doing fwd
            for i in range(0,N):
                y_predict = self.forward(x[i, :].unsqueeze(0))
                # y_predict = self.softmax(y_predict, 1)
                y_predict_index = int(y_predict.argmax())
                y_true_index = int(y[i, :].unsqueeze(0).argmax())
                if(y_predict_index == y_true_index):
                    correct = correct + 1
                    if showEachResult:
                        print(f"x: {find_keys_by_values(self.vocabDict, x[i, :])}, y: {y[i, :]}, y predict: {y_predict}, type: {sentenceTypeDict[y_predict_index]}, result: {'Correct'}")
                else:
                    if showEachResult:
                        print(f"x: {find_keys_by_values(self.vocabDict, x[i, :])}, y: {y[i, :]}, y predict: {y_predict}, type: {sentenceTypeDict[y_predict_index]}, result: {'Wrong'}")
        acc = correct / N
        return acc

    def plotMetric(self, metric, showImmediately):
        plt.figure()
        i = 1
        N = len(metric)
        for key, value in metric.items():
            plt.subplot(1, N, i)
            plt.plot(np.linspace(1,len(value),len(value)), value)
            plt.title(key)
            plt.xlabel('epoch')
            # plt.ylabel('value')
            i = i + 1
        if (showImmediately):
            plt.show()

    def saveModelWeights(self, localPath):
        torch.save(self.state_dict(), localPath)  # touch static method

    def loadModelWeights(self, localPath):
        self.load_state_dict(torch.load(localPath))  # load weights  .path, ***model structure must be same
