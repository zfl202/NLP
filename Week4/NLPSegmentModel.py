# Author: Zhenfei Lu
# Created Date: 4/27/2024
# Version: 1.0
# Email contact: luzhenfei_2017@163.com, zhenfeil@usc.edu

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader

class NLPSegmentModel(nn.Module):
    def __init__(self, vocabDict, vectorDim, rnn_hidden_size, num_rnn_layers):
        super().__init__()
        self.vocabDict = vocabDict
        self.embedding = nn.Embedding(len(vocabDict) + 1, vectorDim)  # embedding layer  空出0号位的字，不使用
        # self.rnn_layer = nn.RNN(input_size=vectorDim,
        #                         hidden_size=rnn_hidden_size,
        #                         batch_first=True,
        #                         num_layers=num_rnn_layers,
        #                         )
        self.rnn_layer = nn.LSTM(input_size=vectorDim,
                                hidden_size=rnn_hidden_size,
                                batch_first=True,
                                num_layers=num_rnn_layers,
                                )  # LSTM
        self.linear1 = nn.Linear(rnn_hidden_size, 2)
        # self.softmax = torch.softmax
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)  # not one-hot code, value=1 index code

    def forward(self, x):
        x = self.embedding(x)   # input shape: (batch_size, sen_len), output shape:(batch_size, sen_len, input_dim)
        x, h = self.rnn_layer(x)  # output shape:(batch_size, sen_len, hidden_size)
        x = self.linear1(x)  # output shape:(batch_size, sen_len, 2)
        # softmax 对分割词效果很差
        # x = self.softmax(x, 2)  # do softmax on dim=2  (dim0, dim1, dim2) = (axis0, axis1, axis2)
        return x

    def calLoss(self, y_predict, y):  # 问题 cross_entropy onehot和label下标index编码 [0,0,1]和2
        return self.loss(y_predict.view(-1, 2), y.view(-1)) # (batch_size, sen_len, 2) ->  (batch_size * sen_len, 2)

    # def data_iter(self, batch_size, features, labels):
    #     num_examples = len(features)
    #     indices = list(range(num_examples))
    #     random.shuffle(indices)
    #     for i in range(0, num_examples, batch_size):
    #         batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
    #         yield features[batch_indices], labels[batch_indices]

    def fit(self, dataset, epoch, learning_rate, batch_size):
        x = torch.LongTensor(np.array([dataset.data[i][0].numpy() for i in range(len(dataset.data))]))
        # print(x.shape)
        y = torch.LongTensor(np.array([dataset.data[i][1].numpy() for i in range(len(dataset.data))]))
        # print(y.shape)
        dataLoader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        metric = dict()
        metric["avg_loss"] = list()
        metric["accuracy"] = list()
        for i in range(1, epoch+1):
            self.train()
            for X, Y in dataLoader:
                optimizer.zero_grad()
                loss = self.calLoss(self.forward(X), Y)
                loss.backward()
                optimizer.step()
            avg_loss = self.calLoss(self.forward(x), y)
            acc = self.calAccuracy(x, y)
            metric["avg_loss"].append(avg_loss.item())
            metric["accuracy"].append(acc)
            if i % 1 == 0 or i == epoch:
                print(f"Epoch: {i}, AvgLoss: {avg_loss}, Accuracy: {acc}")
        return metric

    def calAccuracy(self, x, y, showEachResult:bool=False, sentences:list=None):
        acc = 0
        correct = 0
        N = y.shape[0]
        self.eval()
        with torch.no_grad():  # not calculate auto-grad when doing fwd
            for i in range(0,N):
                y_predict = self.forward(x[i].unsqueeze(0))  # shape (1,sentenceLen,2)
                # print(y_predict.shape)
                y_predict_index = (y_predict.argmax(dim=2)).squeeze(0)
                # print(y_predict_index.shape)
                # print(y[i].shape)
                y_true_index = y[i]
                # print(y_predict_index, y_true_index)

                indices = torch.where(y_true_index == -100)[0]  # find the first -100
                if indices.numel() > 0:
                    first_index = indices[0].item()
                    y_predict_index = y_predict_index[0:(first_index)]
                    y_true_index = y_true_index[0:(first_index)]

                if(torch.equal(y_predict_index, y_true_index)):
                    correct = correct + 1
                    if showEachResult:
                        sentence = sentences[i]
                        print(f"y truth: {y_true_index}, y predict: {y_predict_index}, result: {'Correct'}")
                        for j in range(0, y_predict_index.shape[0]):
                            if(y_predict_index[j] == 1):
                                print(sentence[j], end=" / ")
                            elif(y_predict_index[j] == 0):
                                print(sentence[j], end="")
                        print()
                else:
                    if showEachResult:
                        sentence = sentences[i]
                        print(f"y truth: {y_true_index}, y predict: {y_predict_index}, result: {'Wrong'}")
                        for j in range(0, y_predict_index.shape[0]):
                            if (y_predict_index[j] == 1):
                                print(sentence[j], end=" / ")
                            elif (y_predict_index[j] == 0):
                                print(sentence[j], end="")
                        print()
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
