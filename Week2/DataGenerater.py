# Author: Zhenfei Lu
# Created Date: 4/27/2024
# Version: 1.0
# Email contact: luzhenfei_2017@163.com, zhenfeil@usc.edu

import numpy as np
import torch

class FakeDataGenerater(object):
    def __init__(self, oneSampledim):
        self.oneSampledim = oneSampledim

    def __call__(self, *args, **kwargs):
        if len(kwargs) != 0:
            return self.generateData(**kwargs)
        elif len(args) != 0:
            return self.generateData(*args)

    def generateData(self, num):
        X = np.random.random((num, self.oneSampledim))
        Y = np.zeros((num, 2))
        type1 = 0
        type2 = 0
        for i in range(X.shape[0]):
            if(np.sum(X[i, :]) >= self.oneSampledim/2):
                Y[i, :] = np.array([1, 0])  # one-hot code
                type1 = type1 + 1
            else:
                Y[i, :] = np.array([0, 1])  # one-hot code
                type2 = type2 + 1
        print(f"type1: {type1}, type2: {type2}, total: {type1+type2}")
        return (torch.FloatTensor(X), torch.FloatTensor(Y))