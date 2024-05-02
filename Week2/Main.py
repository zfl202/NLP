# Author: Zhenfei Lu
# Created Date: 4/27/2024
# Version: 1.0
# Email contact: luzhenfei_2017@163.com, zhenfeil@usc.edu

from ClassifyModel import *
from DataGenerater import *

class Solution(object):
    def __init__(self):
        self.test1()
        pass

    def test1(self):
        input_dim = 6
        output_size = 2  # 2 types
        fdg = FakeDataGenerater(input_dim)
        X, Y = fdg(1000)
        model = ClassifyModel(input_dim, output_size)
        print(model)
        history = model.fit(X, Y, 100, 0.001, 10)
        model_path = "myClassifyModelWeight.pth"
        model.saveModelWeights(model_path)

        loadedModel = ClassifyModel(input_dim, output_size)
        loadedModel.loadModelWeights(model_path)
        fdg2 = FakeDataGenerater(input_dim)
        X_valid, Y_valid = fdg2(10)  # generate validation data for test the model
        loadedModel.calAccuracy(X_valid, Y_valid, True)

        model.plotMetric(history, True)


if __name__ == "__main__":
    solution = Solution()
    # A = np.random.random((3,3))
    # B = np.random.random((4,1))
    # print(A)
    # print(B)
    # print(list(zip(A,B)))
    # print('here111')
    # A = np.random.random((3, 3))
    # A_Tensor = torch.FloatTensor(A)
    # print(A_Tensor)
    # print(A_Tensor.reshape(1,3,3))
    # print(A_Tensor.unsqueeze(0))
    # print(A_Tensor[0,:].unsqueeze(0))
    # print(A_Tensor[0,:].reshape(1,-1))
    # print(A.argmax())
    #
    # a = np.array([1.1])
    # print(a.reshape(1,1))
    # print(a.reshape(-1, 1))
    # print(a.reshape(1, -1))
    # a_tensor = torch.FloatTensor(a)
    # print(a_tensor)
    #
    # B = np.random.random((1, 3, 3))
    # print(B)
    # print(np.squeeze(B))
    # print(np.squeeze(B, 0))
    # print(np.squeeze(A))
