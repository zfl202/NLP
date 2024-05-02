# Author: Zhenfei Lu
# Created Date: 4/27/2024
# Version: 1.0
# Email contact: luzhenfei_2017@163.com, zhenfeil@usc.edu

from NLPClassifyModel import *
from NLPDataGenerater import *

class Solution(object):
    def __init__(self):
        self.test2()
        pass

    def test2(self):
        ndg = NLPDataGenerater()
        str_ = "abcdefghijklmnopqrstuvwxyz"
        vocabDict = ndg.buildVocabDict(str_)
        localVocabDictPath = "./vocabDict.json"
        ndg.saveVocabDict(vocabDict, "./vocabDict.json")
        sentenceLength = 7
        X, Y = ndg.generateTrainingData(1000, vocabDict, sentenceLength)
        vectorDim = 25
        output_size = 3
        nlp_model = NLPClassifyModel(vocabDict, vectorDim, sentenceLength, output_size)
        print(nlp_model)
        history = nlp_model.fit(X, Y, 100, 0.001, 10)
        model_path = "myNLPClassifyModelWeight.pth"
        nlp_model.saveModelWeights(model_path)

        loadedVocabDict = ndg.loadVocabDict(localVocabDictPath)
        X_valid, Y_valid = ndg.generateValidationData(vocabDict, sentenceLength)
        loadedModel = NLPClassifyModel(loadedVocabDict, vectorDim, sentenceLength, output_size)
        loadedModel.loadModelWeights(model_path)
        sentenceTypeDict = {0: "BadWords", 1: "GoodWords", 2: "NormalWords"}
        loadedModel.calAccuracy(X_valid, Y_valid, True, sentenceTypeDict)

        nlp_model.plotMetric(history, True)


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
