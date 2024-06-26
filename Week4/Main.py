from ForwardSegment import *
from NLPSegmentModel import *
from DataSetNLP import *
import time

class Solution(object):
    def __init__(self):
        self.runAllTests()

    def test01(self):
        fs = ForwardSegment()
        input_path = "corpus.txt"
        output_path = "cut_method1_output_my.txt"
        word_dict = fs.loadWordsDict("./dict.txt")
        writer = open(output_path, "w", encoding="utf8")
        start_time = time.time()
        with open(input_path, encoding="utf8") as f:
            for line in f:
                words = fs.cutByMaxSize(line.strip(), word_dict)
                writer.write(" / ".join(words) + "\n")
        writer.close()
        print("耗时：", time.time() - start_time)
        return

    def test02(self):
        fs = ForwardSegment()
        input_path = "./corpus.txt"
        output_path = "./cut_method2_output_my.txt"
        word_dict = fs.load_prefix_word_dict("./dict.txt")
        writer = open(output_path, "w", encoding="utf8")
        start_time = time.time()
        with open(input_path, encoding="utf8") as f:
            for line in f:
                words = fs.cutByPrefix(line.strip(), word_dict)
                writer.write(" / ".join(words) + "\n")
        writer.close()
        print("耗时：", time.time() - start_time)
        return

    def test03(self):
        sentenceMaxLen = 20
        ds = DataSetNLP(sentenceMaxLen)
        localVocabDictPath = "./chars.txt"
        vocabDict = ds.buildVocabDict(localVocabDictPath)
        corpus_path = "./corpus.txt"
        ds = ds.generateTrainingData(vocabDict, corpus_path, maxTrainingDataNum=10000)
        vectorDim = 50
        rnn_hidden_size = 100
        num_rnn_layers = 3
        nlpSeg_model = NLPSegmentModel(vocabDict, vectorDim, rnn_hidden_size, num_rnn_layers)
        print(nlpSeg_model)
        history = nlpSeg_model.fit(dataset=ds, epoch=30, learning_rate=0.001, batch_size=20)
        model_path = "myNLPSegmentModelWeight.pth"
        nlpSeg_model.saveModelWeights(model_path)

        X_valid, Y_valid, sentences = ds.generateValidationData(vocabDict)
        loadedModel = NLPSegmentModel(vocabDict, vectorDim, rnn_hidden_size, num_rnn_layers)
        loadedModel.loadModelWeights(model_path)
        loadedModel.calAccuracy(X_valid, Y_valid, showEachResult=True, sentences=sentences)

        nlpSeg_model.plotMetric(history, True)

    def runAllTests(self):
        # self.test01()
        # self.test02()
        self.test03()


if __name__ == "__main__":
    solution = Solution()