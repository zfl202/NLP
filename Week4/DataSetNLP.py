import torch
from torch.utils.data import Dataset
import jieba

class DataSetNLP(Dataset):
    def __init__(self, sentenceMaxLen):
        super().__init__()
        self.max_length = sentenceMaxLen
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def buildVocabDict(self, vocab_path):
        vocabDict = dict()
        with open(vocab_path, "r", encoding="utf8") as f:
            for index, line in enumerate(f):
                char = line.strip()
                vocabDict[char] = index + 1   # 空出0号位的字，不使用
        vocabDict['unk'] = len(vocabDict)
        return vocabDict

    def generateTrainingData(self, vocabDict, corpus_path, maxTrainingDataNum):
        with open(corpus_path, encoding="utf8") as f:
            for line in f:
                indexArr = self.sentence_to_indexArr(line, vocabDict)
                label = self.sequence_to_indexLabel(line)
                indexArr, label = self.padding_indexArr_Label(indexArr, label)  # indexArr  List每行长度必须相同才能变Tensor
                indexArr = torch.LongTensor(indexArr)
                label = torch.LongTensor(label)
                # print(label)
                self.data.append([indexArr, label])
                if len(self.data) > maxTrainingDataNum:
                    break
        # print(len(self.data))
        return self

    def sentence_to_indexArr(self, sentence, vocabDict):
        return [vocabDict.get(sentence[i], vocabDict["unk"]) for i in range(0, len(sentence))]

    def sequence_to_indexLabel(self, sentence):
        words = jieba.lcut(sentence)
        label = [0] * len(sentence)
        pointer = 0
        for word in words:
            pointer += len(word)
            label[pointer - 1] = 1
        return label

    def padding_indexArr_Label(self, indexArr, label):
        indexArr = indexArr[:self.max_length]
        indexArr += [0] * (self.max_length - len(indexArr))
        label = label[:self.max_length]
        label += [-100] * (self.max_length - len(label))
        return indexArr, label

    def generateValidationData(self, vocabDict):
        sentences = ["同时国内有望出台新汽车刺激方案",
                     "沪胶后市有望延续强势",
                     "经过两个交易日的强势调整后",
                     "昨日上海天然橡胶期货价格再度大幅上扬",
                     "上海和北京房价都很高",
                     "艾尔登法环DLC6月21日发售",
                     "原神要凉了",
                     "中国的学术研究环境不如美国",
                     "中国经济完蛋了",
                     "台湾科技比大陆发达的很多"]
        X = []
        Y = []
        for line in sentences:
            line = line.strip()
            indexArr = self.sentence_to_indexArr(line, vocabDict)
            label = self.sequence_to_indexLabel(line)
            indexArr, label = self.padding_indexArr_Label(indexArr, label)  # indexArr  List每行长度必须相同才能变Tensor
            X.append(indexArr)
            Y.append(label)
        return torch.LongTensor(X), torch.LongTensor(Y), sentences
