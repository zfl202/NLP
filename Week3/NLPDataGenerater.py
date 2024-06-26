# Author: Zhenfei Lu
# Created Date: 4/27/2024
# Version: 1.0
# Email contact: luzhenfei_2017@163.com, zhenfeil@usc.edu

import numpy as np
import torch
import json
import random

class NLPDataGenerater(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        if len(kwargs) != 0:
            return self.generateData(**kwargs)
        elif len(args) != 0:
            return self.generateData(*args)

    def buildVocabDict(self, str):
        # str = "abcdefghijklmnopqrstuvwxyz"
        vocabDict = dict()
        for index, char in enumerate(str):
            vocabDict[char] = index
        vocabDict['unk'] = len(vocabDict)
        return vocabDict

    def saveVocabDict(self, vocabDict, localFile):
        writer = open(localFile, "w", encoding="utf8")
        writer.write(json.dumps(vocabDict, ensure_ascii=False, indent=2))  # json serialize (dict 2 string)
        writer.close()

    def loadVocabDict(self, localFile):
        vocabDict = json.load(open(localFile, "r", encoding="utf8"))    # json deserialize (string 2 dict)
        return vocabDict

    def generateTrainingData(self, num, vocabDict, sentence_length):
        X = list()
        Y = list()
        type1, type2, type3 = 0, 0, 0
        for i in range(num):
            sentence = [random.choice(list(vocabDict.keys())) for _ in range(sentence_length)]  # random generate string
            if set("fuck") & set(sentence):
                y = [1, 0, 0]
                type1 = type1 + 1
            elif set("good") & set(sentence):
                y = [0, 1, 0]
                type2 = type2 + 1
            else:
                y = [0, 0, 1]
                type3 = type3 + 1
            x = [vocabDict.get(key, vocabDict['unk']) for key in sentence]  # string (key) to index (value)
            X.append(x)
            Y.append(y)
        print(f"type1: {type1}, type2: {type2}, type3: {type3}, total:{type1+type2+type3}")
        return (torch.LongTensor(X), torch.FloatTensor(Y))

    def generateValidationString(self, vocabDict, sentence_length):
        X = list()
        Y = list()
        sentences = ["fuckyouqqqqqqqqqqqqqqqqqqqq", "goodmanyyyyyyyy", "fukkyouqqq", "nxxxman", "jacyqqq", "gotltmmyyyyyyyyyyyy", "arenigd", "abyzzzx","f","xx"]
        for sentence in sentences:
            if set("fuck") & set(sentence):
                y = [1, 0, 0]
            elif set("good") & set(sentence):
                y = [0, 1, 0]
            else:
                y = [0, 0, 1]
            # x = [vocabDict.get(key, vocabDict['unk']) for key in sentence]  # string (key) to index (value)
            X.append(sentence)
            Y.append(y)
        return (X, Y)
        # return (torch.LongTensor(X), torch.FloatTensor(Y))