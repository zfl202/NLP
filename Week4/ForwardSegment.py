import numpy as np

class ForwardSegment(object):
    def __init__(self):
        pass

    def loadWordsDict(self, filePath):
        words_dict = {}
        with open(filePath, encoding="utf8") as f:
            for line in f:
                words = line.split()[0]
                words_dict[words] = 1
        return words_dict

    def cutByMaxSize(self, string, words_dict):
        N = len(string)
        words_list = []
        left_pointer = 0
        right_pointer = N
        while (left_pointer < N):
            while (left_pointer < right_pointer):
                words = string[left_pointer:right_pointer]
                if words in words_dict:
                    words_list.append(words)
                    break
                right_pointer = right_pointer - 1
            if (left_pointer == right_pointer):
                words_list.append(string[left_pointer:left_pointer + 1])
                left_pointer = left_pointer + 1
            else:
                left_pointer = right_pointer
            right_pointer = N
        return words_list

    def load_prefix_word_dict(self, filePath):
        prefix_dict = {}
        with open(filePath, encoding="utf8") as f:
            for line in f:
                word = line.split()[0]
                for i in range(1, len(word)):
                    if word[:i] not in prefix_dict:  # 不能用前缀覆盖词
                        prefix_dict[word[:i]] = 0  # 前缀
                prefix_dict[word] = 1  # 词
        return prefix_dict

    def cutByPrefix(self, string, prefix_dict):
        N = len(string)
        words_list = []
        left_pointer = 0
        right_pointer = left_pointer + 1
        right_pointer_temp = -1
        while (left_pointer < N):
            while (right_pointer < N + 1):
                words = string[left_pointer:right_pointer]
                if words in prefix_dict:
                    if prefix_dict[words] == 0:
                        right_pointer = right_pointer + 1
                    elif prefix_dict[words] == 1:
                        right_pointer_temp = right_pointer
                        right_pointer = right_pointer + 1
                else:
                    break
            if (right_pointer_temp == -1):
                words_list.append(string[left_pointer:left_pointer + 1])
                left_pointer = left_pointer + 1
                right_pointer = left_pointer + 1
                right_pointer_temp = -1
            else:
                words_list.append(string[left_pointer:right_pointer_temp])
                left_pointer = right_pointer_temp
                right_pointer = left_pointer + 1
                right_pointer_temp = -1
        return words_list