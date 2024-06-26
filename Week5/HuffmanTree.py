import numpy as np

class HuffmanNode(object):
    def __init__(self, index, frequency):
        self.index = index
        self.frequency = frequency
        self.children = []

    def generateHuffmanCode(self, histCodeStr="", currentCodeStr=""):
        if (self is None):
            print("come 2 self is none part")
            return None
        HuffmanCodeDict = {}
        huffmanCode = histCodeStr + currentCodeStr
        if(len(self.children) == 0):
            HuffmanCodeDict[self.index] = huffmanCode
            return HuffmanCodeDict
        for i in range(0, 2):
            HuffmanCodeDict_temp = self.children[i].generateHuffmanCode(huffmanCode, str(i))
            HuffmanCodeDict.update(HuffmanCodeDict_temp)
        return HuffmanCodeDict

    def printTreeDfs(self):
        if (self is None):
            print("come 2 self is none part")
            return None
        print(self.index, self.frequency)
        for child in self.children:
            child.printTreeDfs()


class HuffmanTree(object):
    def __init__(self):
        pass

    def buildTree(self, dict_index_freq):
        huffmanNodeList = []
        for k,v in dict_index_freq.items():
            huffmanNodeList.append(HuffmanNode(k, v))
        index_current = len(dict_index_freq)
        root = None
        while(len(huffmanNodeList) > 1):
            sorted_list = sorted(huffmanNodeList, key=lambda x: x.frequency)
            left_node = sorted_list[0]
            right_node = sorted_list[1]
            huffmanNodeList.remove(sorted_list[0])
            huffmanNodeList.remove(sorted_list[1])
            merged_node = HuffmanNode(index_current, left_node.frequency + right_node.frequency)
            merged_node.children.append(left_node)
            merged_node.children.append(right_node)
            huffmanNodeList.append(merged_node)
            index_current = index_current + 1
            root = merged_node
        return root

    def generate_word_huffmanCode_dict(self, index_words_dict, huffmanCode_dict):
        word_huffmanCode_dict = {}
        for k, v in index_words_dict.items():
            word_huffmanCode_dict[v] = huffmanCode_dict[k]
        return word_huffmanCode_dict

    def encode(self, sentence, word_huffmanCode_dict):
        encode_str = ""
        N = len(sentence)
        left_index = 0
        right_index = left_index + 1
        while (left_index < N):
            while (right_index < N + 1):
                current_words = sentence[left_index:right_index]
                if current_words in word_huffmanCode_dict.keys():
                    encode_str = encode_str + word_huffmanCode_dict[current_words]
                    left_index = right_index
                    right_index = left_index + 1
                    continue
                else:
                    right_index = right_index + 1
            left_index = left_index + 1
            right_index = left_index + 1
        return encode_str

    def decode(self, huffmanStr, huffmanCode_dict):
        inverseHuffmanCode_dict = self.getInverseHuffmanCode_dict(huffmanCode_dict)  # dict = {"010": 5}
        decode_list = []
        N = len(huffmanStr)
        left_index = 0
        right_index = left_index + 1
        while(left_index < N):
            while(right_index < N + 1):
                current_code = huffmanStr[left_index:right_index]
                if current_code in inverseHuffmanCode_dict:
                    decode_list.append(inverseHuffmanCode_dict[current_code])
                    left_index = right_index
                    right_index = left_index + 1
                    continue
                else:
                    right_index = right_index + 1
            left_index = left_index + 1
            right_index = left_index + 1
        return decode_list


    def getInverseHuffmanCode_dict(self, huffmanCode_dict):
        InverseHuffmanCode_dict = {}
        for k, v in huffmanCode_dict.items():
            InverseHuffmanCode_dict[v] = k
        return InverseHuffmanCode_dict

    def generateWordsbyIndexList(self, index_words_dict, decode_list):
        wordsList = []
        for index in decode_list:
            wordsList.append(index_words_dict[index])
        return wordsList


