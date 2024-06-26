from HuffmanTree import *

if __name__ == "__main__":
    index_words_dict = {0: "are",1: "fuck",2: "is",3: "was",4: "you",5:"be"}
    index_freq_dict = {0: 99,1: 1000,2: 60,3: 40,4: 101,5:34}
    print(index_words_dict)
    print(index_freq_dict)

    huffmanTree = HuffmanTree()
    root = huffmanTree.buildTree(index_freq_dict)
    # root.printTreeDfs()

    huffmanCode_dict = root.generateHuffmanCode()
    print(huffmanCode_dict)

    word_huffmanCode_dict = huffmanTree.generate_word_huffmanCode_dict(index_words_dict, huffmanCode_dict)
    print(word_huffmanCode_dict)

    sentence = "you are fuck, fuck you!!!"
    encode_str = huffmanTree.encode(sentence, word_huffmanCode_dict)
    print(encode_str)

    decode_list = huffmanTree.decode(encode_str, huffmanCode_dict)
    print(decode_list)
    wordsList = huffmanTree.generateWordsbyIndexList(index_words_dict, decode_list)
    print(wordsList)
