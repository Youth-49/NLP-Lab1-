# -*- coding:utf-8 -*-

import os
from trie import Trie
from numpy import loadtxt
import time

""" 
    Dictionary: dictionary based on Trie
"""

class Dictionary:
    def __init__(self):
        self.trie = Trie()
        self.pi = {}
        self.A = {}
        self.B = {}

    def load_dict(self, wordFile: str, seqFile: str):
        """load dictionary, wordFile should contain words with tags and seqFile should contain sentence markers

        Args:
            wordFile (str): path of wordFile
            seqFile (str): path of seqFile
        """
        load_st = time.time()
        assert os.path.exists(wordFile)
        assert os.path.exists(seqFile)

        num_item = 0
        sentence_list = loadtxt(seqFile, delimiter='\n')

        rowid = 0
        word_num = 0
        last_tag = None
        with open(wordFile, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                line = line.split('/')
                if len(line) == 2:
                    num_item = num_item + 1
                    # Remove [] from the wordFile
                    if '[' in line[0]:
                        line[0] = line[0][1:]
                    if ']' in line[1]:
                        line[1] = line[1].split(']')[0]
                    self.trie.insert(line[0], 1, line[1])
                    # Count emission times(calculate emission possibility)
                    thisTag = self.B.setdefault(line[1], {})
                    thisTag[line[0]] = thisTag.setdefault(line[0], 0) + 1
                else:
                    print(line)

                word_num = word_num + 1
                if word_num > sentence_list[rowid]:
                    word_num = 1
                    rowid = rowid + 1

                if word_num == 1:
                    # Count begin tag
                    self.pi[line[1]] = self.pi.setdefault(line[1], 0) + 1
                else:
                    # Count transition times(calculate transition matrix)
                    thisTag = self.A.setdefault(last_tag, {})
                    thisTag[line[1]] = thisTag.setdefault(line[1], 0) + 1

                last_tag = line[1]

        f.close()
        load_ed = time.time()
        print('--> Successfully load dictionary!')
        print(f'--> load {num_item} word pieces')
        print(f'--> Cost {load_ed-load_st}s\n')



if __name__ == '__main__':
    dic = Dictionary()
    # dic.trie.insert('lili')
    dic.load_dict('data.txt')
