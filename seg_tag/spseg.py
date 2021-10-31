#-*- coding:utf-8 -*-

from dictionary import Dictionary
from dag import DAG
import time

""" 
    SPseg: shortest path segment(based on dictionary)
"""

class SPseg(Dictionary):
    def __init__(self, wordFile:str = 'data/wordpieces.txt', seqFile: str = 'data/sequences.txt'):
        super().__init__()
        self.load_dict(wordFile, seqFile)
    
    def dp(self, sentence: str, dag: dict, dp: dict):
        """find path with shortest path from end to start

        Args:
            sentence (str): sentence
            dag (dict): DAG built on the given sentence
            dp (dict): dp variables
        """
        N = len(sentence)
        dp[N] = (0,0)
        for nd in range(N-1,-1,-1):
            dp[nd] = min((1 + dp[to][0], to) for to in dag.get(nd))


    def spcut(self, sentence: str = None):
        """tokenize the given sentence

        Args:
            sentence (str, optional): sentence to be tokenized. Defaults to None.

        Returns:
            list: tokens for given sentence
        """
        SPcut_st = time.time()
        if sentence is None or len(sentence) == 0:
            return []

        dag = DAG(sentence, self.trie)
        dp = {}
        self.dp(sentence, dag, dp)
        N = len(sentence)
        st = 0
        ed = 0
        res = []
        while(st < N): 
            ed = dp[st][1]
            res.append(sentence[st:ed])
            st = ed

        SPcut_ed = time.time()
        print(f'--> SP cut costing {SPcut_ed-SPcut_st}s\n')
        return res

if __name__ == '__main__':
    SEG = SPseg()
    print(SEG.spcut('武汉市长江大桥'))