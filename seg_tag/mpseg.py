# -*- coding:utf-8 -*-

from math import log
from dictionary import Dictionary
from dag import DAG
import time

""" 
    MPseg: maximum probability segment(based on probability)
"""


class MPseg(Dictionary):
    def __init__(self, wordFile: str = 'data/wordpieces.txt', seqFile: str = 'data/sequences.txt'):
        super().__init__()
        self.load_dict(wordFile, seqFile)

    def dp(self, sentence: str, dag: dict, dp: dict):
        """find path with maximum probability from end to start

        Args:
            sentence (str): sentence
            dag (dict): DAG built on given sentence
            dp (dict): dp variables
        """
        N = len(sentence)
        log_total_freq = log(self.trie.total_word_freq)
        dp[N] = (0, 0)
        for nd in range(N-1, -1, -1):
            dp[nd] = min((log_total_freq - log(self.trie.get_freq(sentence[nd:to]))
                         + dp[to][0], to) if self.trie.get_freq(sentence[nd:to]) > 0 else (0, to) for to in dag.get(nd))

    def dp_Add1(self, sentence: str, dag: dict, dp: dict):
        """find path with maximum probability(with Add-1 smoothing) from end to start

        Args:
            sentence (str): sentence
            dag (dict): DAG built on given sentence
            dp (dict): dp variables
        """
        print('Adapt Add 1 smoothing method')
        N = len(sentence)
        a = 1
        log_total_freq = log(self.trie.total_word_freq +
                             a*self.trie.total_word)

        dp[N] = (0, 0)
        for nd in range(N-1, -1, -1):
            dp[nd] = min((log_total_freq - log((self.trie.get_freq(sentence[nd:to])
                         or 1)+a) + dp[to][0], to) for to in dag.get(nd))

    def dp_JM(self, sentence: str, dag: dict, dp: dict):
        """find path with maximum probability(with Jelinek-Mercer method) from end to start

        Args:
            sentence (str): sentence
            dag (dict): DAG built on given sentence
            dp (dict): dp variables
        """
        print('Adapt Jelinek-Mercer smoothing method')
        N = len(sentence)
        a = 0.9

        dp[N] = (0, 0)
        for nd in range(N-1, -1, -1):
            dp[nd] = min((-log(a*(self.trie.get_freq(sentence[nd:to]) or 1) / self.trie.total_word_freq +
                         (1-a)/self.trie.total_word) + dp[to][0], to) for to in dag.get(nd))

    def mpcut(self, sentence: str = None, smooth: str = None):
        """tokenize the given sentence

        Args:
            sentence (str, optional): sentence to be tokenized. Defaults to None.
            smooth (bool, optional): if Ture, adapt smoothing method(Jelinek-Mercer). Defaults to False.

        Returns:
            list: tokens for given sentence
        """
        MPcut_st = time.time()
        if sentence is None or len(sentence) == 0:
            return []

        dag = DAG(sentence, self.trie)
        dp = {}
        if not smooth:
            self.dp(sentence, dag, dp)
        else:
            if smooth == 'Add1':
                self.dp_Add1(sentence, dag, dp)
            elif smooth == 'Jelinek-Mercer':
                self.dp_JM(sentence, dag, dp)

        N = len(sentence)
        st = 0
        ed = 0
        res = []
        while(st < N):
            ed = dp[st][1]
            res.append(sentence[st:ed])
            st = ed

        MPcut_ed = time.time()
        print(f'--> MP cutting costing {MPcut_ed-MPcut_st}s\n')
        return res


if __name__ == '__main__':
    SEG = MPseg()
    print(SEG.mpcut('武汉市长江大桥'))
