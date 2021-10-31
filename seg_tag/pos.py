# -*- coding:utf-8 -*-

from math import log
from re import L
from dictionary import Dictionary
import time

""" 
    WordTagging: tag part-of-speech for sentence(based on HMM)
"""


class WordTagging(Dictionary):
    def __init__(self, wordFile: str = 'data/wordpieces.txt', seqFile: str = 'data/sequences.txt'):
        super().__init__()
        self.load_dict(wordFile, seqFile)

    def tagging(self, seq: list):
        """tag part-of-speech for given sentence

        Args:
            seq (list): sentence to be tagged

        Returns:
            list: pos for given sentence
        """
        tagging_st = time.time()
        N_word = len(seq)
        sum_pi = 0
        sum_A = {}
        sum_B = {}
        tag_list = []

        # Calculate total values for probability normalization
        sum_pi = sum(self.pi.values())
        for key in self.A.keys():
            sum_A[key] = sum(self.A[key].values())

        for key in self.B.keys():
            sum_B[key] = sum(self.B[key].values())

        # Viterbi algorithm
        delta = []
        tmp_map = {}
        # assign a little probability to unseen event
        eps = 1e-8
        for tag in self.B.keys():
            if tag not in self.pi.keys():
                trans_p = log(eps)
            else:
                trans_p = log(self.pi[tag]) - log(sum_pi)
            if seq[0] not in self.B[tag]:
                emit_p = log(eps)
            else:
                emit_p = log(self.B[tag][seq[0]]) - log(sum_B[tag])

            tmp_map[tag] = (-trans_p - emit_p, tag)

        delta.append(tmp_map)
        tmp_map = {}
        for id in range(1, N_word):
            for nextTag in self.B.keys():
                for preTag in delta[id-1].keys():
                    if (preTag not in self.A.keys()) or (nextTag not in self.A[preTag].keys()):
                        trans_p = log(eps)
                    else:
                        trans_p = log(self.A[preTag][nextTag]
                                      ) - log(sum_A[preTag])
                    if seq[id] not in self.B[nextTag].keys():
                        emit_p = log(eps)
                    else:
                        emit_p = log(self.B[nextTag][seq[id]]
                                     ) - log(sum_B[nextTag])

                    if (nextTag in tmp_map.keys()):
                        tmp_map[nextTag] = min(
                            tmp_map[nextTag], (delta[id-1][preTag][0] - trans_p - emit_p, preTag))
                    else:
                        tmp_map[nextTag] = (
                            delta[id-1][preTag][0] - trans_p - emit_p, preTag)
            delta.append(tmp_map)
            tmp_map = {}

        # backtracking
        now_tag = max(zip(delta[N_word-1].values(), delta[N_word-1].keys()))
        tag_list.append(now_tag[1])
        for i in range(N_word-2, -1, -1):
            now_tag = delta[i+1][now_tag[1]]
            tag_list.append(now_tag[1])

        tag_list.reverse()
        tagging_ed = time.time()
        print(f'--> Tagging costing {tagging_ed-tagging_st}s\n')
        return tag_list


if __name__ == '__main__':
    wordTag = WordTagging()
    print(wordTag.tagging(['我', '是', '谁']))
