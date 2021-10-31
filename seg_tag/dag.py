#-*- coding:utf-8 -*-

class DAG:
    def __init__(self, sentence, Trie):
        # self.dag = {Nd1:[nextNd1, nextNd2, ...], ...}
        self.dag = {}
        self.dict = Trie
        self.build(sentence)

    def build(self, sentence: str):
        """build DAG with given sentence

        Args:
            sentence (str): sentence
        """
        N = len(sentence)
        for st in range(N):
            ed_list = []
            ed = st+1
            frag = sentence[st]
            while ed <= N:
                if self.dict.search(frag) and self.dict.get_freq(frag):
                    ed_list.append(ed)
                ed = ed + 1
                if (ed > N+1):
                    break
                frag = sentence[st:ed]
            if not ed_list:
                ed_list.append(st+1)
            self.dag[st] = ed_list

    def items(self):
        return self.dag.items()

    def get(self, key: str, default=None):
        """get suffix of the given node(key)

        Args:
            key (str): node
            default (any, optional): default values when given node has no suffix node. Defaults to None.

        Returns:
            list: suffix of the given node
        """
        return self.dag.get(key, default)