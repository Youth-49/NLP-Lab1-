# -*- coding:utf-8 -*-

class Trie:
    def __init__(self):
        self.root = {}
        self.max_word_len = 0
        self.total_word_freq = 0
        self.total_word = 0
        self.end_token = '[END]'
        self.freq_token = '[FREQ]'
        self.tag_token = '[TAG]'

    def insert(self, word: str, freq: int = 1, tag: str = None):
        """insert a word into dictionary

        Args:
            word (str): word to be inserted
            freq (int, optional): the number of occurences of word. Defaults to 1.
            tag (str, optional): the part-of-speech of word. Defaults to None.
        """
        node = self.root
        for char in word:
            node = node.setdefault(char, {})
        self.max_word_len = max(self.max_word_len, len(word))
        if node == {}:
            self.total_word = self.total_word + 1
        node[self.end_token] = self.end_token
        node[self.freq_token] = freq
        node[self.tag_token] = tag
        self.total_word_freq += freq

    def search(self, word: str):
        """search the given word in the dictionary

        Args:
            word (str): word to be searched

        Returns:
            dict or None: if word exists, return the dict={'[FREQ]': , '[TAG]': , '[END]':END}, otherwise return None
        """
        node = self.root
        for char in word:
            if char not in node:
                return None
            node = node[char]
        return node if self.end_token in node else None

    def start_with(self, prefix: str):
        """find prefix in the dictionary

        Args:
            prefix (str): a prefix of a word

        Returns:
            dict: subtree after traverse the prefix
        """
        node = self.root
        for char in prefix:
            if char not in node:
                return None
            node = node[char]
        return node

    def get_freq(self, word: str):
        """get the number of occurences(freq) of the given word

        Args:
            word (str): word

        Returns:
            int: the freq of the given word
        """
        node = self.search(word)
        if node:
            return node.get(self.freq_token, 1)
        else:
            return 0


if __name__ == '__main__':
    t = Trie()
    t.insert('liu')
    t.insert('lily')
    print(t.root)
    a = t.search('liu')
    print(a)
    b = t.start_with('li')
    print(b)
