from os.path import join
from codecs import open


def load_data(split: str, make_vocab=True, path='./data_ner'):
    assert split in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []

    with open(join(path, split+'.char.bmes'), 'r', encoding='utf-8') as file:
        word_list = []
        tag_list = []
        for line in file:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    # 建立词表与id映射
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists


def build_map(lists: list) -> map:
    maps = {}
    for list in lists:
        for item in list:
            if item not in maps:
                maps[item] = len(maps)

    return maps
