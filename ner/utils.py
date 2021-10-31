import pickle
import torch


def save_model(model, path):
    with open(path, 'wb') as file:
        pickle.dump(model, file)


def load_model(path):
    with open(path, 'rb') as file:
        model = pickle.laod(file)
    return model


def extend_maps(word2id, tag2id):
    # for lstm
    # <pad> fill the blank, <unk> represents unknown word
    word2id['<unk>'] = len(word2id)
    tag2id['<unk>'] = len(tag2id)
    word2id['<pad>'] = len(word2id)
    tag2id['<pad>'] = len(tag2id)
    # for crf
    word2id['<start>'] = len(word2id)
    tag2id['<start>'] = len(tag2id)
    word2id['<end>'] = len(word2id)
    tag2id['<end>'] = len(tag2id)

    return word2id, tag2id


def preprocess_for_lstmcrf(word_lists, tag_lists, test=False):
    assert len(word_lists) == len(tag_lists)
    L = len(word_lists)
    for i in range(L):
        word_lists[i].append('<end>')
        if not test:
            tag_lists[i].append('<end>')

    return word_lists, tag_lists


def flatten_lists(lists):
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list


def tensorize(batch, maps):
    # 将输入向量化
    PAD = maps.get('<pad>')
    UNK = maps.get('<unk>')
    max_len = len(batch[0])
    batch_size = len(batch)
    tensoried_batch = torch.ones(batch_size, max_len).long() * PAD
    for i, sent in enumerate(batch):
        for j, word in enumerate(sent):
            tensoried_batch[i][j] = maps.get(word, UNK)

    lengths = [len(sent) for sent in batch]
    return tensoried_batch, lengths


def sort_by_lengths(word_lists, tag_lists):
    # 按batch中sentence长度排序
    pairs = list(zip(word_lists, tag_lists))
    indices = sorted(range(len(pairs)),
                     key=lambda k: len(pairs[k][0]),
                     reverse=True)
    pairs = [pairs[i] for i in indices]

    word_lists, tag_lists = list(zip(*pairs))

    return word_lists, tag_lists, indices


def loss_function(crf_scores, tag_tensor, tag2id):
    """
    计算BiLSTM-CRF模型的loss function
    reference: https://arxiv.org/pdf/1603.01360.pdf
    """
    pad_id = tag2id.get('<pad>')
    start_id = tag2id.get('<start>')
    end_id = tag2id.get('<end>')

    device = crf_scores.device

    # tag_tensor:[B, L] crf_scores:[B, L, T, T]
    batch_size, max_len = tag_tensor.size()
    target_size = len(tag2id)

    # mask = 1 - ((tag_tensor == pad_id) + (tag_tensor == end_id))  # [B, L]
    # mask[i][j] = 1: j-th tag in i-th sentence is not <pad>
    mask = (tag_tensor != pad_id)
    # length: [B, L(i)]
    lengths = mask.sum(dim=1)
    tag_tensor = indexed(tag_tensor, target_size, start_id)
    # # 计算Golden scores
    tag_tensor = tag_tensor.masked_select(mask)  # [real_L] 1-D tensor
    flatten_scores = crf_scores.masked_select(
        mask.view(batch_size, max_len, 1, 1).expand_as(crf_scores)
    ).view(-1, target_size*target_size).contiguous()

    golden_scores = flatten_scores.gather(
        dim=1, index=tag_tensor.unsqueeze(1)).sum()

    # 计算all path scores
    # scores_upto_t[i, j]表示第i个句子的第t个词被标注为j标记的所有t时刻事前的所有子路径的分数之和
    scores_upto_t = torch.zeros(batch_size, target_size).to(device)
    for t in range(max_len):
        # 当前时刻 有效的batch_size（因为有些序列比较短)
        batch_size_t = (lengths > t).sum().item()
        if t == 0:
            scores_upto_t[:batch_size_t] = crf_scores[:batch_size_t,
                                                      t, start_id, :]
        else:
            # We add scores at current timestep to scores accumulated up to previous
            # timestep, and log-sum-exp Remember, the cur_tag of the previous
            # timestep is the prev_tag of this timestep
            # So, broadcast prev. timestep's cur_tag scores
            # along cur. timestep's cur_tag dimension
            scores_upto_t[:batch_size_t] = torch.logsumexp(
                crf_scores[:batch_size_t, t, :, :] +
                scores_upto_t[:batch_size_t].unsqueeze(2),
                dim=1
            )
    all_path_scores = scores_upto_t[:, end_id].sum()

    # logloss = -log(loss)
    loss = (all_path_scores - golden_scores) / batch_size
    return loss


def indexed(tag_tensor, tagset_size, start_id):
    # 将tag_tensor中的数转化为在[T*T]大小序列中的索引,T是标注的种类,便于在reshape中的数组进行索引
    batch_size, max_len = tag_tensor.size()
    for col in range(max_len-1, 0, -1):
        tag_tensor[:, col] += (tag_tensor[:, col-1] * tagset_size)
    tag_tensor[:, 0] += (start_id * tagset_size)
    return tag_tensor
