from copy import deepcopy
from torch import optim
from dataloader import load_data
from utils import extend_maps, preprocess_for_lstmcrf, save_model, sort_by_lengths, tensorize, loss_function
import time
import argparse
import torch
from model import BiLSTM_CRF
from evaluation import Metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--epoches', type=int, default=30)
    parser.add_argument('--print_step', type=int, default=0)
    parser.add_argument('--learning_rate', type=int, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()
    vars(args)['model'] = 'BiLSTM+CRF'

    print("loading data...")
    train_word_lists, train_tag_lists, word2id, tag2id = load_data('train')
    dev_word_lists, dev_tag_lists = load_data('dev', make_vocab=False)
    test_word_lists, test_tag_lists = load_data('test', make_vocab=False)

    print('training Bi-LSTM_CRF model...')
    crf_word2id, crd_tag2id = extend_maps(word2id, tag2id)
    train_word_lists, train_tag_lists = preprocess_for_lstmcrf(
        train_word_lists, train_tag_lists)
    dev_word_lists, dev_tag_lists = preprocess_for_lstmcrf(
        dev_word_lists, dev_tag_lists)
    test_word_lists, test_tag_lists = preprocess_for_lstmcrf(
        test_word_lists, test_tag_lists, test=True)

    vocab_size = len(word2id)
    out_size = len(tag2id)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 如果cuda版本不兼容，禁用cudnn
    # torch.backends.cudnn.enabled = False
    # 如果没有GPU，使用CPU
    device = torch.device('cpu')
    print(f'using device: {device}')

    model = BiLSTM_CRF(vocab_size, args.emb_size,
                       args.hidden_size, out_size).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    min_val_loss = 1e18
    optimal_model = None

    # training
    start = time.time()
    # 便于以batch为整体进行矩阵运算
    train_word_lists, train_tag_lists, _ = sort_by_lengths(
        train_word_lists, train_tag_lists)
    dev_word_lists, dev_tag_lists, _ = sort_by_lengths(
        dev_word_lists, dev_tag_lists)
    for epoch in range(1, args.epoches+1):
        losses = 0.0
        train_step = 0
        for idx in range(0, len(train_word_lists), args.batch_size):
            train_step += 1
            train_batch_sents = train_word_lists[idx: idx+args.batch_size]
            train_batch_tags = train_tag_lists[idx: idx+args.batch_size]

            model.train()
            train_sents_tensor, train_lengths = tensorize(
                train_batch_sents, word2id)
            train_sents_tensor = train_sents_tensor.to(device)
            train_tags_tensor, train_lengths = tensorize(
                train_batch_tags, tag2id)
            train_tags_tensor = train_tags_tensor.to(device)

            scores = model(train_sents_tensor, train_lengths)
            optimizer.zero_grad()
            loss = loss_function(scores, train_tags_tensor, tag2id).to(device)
            loss.backward()
            optimizer.step()

            losses += loss.item()

        model.eval()
        with torch.no_grad():
            val_losses = 0.0
            val_step = 0
            for ind in range(0, len(dev_word_lists), args.batch_size):
                val_step += 1
                dev_batch_sents = dev_word_lists[ind: ind+args.batch_size]
                dev_batch_tags = dev_tag_lists[ind: ind+args.batch_size]

                dev_sents_tensor, dev_lengths = tensorize(
                    dev_batch_sents, word2id)
                dev_sents_tensor = dev_sents_tensor.to(device)
                dev_tags_tensor, dev_lengths = tensorize(
                    dev_batch_tags, tag2id)
                dev_tags_tensor = dev_tags_tensor.to(device)

                dev_scores = model(dev_sents_tensor, dev_lengths)
                val_loss = loss_function(
                    dev_scores, dev_tags_tensor, tag2id).to(device)
                val_losses += val_loss.item()

            val_loss = val_losses/val_step

            if val_loss < min_val_loss:
                print('saving current model...')
                min_val_loss = val_loss
                optimal_model = deepcopy(model)

        print(
            f'Epoch {epoch}, Train Loss: {(losses/train_step):.4f}, Val Loss: {val_loss:.4f}')

    # 训练结束，保存最优模型
    save_model(optimal_model, f'./results/{args.model}.pkl')
    end = time.time()
    print(f'Training finished, costing {end-start}s')
    print(f'Apply {args.model} model to test data...')

    # testing
    test_word_lists, test_tag_lists, indices = sort_by_lengths(
        test_word_lists, test_tag_lists)
    test_sents_tensor, test_lengths = tensorize(test_word_lists, word2id)
    test_sents_tensor = test_sents_tensor.to(device)

    optimal_model.eval()
    with torch.no_grad():
        batch_tagids = optimal_model.test(
            test_sents_tensor, test_lengths, tag2id)

    pred_tag_lists = []
    id2tag = dict((_id, tag) for tag, _id in tag2id.items())
    for i, ids in enumerate(batch_tagids):
        tag_list = []
        for j in range(test_lengths[i] - 1):  # crf解码过程中，<end>被舍弃
            tag_list.append(id2tag[ids[j].item()])
        pred_tag_lists.append(tag_list)

    # indices存有根据长度排序后的索引映射的信息
    # 根据indices将pred_tag_lists和tag_lists转化为原来的顺序
    ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
    indices, _ = list(zip(*ind_maps))
    pred_tag_lists = [pred_tag_lists[i] for i in indices]
    tag_lists = [test_tag_lists[i] for i in indices]

    # 报告指标和混淆矩阵
    metrics = Metrics(tag_lists, pred_tag_lists, remove_O=False)
    metrics.report_scores()
    metrics.report_confusion_matrix()
