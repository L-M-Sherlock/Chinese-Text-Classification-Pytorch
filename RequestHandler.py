import torch
from importlib import import_module
import pickle as pkl
import argparse
ss

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'HITSZQA'  # 数据集
    UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    config.batch_size = 1
    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path))
    model.eval()

    def sentence_to_index(sentence, pad_size=32):
        index = []
        vocab = pkl.load(open(config.vocab_path, 'rb'))
        tokenizer = lambda x: [y for y in x]
        lin = sentence.strip()
        content, label = lin.split('\t')
        words_line = []
        token = tokenizer(content)
        seq_len = len(token)
        if pad_size:
            if len(token) < pad_size:
                token.extend([PAD] * (pad_size - len(token)))
            else:
                token = token[:pad_size]
                seq_len = pad_size
        # word to id
        for word in token:
            words_line.append(vocab.get(word, vocab.get(UNK)))
        index.append((words_line, int(label), seq_len))
        return index


    for i, (trains, labels) in enumerate(build_iterator(sentence_to_index('学校有哪些体育设施\t0', config.pad_size), config)):
        predict = model(trains).data.max(1, keepdim=True)[1]
        print(predict)