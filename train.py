import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from fewshot_re_kit.data_loader import get_loader
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import BERTSentenceEncoder
import models
from models.hcrp import HCRP
import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import torch
import random
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./data',
                        help='file root')
    parser.add_argument('--train', default='train_wiki',
                        help='train file')
    parser.add_argument('--val', default='val_wiki',
                        help='val file')
    parser.add_argument('--test', default='val_wiki',
                        help='test file')
    parser.add_argument('--ispubmed', default=False, type=bool,
                       help='FewRel 2.0 or not')
    parser.add_argument('--pid2name', default='pid2name',
                        help='pid2name file: relation names and description')
    parser.add_argument('--trainN', default=10, type=int,
                        help='N in train')
    parser.add_argument('--N', default=10, type=int,
                        help='N way')
    parser.add_argument('--K', default=1, type=int,
                        help='K shot')
    parser.add_argument('--Q', default=1, type=int,
                        help='Num of query per class')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='batch size')
    parser.add_argument('--train_iter', default=30000, type=int,
                        help='num of iters in training')
    parser.add_argument('--val_iter', default=1000, type=int,
                        help='num of iters in validation')
    parser.add_argument('--test_iter', default=10000, type=int,
                        help='num of iters in testing')
    parser.add_argument('--val_step', default=1000, type=int,
                        help='val after training how many iters')
    parser.add_argument('--model', default='HCRP',
                        help='model name')
    parser.add_argument('--encoder', default='bert',
                        help='encoder: bert')
    parser.add_argument('--max_length', default=128, type=int,
                        help='max length')
    parser.add_argument('--lr', default=2e-5, type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float,
                        help='weight decay')
    parser.add_argument('--lamda', default=2.5, type=float,
                        help='loss combination')
    parser.add_argument('--grad_iter', default=1, type=int,
                        help='accumulate gradient every x iterations')
    parser.add_argument('--optim', default='adamw',
                        help='sgd / adam / adamw')
    parser.add_argument('--hidden_size', default=768, type=int,
                        help='hidden size')
    parser.add_argument('--load_ckpt', default=None,
                        help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
                        help='save ckpt')
    parser.add_argument('--only_test', default=False,
                        help='only test')
    parser.add_argument('--pretrain_ckpt', default='bert-base-uncased',
                        help='bert / roberta pre-trained checkpoint')
    parser.add_argument('--seed', default=19961228, type=int,
                        help='seed')
    parser.add_argument('--path', default=None,
                        help='path to ckpt')

    opt = parser.parse_args()
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    encoder_name = opt.encoder
    max_length = opt.max_length

    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))

    # encoder
    sentence_encoder = BERTSentenceEncoder(opt.pretrain_ckpt, max_length, path=opt.path)

    # train / val / test data loader
    train_data_loader = get_loader(opt.train, opt.pid2name, sentence_encoder,
                                   N=trainN, K=K, Q=Q, batch_size=batch_size, ispubmed=opt.ispubmed, root=opt.root)
    val_data_loader = get_loader(opt.val, opt.pid2name, sentence_encoder,
                                 N=N, K=K, Q=Q, batch_size=batch_size, ispubmed=opt.ispubmed, root=opt.root)
    test_data_loader = get_loader(opt.test, opt.pid2name, sentence_encoder,
                                  N=N, K=K, Q=Q, batch_size=batch_size, ispubmed=opt.ispubmed, root=opt.root)

    framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)

    model = HCRP(sentence_encoder, hidden_size=opt.hidden_size, max_len=max_length)
    if torch.cuda.is_available():
        model.cuda()

    # model save path
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    prefix = '-'.join([model_name, encoder_name, opt.train, opt.val, str(N), str(K)])
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if not opt.only_test:
        T1 = time.clock()
        framework.train(model, prefix, trainN, N, K, Q, learning_rate=opt.lr, weight_decay=opt.weight_decay,
                        lamda=opt.lamda, train_iter=opt.train_iter, val_iter=opt.val_iter,
                        load_ckpt=opt.load_ckpt, save_ckpt=ckpt, val_step=opt.val_step, grad_iter=opt.grad_iter)
        T2 = time.clock()
        print('total training time:%s s' % (T2 - T1))

    else:
        ckpt = opt.load_ckpt

    T3 = time.clock()
    acc = framework.eval(model, N, K, Q, opt.test_iter, ckpt=ckpt)
    T4 = time.clock()
    print('total evaluation time:%s s' % (T4 - T3))
    print("RESULT: %.2f" % (acc * 100))


if __name__ == "__main__":
    main()
