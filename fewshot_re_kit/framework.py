import os
import numpy as np
import sys
from . import sentence_encoder
from . import data_loader
import torch
from torch import optim, nn
from torch.nn import functional as F
from transformers import AdamW, get_linear_schedule_with_warmup


class FewShotREModel(nn.Module):
    def __init__(self, sentence_encoder):
        """
        sentence_encoder: Sentence encoder
        """
        nn.Module.__init__(self)
        self.sentence_encoder = sentence_encoder  # nn.DataParallel(sentence_encoder)
        self.cost = nn.CrossEntropyLoss(reduction='none')
        self.gamma = 1
        self.reduction = 'mean'
        self.sigmoid = nn.Sigmoid()

    def forward(self, support, query, rel_text, N, K, total_Q, is_eval):
        """
        :param support: Inputs of the support set. (B*N*K)
        :param query: Inputs of the query set. (B*total_Q)
        :param rel_text: Inputs of the relation description.  (B*N)
        :param N: Num of classes
        :param K: Num of instances for each class in the support set
        :param total_Q: Num of instances in the query set
        :param is_eval:
        :return: logits, pred, logits_proto, labels_proto, sim_scalar
        """
        raise NotImplementedError

    def l2norm(self, X):
        norm = torch.pow(X, 2).sum(dim=-1, keepdim=True).sqrt()
        X = torch.div(X, norm)
        return X

    def loss(self, logits, label, weight=None):
        """
        logits: Logits with the size (..., class_num)
        label: Label with whatever size.
        return: [Loss] (A single value)
        """
        N = logits.size(-1)

        # focal weights
        logits_ = torch.softmax(self.l2norm(logits), dim=-1)
        logits_ = logits_.view(-1, N)
        logits, label = logits.view(-1, N), label.view(-1)
        probs = torch.stack([logits_[i, t] for i, t in enumerate(label)])
        focal_weight = torch.pow(1 - probs, self.gamma)

        # task adaptive weights
        if weight is not None:
            focal_weight = focal_weight * weight.view(-1)

        # add weights to cross entropy
        ce_loss = self.cost(logits, label)  # (B*totalQ)
        tf_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            tf_loss = (tf_loss / focal_weight.sum()).sum()
        elif self.reduction == 'sum':
            tf_loss = tf_loss.sum()

        return tf_loss

    def accuracy(self, pred, label):
        """
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        """
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))


class FewShotREFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader):
        """
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        """
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.bce_loss = nn.BCEWithLogitsLoss()

    def __load_model__(self, ckpt):
        """
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        """
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def item(self, x):
        """
        PyTorch before and after 0.4
        """
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def train(self,
              model,
              model_name,
              N_for_train, N_for_eval, K, Q,
              learning_rate=2e-5,
              weight_decay=0.01,
              lamda=1,
              train_iter=10000,
              val_iter=1000,
              val_step=2000,
              load_ckpt=None,
              save_ckpt=None,
              warmup_step=300,
              grad_iter=1):
        """
        model: a FewShotREModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        learning_rate: Initial learning rate
        weight_decay: Rate of decaying weight
        lamda: loss combination
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        """
        print("Start training...")

        # Init
        parameters_to_optimize = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize
                        if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in parameters_to_optimize
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(parameters_to_optimize, lr=learning_rate, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter)

        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
            start_iter = 0
        else:
            start_iter = 0

        # Training
        model.train()
        best_acc, iter_loss, iter_right, iter_sample = 0.0, 0.0, 0.0, 0.0
        for it in range(start_iter, start_iter + train_iter):
            support, query, label, rel_text = next(self.train_data_loader)
            if torch.cuda.is_available():
                for k in support:
                    support[k] = support[k].cuda()
                for k in query:
                    query[k] = query[k].cuda()
                for k in rel_text:
                    rel_text[k] = rel_text[k].cuda()
                label = label.cuda()
            logits, pred, logits_proto, labels_proto, sim_task = model(support, query, rel_text, N_for_train, K, Q * N_for_train)
            loss = model.loss(logits, label, sim_task) / float(grad_iter) + lamda * self.bce_loss(logits_proto, labels_proto)
            right = model.accuracy(pred, label)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            if it % grad_iter == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            iter_loss += self.item(loss.data)
            iter_right += self.item(right.data)
            iter_sample += 1

            sys.stdout.write(
                'step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample,
                                                                           100 * iter_right / iter_sample) + '\r')
            sys.stdout.flush()

            if (it + 1) % val_step == 0:
                acc = self.eval(model, N_for_eval, K, Q, val_iter)
                model.train()
                if acc > best_acc:
                    print('Best checkpoint')
                    torch.save({'state_dict': model.state_dict()}, save_ckpt)
                    best_acc = acc
                iter_loss = 0.
                iter_right = 0.
                iter_sample = 0.

        print("\n####################\n")
        print("Finish training " + model_name)

    def eval(self,
             model, N, K, Q,
             eval_iter,
             ckpt=None):
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")

        model.eval()
        if ckpt is None:
            eval_dataset = self.val_data_loader
        else:
            state_dict = self.__load_model__(ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                name = name.replace('module.', '')
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
                # print(name)
            eval_dataset = self.test_data_loader

        iter_right = 0.0
        iter_sample = 0.0
        with torch.no_grad():
            for it in range(eval_iter):

                support, query, label, rel_text = next(eval_dataset)
                if torch.cuda.is_available():
                    for k in support:
                        support[k] = support[k].cuda()
                    for k in query:
                        query[k] = query[k].cuda()
                    for k in rel_text:
                        rel_text[k] = rel_text[k].cuda()
                    label = label.cuda()
                logits, pred, _, _, _ = model(support, query, rel_text, N, K, Q * N, is_eval=True)

                right = model.accuracy(pred, label)
                iter_right += self.item(right.data)
                iter_sample += 1

                sys.stdout.write(
                    '[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) + '\r')
                sys.stdout.flush()
            print("")
        return iter_right / iter_sample
