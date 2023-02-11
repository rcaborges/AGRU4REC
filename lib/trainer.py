import lib
import time
import torch
import numpy as np
import os


class Trainer(object):
    def __init__(self, model, train_data, eval_data, optim, use_cuda, loss_func, item_mapper, args):
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.optim = optim
        self.loss_func = loss_func
        self.evaluation = lib.Evaluation(self.model, self.loss_func, item_mapper, args.audio_dir, use_cuda)
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.args = args
        self.item_mapper = item_mapper
        self.batch_size = args.batch_size
        self.audio_dir = args.audio_dir


    def train(self, start_epoch, end_epoch, resume_training, start_time=None):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time

        if resume_training:
            recall_lst = list(np.load("recall.npy"))
            mrr_lst = list(np.load("mrr.npy"))
            loss_lst = list(np.load("loss.npy"))
        else:
            recall_lst, mrr_lst, loss_lst = [],[],[]

        for epoch in range(start_epoch, end_epoch + 1):
            st = time.time()
            train_loss = self.train_epoch(epoch)

            loss, recall, mrr = self.evaluation.eval(self.eval_data, self.batch_size, 'val', k=100)
            loss_lst.append(loss)
            recall_lst.append(recall)
            mrr_lst.append(mrr)

            print("{}: Epoch: {}, loss: {:.2f}, recall: {:.2f}, mrr: {:.2f}, time: {}".format(50, epoch, loss, recall, mrr, time.time() - st))

            checkpoint = {
                'model': self.model,
                'args': self.args,
                'epoch': epoch,
                'optim': self.optim,
                'loss': loss,
                'recall': recall,
                'mrr': mrr
            }

            model_name = os.path.join(self.args.checkpoint_dir, "model.pt")
            torch.save(checkpoint, model_name)
            print("Save model as %s" % model_name)
            np.save("recall.npy",recall_lst)
            np.save("mrr.npy",mrr_lst)
            np.save("loss.npy",loss_lst)

    def train_epoch(self, epoch):
        self.model.train()
        losses = []

        def reset_hidden(hidden, mask):
            """Helper function that resets hidden state when some sessions terminate"""
            if len(mask) != 0:
                hidden[:, mask, :] = 0
            return hidden

        hidden = self.model.init_hidden()
        dataloader = lib.DataLoader(self.train_data, self.item_mapper, self.audio_dir, batch_size=self.args.batch_size)
        for input_data, target, mask in dataloader:
            input_data = input_data.to(self.device)
            target = target.to(self.device)
            self.optim.zero_grad()
            hidden = reset_hidden(hidden, mask).detach()
            logit, hidden = self.model(input_data, hidden)
            # output sampling
            logit_sampled = logit[:, target.view(-1)]
            loss = self.loss_func(logit_sampled)
            losses.append(loss.item())
            loss.backward()
            self.optim.step()

        mean_losses = np.mean(losses)
        return mean_losses
