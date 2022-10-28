import lib
import numpy as np
import torch
import pandas as pd

class Evaluation(object):
    def __init__(self, model, loss_func, item_mapper, audio_dir, use_cuda, k=20):
        self.model = model
        self.loss_func = loss_func
        self.topk = k
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.item_mapper = item_mapper
        self.audio_dir = audio_dir

    def eval(self, eval_data, k):
        self.model.eval()
        losses = []
        recalls = []
        mrrs = []
        result = []
        #TODO
        dataloader = lib.DataLoader(eval_data, self.item_mapper, self.audio_dir, batch_size=100)
        with torch.no_grad():
            hidden = self.model.init_hidden()
            for input, target, mask in dataloader:
                input = input.to(self.device)
                target = target.to(self.device)
                logit, hidden = self.model(input, hidden)
                logit_sampled = logit[:, target.view(-1)]
                loss = self.loss_func(logit_sampled)

                for i in range(logit.cpu().detach().numpy().shape[0]):
                    dict_out = {}
                    dict_out['missing_terms'] = target.cpu().detach().numpy()[i]
                    _, indices = torch.topk(logit, k, -1)
                    dict_out['recommended_terms'] = " ".join([str(x) for x in indices.cpu().detach().numpy()[i,:]])
                    result.append(dict_out)


                recall, mrr = lib.evaluate(logit, target, k=k)

                # torch.Tensor.item() to get a Python number from a tensor containing a single value
                losses.append(loss.item())
                recalls.append(recall)
                mrrs.append(mrr.item())
        mean_losses = np.mean(losses)
        mean_recall = np.mean(recalls)
        mean_mrr = np.mean(mrrs)
        pd.DataFrame(result).to_csv("result_rec.csv", index=False)

        return mean_losses, mean_recall, mean_mrr
