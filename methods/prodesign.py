from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import os

from .base_method import Base_method
from .utils import cuda
from .prodesign_model import ProDesign_Model
from torch_scatter import scatter_sum


def top_k_acc(logits:torch.Tensor, targets:torch.Tensor, k:int):
    """
    author: Jens Lundsgaard 

    logits: B, num_classes
    targets: B, type=int/long > 0 

    returns: float \in [0,1] representing acc over the batch
    """
    assert k > 0, "k must be >0"
    if (k == 1):
        preds = F.one_hot(logits.argmax(dim=-1), num_classes=logits.shape[1]).float()
        return torch.einsum("bi,bi->b", preds, F.one_hot(targets, num_classes=logits.shape[1]).float()).sum().item() / logits.shape[0]

    hot_logits = torch.zeros_like(logits) 
    indices = torch.topk(logits, k, dim=-1).indices
    hot_logits = hot_logits.scatter_(1, indices, 1).float()
    correct_mask = torch.einsum("bi,bi->b", hot_logits, F.one_hot(targets, num_classes=logits.shape[1]).float())
    return correct_mask.sum().item() / correct_mask.shape[0] 
def batched_bincount(x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
   
    if x.dtype != torch.long:
        raise TypeError("Input tensor x must be of dtype torch.long")
    if values.ndim != 1:
        raise ValueError("values must be a 1D tensor of unique possible values")

    B, N = x.shape
    num_values = values.numel()

    value_to_index = torch.empty(values.max().item() + 1, dtype=torch.long, device=x.device)
    value_to_index[values] = torch.arange(num_values, device=x.device)
    idx = value_to_index[x]  # Shape: (B, N)

    counts = torch.zeros((B, num_values), dtype=torch.long, device=x.device)

    counts.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.long))

    return counts 


class ProDesign(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer, self.scheduler = self._init_optimizer(steps_per_epoch)

    def _build_model(self):
        return ProDesign_Model(self.args).to(self.device)

    def train_one_epoch(self, train_loader):
        self.model.train()
        train_sum, train_weights = 0., 0.

        train_pbar = tqdm(train_loader)
        for step_idx, batch in enumerate(train_pbar):
            self.optimizer.zero_grad()  # 模型中所有可学习参数的梯度归零
            X, S, score, mask, lengths = cuda(batch, device=self.device)
            X, S, score, h_V, h_E, E_idx, batch_id, mask_bw, mask_fw, decoding_order = self.model._get_features(S,
                                                                                                                score,
                                                                                                                X=X,
                                                                                                                mask=mask)
            log_probs = self.model(h_V, h_E, E_idx, batch_id,S,mask)
            loss = self.criterion(log_probs, S)  # 计算预测值和目标值s之间的损失
            loss.backward()  # 计算损失函数对模型参数的梯度
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()  # 更新模型的参数
            self.scheduler.step()  # 更新优化器的学习率的方法

            train_sum += torch.sum(loss * mask).cpu().data.numpy()
            train_weights += torch.sum(mask).cpu().data.numpy()
            train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

        train_loss = train_sum / train_weights
        train_perplexity = np.exp(train_loss)
        return train_loss, train_perplexity

    def valid_one_epoch(self, valid_loader, epoch=-1):
        self.model.eval()
        valid_losses = []
        valid_pbar = tqdm(valid_loader)

        valid_acc_1s, valid_acc_5s, valid_acc_10s = [],[],[]

        os.makedirs(os.path.join("..", "plots"), exist_ok=True)
        with torch.no_grad():
            for i, batch in enumerate(valid_pbar):
                X, S, score, mask, lengths = cuda(batch, device=self.device)


                assert (lengths == X.shape[1]).all().item(), "dataloader is not setup for majority voting"
                X, S, score, h_V, h_E, E_idx, batch_id, mask_bw, mask_fw, decoding_order = self.model._get_features(S,
                                                                                                                    score,
                                                                                                                    X=X,
                                                                                                                    mask=mask)


                _, logits = self.model(h_V, h_E, E_idx, batch_id,S,mask, return_logit=True)
                loss = self.criterion(logits, S)

                logits = logits.reshape(200, -1, 20)
                S = S.reshape(200, -1)
                batch_id = batch_id.reshape(200, -1)
                assert (batch_id == torch.arange(200, device=batch_id.device)[:, None]).all().item(), "S is misshapen"

                valid_losses.append(loss.cpu().item())

                logits = F.softmax(logits)
                if epoch == 7:
                    violin_logits = logits[::10, :20].cpu()
                    for j in range(violin_logits.shape[1]):
                        distributions = violin_logits[:, j].numpy()
                        fig, ax = plt.subplots()
                        ax.violinplot(distributions, np.arange(distributions.shape[0]), points=60, widths=0.7, showmeans=True, showextrema=True, showmedians=True)
                        fig.savefig(os.path.join("..", f"{i}_{j}.png"))
                        plt.close(fig)


                grouped_logits = logits.argmax(dim=-1).T # num_res, 200
                new_logits = batched_bincount(grouped_logits) # num_res, 20

                targets = S[0] # num_res

                valid_acc_1s.append(top_k_acc(new_logits, targets, 1)) 
                valid_acc_5s.append(top_k_acc(new_logits, targets, 5)) 
                valid_acc_10s.append(top_k_acc(new_logits, targets, 10)) 

                valid_pbar.set_description('valid loss: {:.4f}'.format(loss.cpu().item()))

        return np.array(valid_losses), np.array(valid_acc_1s), np.array(valid_acc_5s), np.array(valid_acc_10s)

    def test_one_epoch(self, test_loader):
        self.model.eval()
        test_sum, test_weights = 0., 0.
        test_pbar = tqdm(test_loader)

        with torch.no_grad():
            for batch in test_pbar:
                X, S, score, mask, lengths = cuda(batch, device=self.device)
                X, S, score, h_V, h_E, E_idx, batch_id, mask_bw, mask_fw, decoding_order = self.model._get_features(S,
                                                                                                                    score,
                                                                                                                    X=X,
                                                                                                                    mask=mask)
                log_probs = self.model(h_V, h_E, E_idx, batch_id,S,mask)
                loss, loss_av = self.loss_nll_flatten(S, log_probs)
                mask = torch.ones_like(loss)
                test_sum += torch.sum(loss * mask).cpu().data.numpy()
                test_weights += torch.sum(mask).cpu().data.numpy()
                test_pbar.set_description('test loss: {:.4f}'.format(loss.mean().item()))

            test_recovery, test_subcat_recovery = self._cal_recovery(test_loader.dataset, test_loader.featurizer)

        test_loss = test_sum / test_weights
        test_perplexity = np.exp(test_loss)

        return test_perplexity, test_recovery, test_subcat_recovery

    def _cal_recovery(self, dataset, featurizer):
        self.residue_type_cmp = torch.zeros(20, device='cuda:0')  # 创建两个张量，前者存储某种比较值，后者存储某种计数值
        self.residue_type_num = torch.zeros(20, device='cuda:0')
        recovery = []
        subcat_recovery = {}
        with torch.no_grad():  # 告诉PyTorch不要计算梯度
            for protein in tqdm(dataset):
                p_category = protein['category'] if 'category' in protein.keys() else 'Unknown'
                if p_category not in subcat_recovery.keys():
                    subcat_recovery[p_category] = []

                protein = featurizer([protein])
                X, S, score, mask, lengths = cuda(protein, device=self.device)
                X, S, score, h_V, h_E, E_idx, batch_id, mask_bw, mask_fw, decoding_order = self.model._get_features(S,
                                                                                                                    score,
                                                                                                                    X=X,
                                                                                                                    mask=mask)
                log_probs = self.model(h_V, h_E, E_idx, batch_id,S,mask)
                S_pred = torch.argmax(log_probs, dim=1)
                cmp = (S_pred == S)
                recovery_ = cmp.float().mean().cpu().numpy()

                self.residue_type_cmp += scatter_sum(cmp.float(), S.long(), dim=0, dim_size=20)
                self.residue_type_num += scatter_sum(torch.ones_like(cmp.float()), S.long(), dim=0, dim_size=20)

                if np.isnan(recovery_): recovery_ = 0.0

                subcat_recovery[p_category].append(recovery_)
                recovery.append(recovery_)

            for key in subcat_recovery.keys():
                subcat_recovery[key] = np.median(subcat_recovery[key])

        self.mean_recovery = np.mean(recovery)
        self.std_recovery = np.std(recovery)
        self.min_recovery = np.min(recovery)
        self.max_recovery = np.max(recovery)
        self.median_recovery = np.median(recovery)
        recovery = np.median(recovery)
        return recovery, subcat_recovery

    def loss_nll_flatten(self, S, log_probs):
        """ Negative log probabilities """
        criterion = torch.nn.NLLLoss(reduction='none')  # 表示不对损失进行平均或求和，而是返回每个样本的单独损失
        loss = criterion(log_probs, S)
        loss_av = loss.mean()
        return loss, loss_av

    def loss_nll_smoothed(self, S, log_probs, weight=0.1):
        """ Negative log probabilities """
        S_onehot = torch.nn.functional.one_hot(S, num_classes=20).float()
        S_onehot = S_onehot + weight / float(S_onehot.size(-1))
        S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)  # [4, 463, 20]/[4, 463, 1] --> [4, 463, 20]

        loss = -(S_onehot * log_probs).sum(-1).mean()
        loss_av = torch.sum(loss)
        return loss, loss_av
