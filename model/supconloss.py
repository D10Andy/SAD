import torch
import torch.nn.functional as F


class SupConLoss(torch.nn.Module):

    def __init__(self, temperature=0.5, contrast_mode='all',
                 base_temperature=0.07, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, x, labels=None, dev_score=None, mask=None):
        x = F.normalize(x, p=2, dim=1)
        features = x

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        ##############
        dev_score = torch.where(dev_score > 5.0, torch.ones_like(dev_score)*5.0, dev_score)
        dev_score = torch.where(dev_score < -5.0, torch.ones_like(dev_score)*(-5.0), dev_score)
        dev_score = dev_score.view(1, -1)
        dev_score_b = (dev_score-dev_score.T).abs()
        mask = torch.where(dev_score_b.abs() <= 1, torch.ones_like(dev_score_b), torch.zeros_like(dev_score_b))
        mask = mask.float().to(device)
        ##############

        # labels = labels.contiguous().view(-1, 1)
        # mask = torch.eq(labels, labels.T).float().to(device)

        # print('>>>>>>>>>>>>>>>>>>mask', mask, '\t shape:', mask.shape)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        

        # tile mask
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask

        ################
        sim_coeff = 1.0/(1.0+dev_score_b)
        ################

        num_positives_per_row = torch.sum(positives_mask, axis=1)  # 除了自己之外，正样本的个数  [2 0 2 2]
        denominator = torch.sum(
            exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = (logits - torch.log(denominator + 1e-6))*sim_coeff
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = torch.sum(
            log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
                        num_positives_per_row > 0]
        '''
        计算正样本平均的log-likelihood
        考虑到一个类别可能只有一个样本，就没有正样本了 比如我们labels的第二个类别 labels[1,2,1,1]
        所以这里只计算正样本个数>0的
        '''
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss


