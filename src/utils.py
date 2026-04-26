import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """Bidirectional contrastive retrieval loss used by HRCH."""

    def __init__(self, margin=0.0, shift=2.0, measure=False, max_violation=False):
        super().__init__()
        self.margin = margin
        self.shift = shift
        self.sim = order_sim if measure == "order" else (lambda x, y: x.mm(y.t()))
        self.max_violation = max_violation

    def set_margin(self, margin):
        self.margin = margin

    def forward(self, image_features, text_features=None, tau=1.0, lab=None):
        if text_features is None:
            scores = image_features
            diagonal = scores[:, 0].view(scores.size(0), 1)
            margin_cost = (self.margin + scores - diagonal.expand_as(scores)).clamp(min=0)
            return margin_cost.max(1)[0].sum() if self.max_violation else margin_cost.sum()

        scores = self.sim(image_features, text_features)
        diagonal = scores.diag().view(image_features.size(0), 1)
        cost_s = _shifted_cost(scores, diagonal.expand_as(scores), self.margin, self.shift)
        cost_i = _shifted_cost(scores, diagonal.t().expand_as(scores), self.margin, self.shift)
        sentence_loss = (-cost_s.diag() + tau * (cost_s / tau).exp().sum(1).log() + self.margin).mean()
        image_loss = (-cost_i.diag() + tau * (cost_i / tau).exp().sum(0).log() + self.margin).mean()
        return sentence_loss + image_loss


def _shifted_cost(scores, reference, margin, shift):
    mask = (scores >= (reference - margin)).float().detach()
    return scores * mask + (1.0 - mask) * (scores - shift)


def order_sim(image_features, text_features):
    positive = torch.relu(text_features.unsqueeze(1) - image_features.unsqueeze(0))
    return -positive.pow(2).sum(2).sqrt().t()
