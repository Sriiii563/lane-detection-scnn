import numpy as np
import torch

def pixel_accuracy(output, mask):
    """
    Compute pixel-wise accuracy.
    output: torch.Tensor of shape (B, C, H, W)
    mask: torch.LongTensor of shape (B, H, W)
    """
    with torch.no_grad():
        preds = output.argmax(dim=1)
        correct = (preds == mask).float()
        return correct.sum() / correct.numel()


def intersection_and_union(output, mask, num_classes):
    """
    Compute intersection and union per class.
    Returns:
        area_inter: torch.Tensor of shape (num_classes,)
        area_union: torch.Tensor of shape (num_classes,)
    """
    with torch.no_grad():
        preds = output.argmax(dim=1)
        preds = preds.view(-1)
        mask = mask.view(-1)

        area_inter = torch.zeros(num_classes, dtype=torch.long)
        area_union = torch.zeros(num_classes, dtype=torch.long)

        for cls in range(num_classes):
            pred_inds = preds == cls
            target_inds = mask == cls
            intersection = (pred_inds & target_inds).sum().item()
            union = (pred_inds | target_inds).sum().item()
            area_inter[cls] = intersection
            area_union[cls] = union

        return area_inter, area_union


def mean_iou(output, mask, num_classes):
    """
    Compute mean IoU across classes (excluding background if desired).
    """
    area_inter, area_union = intersection_and_union(output, mask, num_classes)
    iou = area_inter.float() / (area_union.float() + 1e-6)
    return iou.mean().item()


def precision_recall_f1(output, mask, positive_classes):
    """
    Compute precision, recall, and F1 for specified positive classes.
    positive_classes: list of class indices to treat as "lane"
    """
    with torch.no_grad():
        preds = output.argmax(dim=1).view(-1)
        mask = mask.view(-1)

        tp = ((preds.unsqueeze(1) == torch.tensor(positive_classes)) &
              (mask.unsqueeze(1) == torch.tensor(positive_classes))).any(dim=1).sum().item()
        fp = ((preds.unsqueeze(1) == torch.tensor(positive_classes)) &
              (mask.unsqueeze(1) != preds.unsqueeze(1))).any(dim=1).sum().item()
        fn = ((mask.unsqueeze(1) == torch.tensor(positive_classes)) &
              (preds.unsqueeze(1) != mask.unsqueeze(1))).any(dim=1).sum().item()

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        return precision, recall, f1