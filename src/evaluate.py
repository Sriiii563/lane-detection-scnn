import os
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config
from datasets.culane import CULaneDataset
from models.scnn import SCNN
from utils.metrics import pixel_accuracy, mean_iou, precision_recall_f1


def evaluate(split='val', output_csv=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    # Load dataset
    list_file = config.val_list if split == 'val' else config.test_list
    dataset = CULaneDataset(config.dataset_root, list_file)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Load model
    model = SCNN(backbone=config.backbone, pretrained=False, num_classes=config.num_classes)
    model.load_state_dict(torch.load(os.path.join(config.save_dir, 'best_model.pth'), map_location=device))
    model.to(device).eval()

    # Prepare metrics
    metrics_list = []

    for idx, (images, masks) in enumerate(tqdm(loader, desc="Evaluating")):
        images, masks = images.to(device), masks.to(device)
        with torch.no_grad():
            outputs = model(images)

        acc = pixel_accuracy(outputs, masks).item() if hasattr(pixel_accuracy(outputs, masks), 'item') else pixel_accuracy(outputs, masks)
        miou = mean_iou(outputs, masks, config.num_classes)
        p, r, f1 = precision_recall_f1(outputs, masks, positive_classes=list(range(1, config.num_classes)))

        metrics_list.append({
            'batch_idx': idx,
            'pixel_acc': acc,
            'mIoU': miou,
            'precision': p,
            'recall': r,
            'f1': f1
        })

    # Compute overall averages
    avg_metrics = {
        'pixel_acc': sum(m['pixel_acc'] for m in metrics_list) / len(metrics_list),
        'mIoU': sum(m['mIoU'] for m in metrics_list) / len(metrics_list),
        'precision': sum(m['precision'] for m in metrics_list) / len(metrics_list),
        'recall': sum(m['recall'] for m in metrics_list) / len(metrics_list),
        'f1': sum(m['f1'] for m in metrics_list) / len(metrics_list),
    }

    print("=== Overall Metrics ===")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")

    # Save to CSV if needed
    if output_csv:
        keys = ['batch_idx', 'pixel_acc', 'mIoU', 'precision', 'recall', 'f1']
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metrics_list)
        print(f"Metrics saved to {output_csv}")

if __name__ == '__main__':
    evaluate(split='val', output_csv=os.path.join(config.log_dir, 'eval_metrics.csv'))