import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config
from datasets.culane import CULaneDataset
from models.scnn import SCNN
from utils.metrics import pixel_accuracy, mean_iou, precision_recall_f1
from utils.visualizer import visualize_predictions


def test(split='val', save_vis=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    # Load dataset
    list_file = config.val_list if split == 'val' else config.test_list
    dataset = CULaneDataset(config.dataset_root, list_file)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Load model
    model = SCNN(backbone=config.backbone, pretrained=False, num_classes=config.num_classes)
    model.load_state_dict(torch.load(os.path.join(config.save_dir, 'best_model.pth'), map_location=device))
    model.to(device).eval()

    # Metrics accumulators
    total_acc = 0
    total_iou = 0
    all_precision = []
    all_recall = []
    all_f1 = []

    for images, masks in tqdm(loader, desc="Testing"):
        images, masks = images.to(device), masks.to(device)
        with torch.no_grad():
            outputs = model(images)

        total_acc += pixel_accuracy(outputs, masks)
        total_iou += mean_iou(outputs, masks, config.num_classes)
        p, r, f1 = precision_recall_f1(outputs, masks, positive_classes=list(range(1, config.num_classes)))
        all_precision.append(p)
        all_recall.append(r)
        all_f1.append(f1)

        if save_vis:
            preds = outputs.argmax(dim=1)
            visualize_predictions(images, masks, preds, save_dir=os.path.join(config.log_dir, 'test_vis'))

    n = len(loader)
    print(f"Avg Pixel Acc: {total_acc/n:.4f}")
    print(f"Avg mIoU: {total_iou/n:.4f}")
    print(f"Precision: {sum(all_precision)/n:.4f}")
    print(f"Recall:    {sum(all_recall)/n:.4f}")
    print(f"F1 Score:  {sum(all_f1)/n:.4f}")

if __name__ == '__main__':
    test(split='val', save_vis=True)
