import os
import torch

class Config:
    # ==== General ====
    project_name = 'lane-detection-scnn'
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    # ==== Dataset ====
    dataset_root = os.path.join('data', 'CULane')  # Update path if needed
    train_split = 'list/train_gt.txt'
    val_split = 'list/val_gt.txt'
    test_split = 'list/test.txt'

    image_width = 1640
    image_height = 590
    num_classes = 5  # background + 4 lanes

    # ==== Training ====
    epochs = 50
    batch_size = 8
    learning_rate = 1e-4
    weight_decay = 1e-5
    num_workers = 4
    save_interval = 5  # save every 5 epochs

    # ==== Model ====
    backbone = 'vgg'  # or 'resnet'
    pretrained = True

    # ==== Paths ====
    save_dir = 'checkpoints'
    log_dir = 'logs'

    # ==== Visualization ====
    visualize = True
    vis_interval = 1  # visualize every N epochs

config = Config()
