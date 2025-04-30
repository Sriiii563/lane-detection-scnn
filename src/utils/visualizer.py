import os
import matplotlib.pyplot as plt
import numpy as np

def overlay_mask(image, mask, alpha=0.5, colormap=None):
    """
    Overlay a segmentation mask on an image.
    image: numpy array HxWx3 (0-1 or 0-255)
    mask: numpy array HxW (integer classes)
    alpha: transparency of mask
    colormap: dict of {class: (r, g, b)} in 0-1 range
    """
    if image.max() <= 1.0:
        img = image
    else:
        img = image / 255.0

    h, w = mask.shape
    mask_rgb = np.zeros((h, w, 3))

    if colormap is None:
        # default random colors
        classes = np.unique(mask)
        colormap = {c: np.random.rand(3,) for c in classes if c != 0}
        colormap[0] = (0, 0, 0)

    for cls, color in colormap.items():
        mask_rgb[mask == cls] = color

    overlay = img * (1 - alpha) + mask_rgb * alpha
    return overlay


def visualize_predictions(images, masks, preds, save_dir=None):
    """
    Display or save side-by-side image, ground truth, and prediction.
    images: batch of torch.Tensor Bx3xHxW
    masks, preds: BxHxW
    """
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    preds = preds.cpu().numpy()
    B = images.shape[0]

    for i in range(B):
        img = np.transpose(images[i], (1, 2, 0))
        mask = masks[i]
        pred = preds[i]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].set_title('Image')
        axes[0].axis('off')

        axes[1].imshow(overlay_mask(img, mask))
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        axes[2].imshow(overlay_mask(img, pred))
        axes[2].set_title('Prediction')
        axes[2].axis('off')

        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'vis_{i}.png'))
            plt.close(fig)
        else:
            plt.show()