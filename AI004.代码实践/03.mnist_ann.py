import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class SimpleANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 100),
            nn.GELU(),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        return self.net(x)

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # use class indices directly for CrossEntropyLoss (expects raw logits)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}: Train loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # compute cross-entropy loss using class indices
            loss = criterion(output, target)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    acc = 100.0 * correct / (len(test_loader.dataset))
    print(f"Test loss: {test_loss:.4f}, Accuracy: {acc:.2f}%")
    return test_loss, acc

def interactive_browser(model, device, test_dataset, n=20):
    """Interactive viewer: show n random test samples, Down arrow -> refresh, Q -> quit."""
    model.eval()
    mean = 0.1307
    std = 0.3081
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    # flatten axes array for easy indexing
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    current_indices = []

    def draw_new():
        nonlocal current_indices
        current_indices = np.random.choice(len(test_dataset), size=n, replace=False)
        with torch.no_grad():
            for ax in axes_flat:
                ax.clear()
                ax.axis('off')
            for i, idx in enumerate(current_indices):
                img_t, label = test_dataset[idx]
                output = model(img_t.unsqueeze(0).to(device))
                pred = int(output.argmax(dim=1).item())
                img = img_t * std + mean
                img_np = img.squeeze(0).cpu().numpy()
                ax = axes_flat[i]
                ax.imshow(img_np, cmap='gray')
                # add red border if prediction != label
                if pred != int(label):
                    rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor='red', linewidth=3)
                    ax.add_patch(rect)
                color = 'red' if pred != int(label) else 'black'
                ax.set_title(f"pred: {pred} (true: {int(label)})", color=color)
                ax.axis('off')
            fig.suptitle('Interactive samples (press Down to refresh, Q to quit)')
            fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ('down', 'pagedown', 'right'):
            draw_new()
        elif event.key in ('q', 'Q'):
            plt.close(fig)
        elif event.key in ('w', 'W'):
            print("Displaying hidden and output layer weights...")
            display_weights(model)

    cid = fig.canvas.mpl_connect('key_press_event', on_key)
    draw_new()
    plt.show()
    try:
        fig.canvas.mpl_disconnect(cid)
    except Exception:
        pass

def display_weights(model, save_dir=None, dpi=120):
    """Display all weights as images.

    - Hidden layer weights (Linear(28*28, H)) are reshaped to (H, 28, 28) and shown in a grid.
    - Output layer weights (Linear(H, C)) are reshaped to (C, out_h, out_w) where (out_h, out_w)
      are factor pairs of H chosen near a square (e.g., 10x10 for H=100), and shown in a grid.
    If save_dir is provided, the grids and individual weight images are saved there.
    """
    try:
        w_hidden = model.net[1].weight.detach().cpu().numpy()  # (H, 28*28)
        b_hidden = model.net[1].bias.detach().cpu().numpy()
        w_out = model.net[3].weight.detach().cpu().numpy()     # (C, H)
        b_out = model.net[3].bias.detach().cpu().numpy()
    except Exception as e:
        print("Unable to extract weights:", e)
        return

    import matplotlib.pyplot as plt

    # --- Hidden layer: reshape each weight vector to 28x28 image ---
    try:
        H = w_hidden.shape[0]
        w_hidden_imgs = w_hidden.reshape(H, 28, 28)
    except Exception as e:
        print("Failed to reshape hidden weights to 28x28:", e)
        return

    cols = 10
    rows = int(np.ceil(H / cols))
    fig_h, axes_h = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    axes_h_flat = axes_h.flatten() if hasattr(axes_h, 'flatten') else [axes_h]
    vmin = w_hidden_imgs.min()
    vmax = w_hidden_imgs.max()
    im = None
    for i in range(rows * cols):
        ax = axes_h_flat[i]
        ax.axis('off')
        if i < H:
            im = ax.imshow(w_hidden_imgs[i], cmap='seismic', vmin=vmin, vmax=vmax)
            ax.set_title(f'h{i}', fontsize=8)
    fig_h.suptitle('Hidden layer weights (each 28x28)')
    fig_h.subplots_adjust(wspace=0.3, hspace=0.5)
    if im is not None:
        cax = fig_h.add_axes([0.92, 0.15, 0.02, 0.7])
        fig_h.colorbar(im, cax=cax)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig_h.savefig(os.path.join(save_dir, 'hidden_weights_grid.png'), dpi=dpi, bbox_inches='tight')
        for i in range(H):
            plt.imsave(os.path.join(save_dir, f'hidden_w_{i:03d}.png'), w_hidden_imgs[i], cmap='seismic', vmin=vmin, vmax=vmax)

    plt.show()

    # --- Output layer: reshape each weight vector using factors of H (e.g., 10x10 for H=100) ---
    try:
        C = w_out.shape[0]
        # find factor pair of H near square
        def _factor_pair(n):
            r = int(np.sqrt(n))
            for i in range(r, 0, -1):
                if n % i == 0:
                    return i, n // i
            return n, 1
        out_h, out_w = _factor_pair(H)
        w_out_imgs = w_out.reshape(C, out_h, out_w)
    except Exception as e:
        print("Failed to reshape output weights:", e)
        return

    cols_o = min(5, C)
    rows_o = int(np.ceil(C / cols_o))
    fig_o, axes_o = plt.subplots(rows_o, cols_o, figsize=(cols_o * 2, rows_o * 2))
    axes_o_flat = axes_o.flatten() if hasattr(axes_o, 'flatten') else [axes_o]
    vmin_o = w_out_imgs.min()
    vmax_o = w_out_imgs.max()
    im2 = None
    for i in range(rows_o * cols_o):
        ax = axes_o_flat[i]
        ax.axis('off')
        if i < C:
            im2 = ax.imshow(w_out_imgs[i], cmap='seismic', vmin=vmin_o, vmax=vmax_o)
            ax.set_title(f'out{i} b={b_out[i]:.2f}', fontsize=9)
    fig_o.suptitle(f'Output layer weights (reshaped to {out_h}x{out_w})')
    fig_o.subplots_adjust(wspace=0.4, hspace=0.6)
    if im2 is not None:
        cax2 = fig_o.add_axes([0.92, 0.15, 0.02, 0.7])
        fig_o.colorbar(im2, cax=cax2)

    if save_dir:
        fig_o.savefig(os.path.join(save_dir, 'output_weights_grid.png'), dpi=dpi, bbox_inches='tight')
        for i in range(C):
            plt.imsave(os.path.join(save_dir, f'output_w_{i:02d}.png'), w_out_imgs[i], cmap='seismic', vmin=vmin_o, vmax=vmax_o)

    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)

model = SimpleANN().to(device)

# 数据变换和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

data_root = os.path.join(os.path.dirname(__file__), "data")
train_dataset = datasets.MNIST(root=data_root, train=True, transform=transform, download=False)
test_dataset = datasets.MNIST(root=data_root, train=False, transform=transform, download=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 使用交叉熵损失（CrossEntropyLoss），直接接受类索引作为目标（无需 one-hot）
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

print(f"Training on device: {device}. Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

for epoch in range(5):
    train(model, device, train_loader, optimizer, criterion, epoch)
    evaluate(model, device, test_loader, criterion)

# 显示隐藏层和输出层权重
display_weights(model)

interactive_browser(model, device, test_dataset, n=10)