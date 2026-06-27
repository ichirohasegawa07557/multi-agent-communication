from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

def savefig(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()

def plot_bar(df, category_col, value_col, path, title, ylabel):
    plt.figure(figsize=(8, 5))
    plt.bar(df[category_col].astype(str), df[value_col])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=15, ha="right")
    savefig(path)

def plot_message_usage(messages, path):
    counts = messages["symbol"].value_counts().reset_index()
    counts.columns = ["symbol", "count"]
    plt.figure(figsize=(8, 5))
    plt.barh(counts["symbol"], counts["count"])
    plt.xlabel("count")
    plt.title("Message symbol usage")
    savefig(path)

def plot_matrix(mat, path, title, colorbar_label="score"):
    plt.figure(figsize=(6, 5))
    plt.imshow(mat, cmap="viridis")
    plt.colorbar(label=colorbar_label)
    plt.xlabel("source")
    plt.ylabel("target")
    plt.title(title)
    savefig(path)

def rollout_gif(df, path):
    cols = [c for c in df.columns if c.startswith("x") and c[1:].isdigit()]
    n = len(cols)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    px, py = np.cos(angles), np.sin(angles)
    imgs = []
    for idx in range(0, len(df), max(1, len(df)//20)):
        vals = df[cols].iloc[idx].values
        fig, ax = plt.subplots(figsize=(5, 5))
        sc = ax.scatter(px, py, c=vals, s=320, cmap="coolwarm", vmin=-2, vmax=2)
        for i in range(n):
            ax.text(px[i], py[i], f"x{i}", ha="center", va="center")
        ax.set_title(f"multi-agent world state t={idx}")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis("off")
        fig.colorbar(sc, ax=ax)
        fig.tight_layout()
        fig.canvas.draw()
        imgs.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)
    imageio.mimsave(path, imgs, duration=0.28)
