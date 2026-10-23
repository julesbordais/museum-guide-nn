import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import RESULTS


def learning_curves(hists: dict, fname, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for name, h in hists.items():
        e = range(1, len(h["train_loss"]) + 1)
        l, = ax[0].plot(e, h["train_loss"], label=f"{name} train")
        ax[0].plot(e, h["val_loss"], "--", color=l.get_color(), label=f"{name} val")
        ax[1].plot(e, h["val_acc"], color=l.get_color(), label=name)
    ax[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss (solid=train, dashed=val)")
    ax[1].set(xlabel="Epoch", ylabel="Validation accuracy", title="Validation accuracy")
    for a in ax: a.legend(fontsize=7); a.grid(alpha=.3)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(RESULTS / fname, dpi=130); plt.close(fig)


def grad_norms(hists: dict, fname, epoch=0):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, h in hists.items():
        if h["grad_norms"]:
            g = h["grad_norms"][min(epoch, len(h["grad_norms"]) - 1)]
            ax.semilogy(range(1, len(g) + 1), g, "o-", label=name)
    ax.set(xlabel="Layer index (input -> output)", ylabel="Mean ||dL/dW||_F (log)",
           title=f"Per-layer gradient norms, epoch {epoch+1}")
    ax.legend(); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(RESULTS / fname, dpi=130); plt.close(fig)


def confusion(cm, classes, fname, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    fig.tight_layout(); fig.savefig(RESULTS / fname, dpi=130); plt.close(fig)
