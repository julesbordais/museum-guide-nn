"""Part 3: pretrained CNN (ResNet-18, ImageNet weights) for artwork classification.

A. Linear probe - freeze the backbone, train only the new fc head.
B. Fine-tune    - unfreeze layer4 + fc, lr 1e-4.
Also exports 512-d penultimate embeddings (used by the trade-off study).
"""
import argparse
import numpy as np
import torch, torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import confusion_matrix
from .common import seed_everything, device, save_json, RESULTS
from .data import loaders
from .train import fit, evaluate
from . import plots


def build(n_classes):
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for p in m.parameters():
        p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


@torch.no_grad()
def embed(model, loader, dev):
    feat = nn.Sequential(*list(model.children())[:-1]).to(dev).eval()
    X, Y = [], []
    for x, y in loader:
        X.append(feat(x.to(dev)).flatten(1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(X), np.concatenate(Y)


def main(quick=False):
    seed_everything(); dev = device()
    ld, classes = loaders("resnet", batch_size=32)
    m = build(len(classes)); E = 2 if quick else 8
    m, hA = fit(m, ld, dev, epochs=E, lr=1e-3, warmup_epochs=1, patience=4, log_name="probe")
    _, accA = evaluate(m, ld["test"], dev, nn.CrossEntropyLoss())
    for p in m.layer4.parameters():
        p.requires_grad = True
    m, hB = fit(m, ld, dev, epochs=E, lr=1e-4, warmup_epochs=1, patience=4, log_name="finetune")
    _, accB = evaluate(m, ld["test"], dev, nn.CrossEntropyLoss())
    torch.save(m.state_dict(), RESULTS / "resnet18_finetuned.pt")
    ys, ps = [], []; m.eval()
    with torch.no_grad():
        for x, y in ld["test"]:
            ps += m(x.to(dev)).argmax(1).cpu().tolist(); ys += y.tolist()
    plots.confusion(confusion_matrix(ys, ps), classes, "resnet_confusion.png", "ResNet-18 fine-tuned - test")
    plots.learning_curves({"linear probe": hA, "fine-tune layer4": hB},
                          "resnet_curves.png", "ResNet-18 transfer learning")
    ld_eval, _ = loaders("resnet", batch_size=64)
    ld_eval["train"].dataset.transform = ld_eval["val"].dataset.transform
    for split in ["train", "test"]:
        dl = torch.utils.data.DataLoader(ld_eval[split].dataset, batch_size=64)
        X, Y = embed(m, dl, dev)
        np.savez(RESULTS / f"resnet_emb_{split}.npz", X=X, Y=Y)
    save_json(dict(linear_probe_test_acc=accA, finetune_test_acc=accB,
                   trainable_params_probe=sum(p.numel() for p in m.fc.parameters()),
                   total_params=sum(p.numel() for p in m.parameters())), "resnet_results.json")
    print(f"linear probe test acc={accA:.3f} | fine-tuned test acc={accB:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
