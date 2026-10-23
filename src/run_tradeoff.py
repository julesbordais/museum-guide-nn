"""Part 5: neural vs classical - production trade-off (run after run_mlp and run_resnet).

HOG+colour -> LogReg / RBF-SVM  vs  MLP  vs  ResNet-18 fine-tuned  vs  ResNet embeddings -> LogReg.
Measures: test accuracy, macro-F1, training time, CPU latency per image, model size.
"""
import io, pickle, time
import numpy as np, torch
from PIL import Image
from torchvision import datasets
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from .common import RESULTS, save_json, seed_everything
from .data import ROOT, tf_mlp, tf_resnet
from .mlp import ArtMLP
from .run_resnet import build as build_resnet


def feats(img):
    g = np.asarray(img.convert("L").resize((64, 64)), dtype=np.float32) / 255
    h = hog(g, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    hsv = np.asarray(img.convert("HSV").resize((64, 64)))
    ch = np.concatenate([np.histogram(hsv[..., c], bins=16, range=(0, 255), density=True)[0] for c in range(3)])
    return np.concatenate([h, ch])


def load(split):
    ds = datasets.ImageFolder(ROOT / split)
    return ds, np.stack([feats(Image.open(p)) for p, _ in ds.samples]), np.array(ds.targets)


def size_mb(obj):
    if isinstance(obj, torch.nn.Module):
        b = io.BytesIO(); torch.save(obj.state_dict(), b); return b.tell() / 1e6
    return len(pickle.dumps(obj)) / 1e6


def latency_ms(fn, x, n=50):
    fn(x); t = time.perf_counter()
    for _ in range(n): fn(x)
    return (time.perf_counter() - t) / n * 1e3


def main():
    seed_everything(); torch.set_num_threads(4); res = {}
    ds_tr, Xtr, ytr = load("train"); ds_te, Xte, yte = load("test")
    classes = ds_tr.classes
    for name, clf in [("hog_logreg", LogisticRegression(max_iter=3000, C=0.5)),
                      ("hog_svm_rbf", SVC(C=5, gamma="scale"))]:
        pipe = make_pipeline(StandardScaler(), clf)
        t = time.time(); pipe.fit(Xtr, ytr); tt = time.time() - t
        p = pipe.predict(Xte)
        res[name] = dict(test_acc=accuracy_score(yte, p), macro_f1=f1_score(yte, p, average="macro"),
                         train_s=tt, cpu_ms_per_image=latency_ms(pipe.predict, Xte[:1]),
                         size_mb=size_mb(pipe), gpu_needed=False)
    imgs = [Image.open(p).convert("RGB") for p, _ in ds_te.samples]
    for name, model, tf, ck in [
        ("mlp", ArtMLP(n_classes=len(classes)), tf_mlp(), "mlp_baseline.pt"),
        ("resnet18_finetuned", build_resnet(len(classes)), tf_resnet(False), "resnet18_finetuned.pt")]:
        model.load_state_dict(torch.load(RESULTS / ck, map_location="cpu")); model.eval()
        with torch.no_grad():
            p = np.concatenate([model(torch.stack([tf(im) for im in imgs[i:i+64]])).argmax(1).numpy()
                                for i in range(0, len(imgs), 64)])
            lat = latency_ms(lambda x: model(x), tf(imgs[0]).unsqueeze(0))
        res[name] = dict(test_acc=accuracy_score(yte, p), macro_f1=f1_score(yte, p, average="macro"),
                         cpu_ms_per_image=lat, size_mb=size_mb(model),
                         params=sum(q.numel() for q in model.parameters()), gpu_needed="training only")
    tr, te = np.load(RESULTS / "resnet_emb_train.npz"), np.load(RESULTS / "resnet_emb_test.npz")
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    t = time.time(); pipe.fit(tr["X"], tr["Y"]); tt = time.time() - t
    p = pipe.predict(te["X"])
    res["resnet_emb_logreg"] = dict(test_acc=accuracy_score(te["Y"], p),
                                    macro_f1=f1_score(te["Y"], p, average="macro"), train_s=tt,
                                    note="latency ~= ResNet backbone + negligible head")
    save_json(res, "tradeoff_results.json")


if __name__ == "__main__":
    main()
