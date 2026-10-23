"""Part 1 + 2: feed-forward classifier, training monitoring and failure diagnosis.

python -m src.run_mlp            # full suite (baseline + 5 controlled failure runs)
python -m src.run_mlp --quick    # 3 epochs each, smoke test
"""
import argparse
import torch
from sklearn.metrics import confusion_matrix, classification_report
from .common import seed_everything, device, save_json, RESULTS
from .data import loaders
from .mlp import ArtMLP, count_params
from .train import fit, evaluate
from . import plots

DEEP = (512, 256, 256, 128, 128, 64, 64, 64)
# Each failure run changes ONE factor vs the baseline (controlled experiment, same seed).
CONFIGS = {
    "baseline":          dict(model=dict(activation="relu", init="he", batchnorm=True, dropout=0.3),
                              train=dict(lr=1e-3, weight_decay=1e-4)),
    "lr_too_high":       dict(model=dict(activation="relu", init="he", batchnorm=True, dropout=0.3),
                              train=dict(lr=1.0, weight_decay=1e-4, opt="sgd", clip=None)),
    "deep_sigmoid":      dict(model=dict(activation="sigmoid", init="xavier", batchnorm=False,
                                         dropout=0.0, hidden=DEEP),
                              train=dict(lr=1e-2, opt="sgd", weight_decay=0)),
    "deep_relu_he":      dict(model=dict(activation="relu", init="he", batchnorm=False,
                                         dropout=0.0, hidden=DEEP),
                              train=dict(lr=1e-2, opt="sgd", weight_decay=0)),
    "no_regularisation": dict(model=dict(activation="relu", init="he", batchnorm=False, dropout=0.0),
                              train=dict(lr=1e-3, weight_decay=0, patience=999)),
    "bad_init_large":    dict(model=dict(activation="relu", init="large", batchnorm=False, dropout=0.0),
                              train=dict(lr=1e-3, weight_decay=0, clip=None)),
}


def main(quick=False):
    seed_everything(); dev = device()
    ld, classes = loaders("mlp", batch_size=64)
    results, hists = {}, {}
    for name, cfg in CONFIGS.items():
        seed_everything()
        m = ArtMLP(n_classes=len(classes), **cfg["model"])
        tr = dict(epochs=3 if quick else 40, patience=8, log_name=name); tr.update(cfg["train"])
        if quick: tr["patience"] = 99
        m, h = fit(m, ld, dev, **tr)
        tl, ta = (evaluate(m, ld["test"], dev, torch.nn.CrossEntropyLoss())
                  if not h["nan"] else (float("nan"), float("nan")))
        hists[name] = h
        results[name] = dict(params=count_params(m), best_val_acc=max(h["val_acc"], default=0),
                             best_epoch=h["best_epoch"], test_acc=ta, test_loss=tl,
                             diverged=bool(h["nan"]),
                             first_layer_grad_epoch1=h["grad_norms"][0][0] if h["grad_norms"] else None,
                             last_layer_grad_epoch1=h["grad_norms"][0][-1] if h["grad_norms"] else None,
                             dead_relu_last_epoch=h["dead_relu"][-1] if h["dead_relu"] else None)
        if name == "baseline":
            torch.save(m.state_dict(), RESULTS / "mlp_baseline.pt")
            ys, ps = [], []; m.eval()
            with torch.no_grad():
                for x, y in ld["test"]:
                    ps += m(x.to(dev)).argmax(1).cpu().tolist(); ys += y.tolist()
            plots.confusion(confusion_matrix(ys, ps), classes, "mlp_confusion.png", "MLP - test set")
            results[name]["report"] = classification_report(ys, ps, target_names=classes, output_dict=True)
    save_json(results, "mlp_results.json")
    save_json(hists, "mlp_histories.json")
    plots.learning_curves({k: hists[k] for k in ["baseline", "no_regularisation"]},
                          "mlp_overfitting.png", "Regularisation (dropout+BN+weight decay) vs none")
    plots.learning_curves({k: hists[k] for k in ["baseline", "lr_too_high", "bad_init_large"]},
                          "mlp_instability.png", "Exploding loss: LR too high / bad init")
    plots.grad_norms({k: hists[k] for k in ["deep_sigmoid", "deep_relu_he"]},
                     "mlp_vanishing_gradients.png")
    for k, v in results.items():
        print(f"{k:18s} val={v['best_val_acc']:.3f} test={v['test_acc']:.3f} diverged={v['diverged']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
