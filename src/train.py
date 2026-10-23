"""Generic training loop with production discipline + monitoring.

Monitors every epoch: train/val loss & accuracy, per-layer gradient Frobenius norms,
fraction of dead ReLU units, learning rate. Implements: warmup + cosine LR schedule,
gradient clipping, early stopping on val loss, best-checkpoint selection (val only,
test touched once at the end), optional mixed precision on CUDA.
"""
import math, time, copy
import torch
import torch.nn as nn


def lr_lambda(warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    return f


@torch.no_grad()
def evaluate(model, loader, dev, loss_fn):
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        out = model(x)
        loss_sum += loss_fn(out, y).item() * len(y)
        correct += (out.argmax(1) == y).sum().item(); tot += len(y)
    return loss_sum / tot, correct / tot


def dead_relu_fraction(model, x):
    """Share of ReLU units that output 0 for EVERY example of the probe batch."""
    fracs, hooks = [], []
    def hook(_, __, out):
        fracs.append(((out <= 0).all(0)).float().mean().item())
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            hooks.append(m.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(x)
    for h in hooks: h.remove()
    return fracs


def fit(model, loaders, dev, epochs=30, lr=1e-3, weight_decay=1e-4, opt="adamw",
        warmup_epochs=2, clip=1.0, patience=6, amp=True, log_name="run"):
    model.to(dev)
    loss_fn = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optim = (torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay) if opt == "adamw"
             else torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay))
    steps = epochs * len(loaders["train"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda(warmup_epochs * len(loaders["train"]), steps))
    use_amp = amp and dev.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Conv2d))]

    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "lr",
                            "grad_norms", "dead_relu", "nan"]}
    best, best_state, bad = float("inf"), None, 0
    probe = next(iter(loaders["val"]))[0][:256].to(dev)
    for ep in range(epochs):
        model.train(); t0 = time.time()
        tot, correct, ls, gn_acc, nb = 0, 0, 0.0, None, 0
        for x, y in loaders["train"]:
            x, y = x.to(dev), y.to(dev)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                out = model(x); loss = loss_fn(out, y)
            if not torch.isfinite(loss):
                hist["nan"].append(ep); print(f"!! non-finite loss at epoch {ep}"); break
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            # note: with AMP the GradScaler may produce inf grads on its first steps
            # (it then skips the step and lowers the scale) -> inf can appear in epoch-1 norms
            g = [m.weight.grad.norm().item() if m.weight.grad is not None else 0.0
                 for _, m in linears]
            gn_acc = g if gn_acc is None else [a + b for a, b in zip(gn_acc, g)]; nb += 1
            if clip:
                nn.utils.clip_grad_norm_(params, clip)
            scaler.step(optim); scaler.update(); sched.step()
            ls += loss.item() * len(y); correct += (out.argmax(1) == y).sum().item(); tot += len(y)
        if hist["nan"]:
            break
        vl, va = evaluate(model, loaders["val"], dev, loss_fn)
        hist["train_loss"].append(ls / tot); hist["train_acc"].append(correct / tot)
        hist["val_loss"].append(vl); hist["val_acc"].append(va)
        hist["lr"].append(sched.get_last_lr()[0])
        hist["grad_norms"].append([v / max(nb, 1) for v in gn_acc])
        hist["dead_relu"].append(dead_relu_fraction(model, probe))
        print(f"[{log_name}] ep{ep:02d} train {ls/tot:.3f}/{correct/tot:.3f} "
              f"val {vl:.3f}/{va:.3f} ({time.time()-t0:.0f}s)")
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {ep}"); break
    if best_state is not None:
        model.load_state_dict(best_state)
    hist["best_epoch"] = int(min(range(len(hist["val_loss"])), key=hist["val_loss"].__getitem__)) \
        if hist["val_loss"] else -1
    hist["layer_names"] = [n for n, _ in linears]
    return model, hist
