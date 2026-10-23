"""Turn results/*.json into Markdown tables (results/tables.md)."""
import json
from .common import RESULTS


def load(n):
    p = RESULTS / n
    return json.load(open(p)) if p.exists() else {}


def main():
    out = []
    m = load("mlp_results.json")
    if m:
        out += ["### MLP runs", "| Run | Params | Best val acc | Test acc | Best epoch | Diverged | grad L1 / last (ep1) |",
                "|---|---|---|---|---|---|---|"]
        for k, v in m.items():
            g1, gl = v.get("first_layer_grad_epoch1"), v.get("last_layer_grad_epoch1")
            gs = f"{g1:.2e} / {gl:.2e}" if g1 is not None else "-"
            out.append(f"| {k} | {v['params']:,} | {v['best_val_acc']:.3f} | {v['test_acc']:.3f} | "
                       f"{v['best_epoch']} | {v['diverged']} | {gs} |")
    r = load("resnet_results.json")
    if r:
        out += ["", "### ResNet-18", f"Linear probe test acc: **{r['linear_probe_test_acc']:.3f}** — "
                f"fine-tuned (layer4+fc): **{r['finetune_test_acc']:.3f}**"]
    q = load("qa_results.json")
    if q:
        out += ["", "### Question → artwork retrieval", "| Method | R@1 | R@5 | MRR |", "|---|---|---|---|"]
        for k in ["tfidf", "minilm", "hybrid_0.5"]:
            v = q[k]; out.append(f"| {k} | {v['recall_at_1']:.3f} | {v['recall_at_5']:.3f} | {v['mrr']:.3f} |")
    t = load("tradeoff_results.json")
    if t:
        out += ["", "### Neural vs classical", "| Model | Test acc | Macro-F1 | CPU ms/img | Size (MB) |", "|---|---|---|---|---|"]
        for k, v in t.items():
            out.append(f"| {k} | {v['test_acc']:.3f} | {v['macro_f1']:.3f} | "
                       f"{v.get('cpu_ms_per_image', float('nan')):.2f} | {v.get('size_mb', float('nan')):.2f} |")
    (RESULTS / "tables.md").write_text("\n".join(out)); print("\n".join(out))


if __name__ == "__main__":
    main()
