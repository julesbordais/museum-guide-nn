# Museum Intelligent Guide — Neural Network components
**MSA-DATB09-03 · Neural Networks & Deep Learning Foundations · Mutualised project (CC2)**
Author: Jules Bordais

A French museum's AI team is building an intelligent guide. This repo delivers its neural-network building blocks:

| # | Component | Headline result (test set) |
|---|---|---|
| 1 | Feed-forward network (PyTorch) for artwork style classification | 42.7 % accuracy (6 classes, chance 16.7 %) |
| 2 | Training monitoring + failure diagnosis (5 controlled failures) | vanishing gradients: layer-1 gradient ~3 000× smaller than last layer |
| 3 | Pretrained CNN (ResNet-18) | 76.2 % linear probe → **89.2 %** fine-tuned |
| 4 | Sentence embeddings, question → artwork | MiniLM R@1 93.8 % vs TF-IDF 85.4 %; hybrid **95.8 %** |
| 5 | Neural vs classical production trade-off | ResNet embeddings + LogReg: **89.6 %**, retrains in 0.05 s |

Full write-up: [`REPORT.md`](REPORT.md) · Oral plan: [`DEFENCE.md`](DEFENCE.md) · Figures & JSON: [`results/`](results/)

## Reproduce
```bash
pip install -r requirements.txt
./run_all.sh            # ~20 min on a Colab T4 GPU (notebooks/colab_runner.ipynb)
```
| Step | Command | Outputs (`results/`) |
|---|---|---|
| Data — WikiArt, 6 styles × 300 | `python -m src.data --per_class 300` | `data/wikiart/` (git-ignored) |
| MLP + diagnosis | `python -m src.run_mlp` | `mlp_*.json`, `mlp_*.png` |
| ResNet-18 | `python -m src.run_resnet` | `resnet_*.json/png` |
| Question → artwork | `python -m src.run_qa` | `qa_results.json` |
| Trade-off | `python -m src.run_tradeoff` | `tradeoff_results.json` |
| Tables | `python -m src.make_report_tables` | `tables.md` |

## Layout
```
src/common.py         seeding, device, paths
src/data.py           WikiArt streaming download (decode only kept images), 70/15/15 split, transforms
src/mlp.py            ArtMLP — configurable activation / init (He, Xavier) / BatchNorm / dropout
src/train.py          AdamW, warmup+cosine LR, grad clipping, early stopping, best-val checkpoint,
                      AMP, per-layer grad norms, dead-ReLU monitor
src/run_mlp.py        baseline + 5 controlled failure runs (one factor changed, same seed)
src/run_resnet.py     ResNet-18 linear probe → fine-tune layer4, exports embeddings
src/run_qa.py         MiniLM vs TF-IDF vs hybrid retrieval (R@1, R@5, MRR)
src/run_tradeoff.py   HOG/colour + LogReg/SVM vs MLP vs ResNet: accuracy, latency, size
notebooks/            Colab runner
```

## Protocol
- Loss: cross-entropy on raw logits (= maximum likelihood of a categorical model).
- Model selection on validation only; the test set is evaluated once per model.
- Single seed (42): rankings are evidence under this setup, not universal.
