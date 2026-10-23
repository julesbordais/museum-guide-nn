### MLP runs
| Run | Params | Best val acc | Test acc | Best epoch | Diverged | ||dW|| layer1 / last (ep1) |
|---|---|---|---|---|---|---|
| baseline | 6,458,758 | 0.484 | 0.427 | 6 | False | 1.15e+01 / 1.48e+00 |
| lr_too_high | 6,458,758 | 0.331 | 0.215 | 4 | False | 2.97e+00 / 1.35e+00 |
| deep_sigmoid | 6,555,462 | 0.174 | 0.150 | 9 | False | 2.97e-04 / 8.96e-01 |
| deep_relu_he | 6,555,462 | 0.445 | 0.377 | 3 | False | 1.72e+01 / inf |
| no_regularisation | 6,456,966 | 0.459 | 0.319 | 2 | False | 1.65e+01 / inf |
| bad_init_large | 6,456,966 | 0.000 | nan | -1 | True | - |

### ResNet-18
Linear probe test acc: **0.762** — fine-tuned (layer4+fc): **0.892**

### Question → artwork retrieval
| Method | R@1 | R@5 | MRR | Top-1 style acc | ms/query |
|---|---|---|---|---|---|
| tfidf | 0.854 | 0.979 | 0.897 | 0.854 | 0.11 |
| minilm | 0.938 | 0.958 | 0.951 | 0.938 | 0.52 |
| hybrid_0.5 | 0.958 | 0.979 | 0.969 | nan | nan |

### Neural vs classical
| Model | Test acc | Macro-F1 | CPU ms/img | Size (MB) |
|---|---|---|---|---|
| hog_logreg | 0.481 | 0.485 | 0.97 | 0.13 |
| hog_svm_rbf | 0.538 | 0.545 | 2.32 | 18.22 |
| mlp | 0.427 | 0.408 | 1.82 | 25.85 |
| resnet18_finetuned | 0.892 | 0.891 | 59.96 | 44.80 |
| resnet_emb_logreg | 0.896 | 0.894 | nan | nan |