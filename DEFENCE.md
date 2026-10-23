# Oral defence — 10 min + 5 min Q&A (slides: `museum_guide_nn_defence.pptx`)

| Time | Slide | Message |
|---|---|---|
| 0:00 | 1 Title | Museum guide: recognise style + answer visitor questions |
| 0:30 | 2 Problem & data | WikiArt 6×300, 70/15/15, val for selection, test once |
| 1:30 | 3 MLP | 6.5 M params, 42.7 % — Baroque "sink" |
| 2:45 | 4 Overfitting | 100 % train vs val loss 4.0 → regularisation |
| 3:45 | 5 Vanishing gradients | 3 000× gap, sigmoid vs ReLU+He |
| 4:45 | 6 Instability | LR 1.0 → 98 % dead ReLUs; N(0,1) init → NaN |
| 5:30 | 7 ResNet | 76 → 89 %, why convolution + residuals |
| 6:45 | 8 Q→artwork | MiniLM vs TF-IDF, hybrid 95.8 % |
| 8:00 | 9 Trade-off | table + recommendation |
| 9:15 | 10 Takeaways | limits + next steps |

## Likely questions
1. **Why cross-entropy?** Softmax = categorical distribution; minimising CE = maximising log-likelihood. Binary + sigmoid = logistic-regression log-likelihood.
2. **`loss.backward()` vs `optimizer.step()`?** Backward computes ∂L/∂θ by reverse-mode autodiff; step applies the update.
3. **Why He?** ReLU zeroes ≈ half a symmetric signal → E[a²] ≈ ½E[z²] → Var(W) = 2/fan_in.
4. **BatchNorm vs LayerNorm?** BN: per feature across the batch (CNNs, train/eval differ). LN: per example across features (batch-independent → transformers).
5. **Why did deep sigmoid fail?** σ′ ≤ 0.25 per layer; measured 3e-4 vs 0.9 → layer 1 barely learns; test = chance.
6. **Why is the MLP worse than HOG+SVM?** No locality / weight sharing; HOG encodes edge orientation that the MLP must learn from 1 260 images.
7. **Why residual connections?** ∂(x+F(x))/∂x = I + ∂F/∂x — identity path keeps the gradient alive.
8. **Receptive field?** r_l = r_{l−1} + (k_l − 1)·∏ strides of previous layers.
9. **Why `inf` in some gradient norms?** AMP GradScaler's first steps overflow; it skips them and rescales — artefact, not divergence.
10. **Is the ranking reliable?** One seed, one budget → conditional evidence; would rerun 3 seeds.
11. **Why recommend embeddings + LogReg over fine-tuning?** Same accuracy (89.6 vs 89.2), head retrains in 0.05 s, simpler ops.
