# Intelligent Museum Guide — Neural Network Components
*Jules Bordais · MSA-DATB09-03 · all numbers from `results/` (single seed 42, Colab T4)*

## 1. Problem & data
- **Task:** recognise the style of an artwork photo (Impressionism, Baroque, Cubism, Ukiyo-e, Pop Art, High Renaissance) so the guide can contextualise it, and match visitor questions to artworks.
- **Data:** WikiArt (`huggan/wikiart`), 300 images per style = 1 800 images, random 70/15/15 split per image → ≈1 259 train / 281 val / 260 test. Test support per class ranges 30–57, so we report macro-F1 too.
- **Metrics:** accuracy, macro-F1, confusion matrix; Recall@k and MRR for retrieval.

## 2. Feed-forward network
- 12 288 (3×64×64) → 512 → 256 → 128 → 6; Linear → BatchNorm → ReLU → Dropout 0.3; He init; **6 458 758 parameters**.
- AdamW (lr 1e-3, wd 1e-4), 2-epoch warmup + cosine, gradient clipping 1.0, early stopping on val loss (patience 8).
- **Result:** best val acc 48.4 % (epoch 7, stopped at 15), **test 42.7 %**, macro-F1 0.408 (chance 16.7 %).
- **Error analysis:** Baroque acts as a "dark-painting sink": recall 0.70 but precision 0.28 — 18/35 Renaissance and 15/44 Ukiyo-e images are predicted Baroque. Flattening destroys spatial structure, so the MLP mostly relies on colour/brightness statistics.

## 3. Training monitoring & failure diagnosis
Logged every epoch: train/val loss & accuracy, LR, per-layer ‖∂L/∂W‖_F, dead-ReLU fraction, non-finite loss. Each failure run changes **one factor** vs the baseline.

| Pathology | Run | Observed evidence | Mechanism | Fix |
|---|---|---|---|---|
| Overfitting | `no_regularisation` (no BN, dropout, wd; no early stop) | train acc → 100 %, train loss 0.005 while val loss climbs 2.3 → 4.0; test 31.9 % vs 42.7 % | 6.5 M parameters memorise ~1 260 images | dropout + BN + weight decay + early stopping |
| Exploding / unstable loss | `lr_too_high` (SGD lr = 1.0, no clipping) | epoch-1 val loss 6.37; val acc falls to 14 % (below chance); **98 % of last-layer ReLUs dead** | steps overshoot; large updates push pre-activations negative for good | lower LR, warmup, clipping |
| Bad initialisation | `bad_init_large` (N(0,1) weights) | loss non-finite in the first epoch → run aborted | Var(z) ≈ fan_in·Var(W) = 12 288 → activations/logits overflow | He (2/fan_in) or Xavier init |
| Vanishing gradients | `deep_sigmoid` (8 hidden layers, Xavier) | ‖dW₁‖ = 3.0e-4 vs ‖dW₉‖ = 0.90 (**~3 000×** smaller); test 15 % = chance | σ′ ≤ 0.25 multiplies across layers | `deep_relu_he`: ‖dW₁‖ = 17.2, flat profile, test 37.7 % |
| Dead ReLUs | monitor | baseline 0 % dead; deep ReLU (no BN) up to 25–31 % in late layers; LR = 1.0 → 98 % | units stuck at z < 0 get zero gradient | He init, BN, moderate LR |

Figures: `mlp_overfitting.png`, `mlp_instability.png`, `mlp_vanishing_gradients.png`.
*Note:* some epoch-1 last-layer norms are `inf` — the AMP GradScaler produces inf gradients on its first steps, skips those updates and lowers its scale. It is a measurement artefact, not a divergence.

## 4. Pretrained CNN — ResNet-18
- **Why a CNN:** local, weight-shared, translation-equivariant filters fit grid data; depth + stride grow the receptive field; residual connections y = x + F(x) give gradients an identity path (∂y/∂x = I + ∂F/∂x).
- **Protocol:** (A) freeze the ImageNet backbone, train the new fc head only (3 078 params); (B) unfreeze layer4, lr 1e-4. Augmentation: random-resized crop + flip.
- **Result:** linear probe **76.2 %** → fine-tuned **89.2 %** test (macro-F1 0.891), vs 42.7 % for the MLP. Fine-tuning reaches ~0.90 val acc by epoch 3; train loss keeps falling (0.03) while val loss plateaus ≈ 0.3, so longer training would only overfit.
- **Remaining errors:** Cubism→Impressionism (4), Pop Art→Ukiyo-e (4) — both share flat colour areas / fragmented brushwork.

## 5. Question → artwork similarity
- Catalogue text per artwork (genre, artist, style description); 2 visitor questions per (artist, style) written in visitor vocabulary.
- Neural: `all-MiniLM-L6-v2` (384-d, cosine). Classical: TF-IDF 1-2-grams (cosine).

| Method | R@1 | R@5 | MRR | ms / query |
|---|---|---|---|---|
| TF-IDF | 0.854 | **0.979** | 0.897 | 0.11 |
| MiniLM | 0.938 | 0.958 | 0.951 | 0.52 |
| Hybrid (0.5 / 0.5) | **0.958** | **0.979** | **0.969** | ≈0.6 |

**Reading:** embeddings understand paraphrases ("faces broken into shapes" ≈ Cubism) → +8.4 pts R@1; TF-IDF is strong on exact artist names → better R@5. The hybrid gets the best of both. **Limitation:** questions are synthetic and contain the artist's name; real visitor logs are needed.

## 6. Neural vs classical — production trade-off
| Model | Test acc | Macro-F1 | CPU ms / img | Size MB |
|---|---|---|---|---|
| HOG + colour → LogReg | 0.481 | 0.485 | 0.97 | 0.13 |
| HOG + colour → RBF-SVM | 0.538 | 0.545 | 2.32 | 18.2 |
| MLP (ours) | 0.427 | 0.408 | 1.82 | 25.8 |
| ResNet-18 fine-tuned | 0.892 | 0.891 | 60.0 | 44.8 |
| **ResNet-18 embeddings → LogReg** | **0.896** | **0.894** | ≈60 (backbone) | ≈44.8 |

- The MLP is **beaten by hand-crafted features** (HOG+SVM, +11 pts): without a spatial inductive bias, 6.5 M parameters on 1 260 images learn less than texture/colour descriptors.
- Pretrained features are the step change (+35 pts over the best classical model).
- **Recommendation:** frozen ResNet-18 backbone + logistic-regression head. Same accuracy as full fine-tuning, the head retrains in **0.05 s** when the museum adds a room, and 60 ms/image on CPU is fine for a kiosk (no GPU at inference). Q&A: hybrid TF-IDF + MiniLM. Keep HOG + LogReg (0.13 MB, <1 ms) as an offline fallback.

## 7. Limitations & next steps
Single seed → rerun 3 seeds and report mean ± std. Style labels are fuzzy (Renaissance vs Baroque). WikiArt scans ≠ visitor phone photos (glare, angle) → collect an in-gallery test set. Synthetic questions → log real ones (GDPR: no images or personal data stored).
