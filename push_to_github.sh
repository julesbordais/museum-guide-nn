#!/usr/bin/env bash
# Creates a readable step-by-step commit history, then pushes.
# Usage: ./push_to_github.sh https://github.com/<you>/museum-guide-nn.git
set -e
REMOTE="$1"; [ -z "$REMOTE" ] && { echo "usage: $0 <remote-url>"; exit 1; }
git init -b main
c(){ git add "${@:2}"; git commit -m "$1"; }
c "init: repo structure, requirements, gitignore" .gitignore requirements.txt src/__init__.py src/common.py
c "data: WikiArt streaming download and 70/15/15 split" src/data.py
c "mlp: ArtMLP with configurable activation/init/BN/dropout" src/mlp.py
c "train: warmup-cosine, clipping, early stopping, grad-norm and dead-ReLU monitors" src/train.py src/plots.py
c "exp: baseline + 5 controlled failure-diagnosis runs" src/run_mlp.py
c "cnn: ResNet-18 linear probe and layer4 fine-tune" src/run_resnet.py
c "qa: MiniLM sentence embeddings vs TF-IDF retrieval" src/run_qa.py
c "tradeoff: classical vs neural benchmark" src/run_tradeoff.py src/make_report_tables.py
c "pipeline: run_all script and Colab runner" run_all.sh notebooks/
c "results: metrics JSON and figures from Colab T4 run" results/
c "docs: README, report and defence plan" README.md REPORT.md DEFENCE.md push_to_github.sh
[ -f museum_guide_nn_defence.pptx ] && c "slides: 10-slide defence deck" museum_guide_nn_defence.pptx
git remote add origin "$REMOTE"
git push -u origin main
