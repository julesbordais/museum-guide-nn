#!/usr/bin/env bash
# Full pipeline. ~20 min on a Colab T4 GPU.
set -e
python -m src.data --per_class 300
python -m src.run_mlp
python -m src.run_resnet
python -m src.run_qa
python -m src.run_tradeoff
python -m src.make_report_tables
