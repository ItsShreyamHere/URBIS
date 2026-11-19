#!/bin/sh
# URBIS installer: sh install.sh   (needs git + Python 3.11+; ~4 GB disk; runs on your own computer)
set -e
REPO="${URBIS_REPO:-https://github.com/ItsShreyamHere/URBIS.git}"      # TODO owner: set to the real public repo
REL="${URBIS_RELEASE:-https://github.com/ItsShreyamHere/URBIS/releases/download/weights-v1}"  # TODO owner: upload the 3 .pt files here
git clone --depth 1 "$REPO" urbis && cd urbis
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q numpy scipy scikit-learn scikit-image lightgbm rasterio pyogrio pyproj shapely duckdb requests pillow matplotlib torch timm
mkdir -p runs/kaggle_retrain3/data runs/kaggle_retrain3/out runs/kaggle_outline/out1
curl -fL "$REL/visible_verifier_sat_ibge.pt" -o runs/kaggle_retrain3/data/visible_verifier_sat_ibge.pt
curl -fL "$REL/twoview_e2.pt" -o runs/kaggle_retrain3/out/twoview_e2.pt
curl -fL "$REL/mvit.pt" -o runs/kaggle_outline/out1/mvit.pt
echo "Done. Start with:  cd urbis && ./urbis.sh   then open http://localhost:8770"
