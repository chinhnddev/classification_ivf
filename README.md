# hv_embryo_scratch_newmodel

Scratch embryo good/poor classification for Hung Vuong Day3/Day5 images.
Single custom CNN, single training loop, single evaluation path.

## Dataset
CSV: `data/metadata/hv_day3_day5.csv`

Columns:
- image_path (relative path to image file)
- label (0=poor, 1=good)
- day (3 or 5)
- embryo_id (optional; ignored)
- optional: patient_id

Images can be RGB or grayscale.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Validate data
```bash
python -m hv.validate_data --config configs/base.yaml
```

## Train
```bash
python train.py --config configs/base.yaml --seed 0
```

Overfit sanity check (no aug, expect ~100% train acc):
```bash
python train.py --config configs/base.yaml --seed 0 --overfit_n 20
```

## Evaluate
```bash
python eval.py --ckpt outputs/best.ckpt
```

The evaluator:
- Computes best threshold on val
- Applies it to test for F1
- Saves `outputs/metrics.json` and `outputs/predictions.csv`

## Outputs
Each run writes:
- `outputs/config.yaml`
- `outputs/splits.csv`
- `outputs/best.ckpt`
- `outputs/logs/` (CSV + TensorBoard)
- `outputs/metrics.json`
- `outputs/predictions.csv`

## Notes
- Single model only (custom CNN).
- Image-level stratified split: 80/10/10.
- Day is never used as an input feature.
"# classification_ivf" 
