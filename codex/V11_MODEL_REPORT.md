# V11 Tri-Expert Fusion

## What It Is

`v11_triexpert_fusion.py` implements a patient-aware fusion model for pediatric multi-lead ECG classification. It reuses the exact honest outer-fold protocol from `pediatric_ecg_V9+XAI.ipynb` and combines three complementary experts:

1. `V8` residual dual-branch waveform expert
2. `V9` wider residual dual-branch waveform expert with TTA
3. `XGBoost` physiology expert trained on the 174 handcrafted beat features

The fusion layer is selected on a record-level calibration split inside each outer fold:

- `meta_logreg`: multinomial logistic regression on expert probabilities and confidence descriptors
- `weighted_grid`: validation-selected weighted averaging across `V8`, `V9`, and `XGBoost`

The best calibration rule is then evaluated on the untouched outer test fold.

## Novelty

This version is not just a plain checkpoint average. Its novelty is the **patient-aware tri-expert late fusion** of:

- deep morphology-rhythm waveform encoders
- handcrafted physiological beat descriptors
- calibration-time expert selection per fold

This gives us a cleaner publication angle than "just make V9 wider again", while keeping the setup reproducible and XAI-friendly.

## Honest 5-Fold Results

Stored in:

- [summary_all5.json](/C:/Users/Admin/Desktop/Projects/8th%20Sem/Codex/outputs/v11_triexpert_fusion/summary_all5.json)
- [summary_all5.csv](/C:/Users/Admin/Desktop/Projects/8th%20Sem/Codex/outputs/v11_triexpert_fusion/summary_all5.csv)

### Mean Metrics

| Model | Macro F1 | F1 Std | Accuracy | AUROC |
|---|---:|---:|---:|---:|
| V11 fusion | 0.8818 | 0.0074 | 0.8837 | 0.9634 |
| V9 expert | 0.8747 | 0.0072 | 0.8769 | 0.9622 |
| V8 expert | 0.8714 | 0.0045 | 0.8737 | 0.9589 |
| XGBoost expert | 0.6221 | 0.0086 | 0.6260 | 0.8028 |

### Gain Over V9

- `Delta Macro F1 = +0.0070`
- `Delta Accuracy = +0.0068`
- `Delta AUROC = +0.0012`

## Per-Fold Fusion Choices

- Fold 1: `meta_logreg`
- Fold 2: `meta_logreg`
- Fold 3: `weighted_grid`
- Fold 4: `weighted_grid`
- Fold 5: `weighted_grid`

## Main Files

- [v11_triexpert_fusion.py](/C:/Users/Admin/Desktop/Projects/8th%20Sem/Codex/v11_triexpert_fusion.py)
- [fold_1/fold_summary.json](/C:/Users/Admin/Desktop/Projects/8th%20Sem/Codex/outputs/v11_triexpert_fusion/fold_1/fold_summary.json)
- [fold_2/fold_summary.json](/C:/Users/Admin/Desktop/Projects/8th%20Sem/Codex/outputs/v11_triexpert_fusion/fold_2/fold_summary.json)
- [fold_3/fold_summary.json](/C:/Users/Admin/Desktop/Projects/8th%20Sem/Codex/outputs/v11_triexpert_fusion/fold_3/fold_summary.json)
- [fold_4/fold_summary.json](/C:/Users/Admin/Desktop/Projects/8th%20Sem/Codex/outputs/v11_triexpert_fusion/fold_4/fold_summary.json)
- [fold_5/fold_summary.json](/C:/Users/Admin/Desktop/Projects/8th%20Sem/Codex/outputs/v11_triexpert_fusion/fold_5/fold_summary.json)

## Rerun

```powershell
python "C:\Users\Admin\Desktop\Projects\8th Sem\Codex\v11_triexpert_fusion.py" --folds 1,2,3,4,5 --batch-size 512 --v9-tta 5
```

## XAI Direction

This model is a good XAI candidate because we can explain it at three levels:

1. waveform-level attribution on the V8/V9 experts
2. handcrafted-feature importance via XGBoost
3. fusion-level expert reliance via meta weights or selected expert probabilities
