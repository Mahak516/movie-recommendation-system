# Mood classifier: handling class imbalance

Stratified split: 8857 train / 2953 test. Features: TF-IDF over keywords and cast.

## Class counts (full dataset)

| Mood | Titles |
|---|---|
| Adrenaline | 2925 |
| Feel-Good | 2456 |
| Dark | 2021 |
| Epic/Myth | 1998 |
| Romantic | 1988 |
| Neutral | 411 |
| Futuristic | 11 |

## Overall

| Config | Accuracy | Macro-F1 |
|---|---|---|
| unweighted | 0.535 | 0.395 |
| class_weight=balanced | 0.541 | 0.427 |

## Per-class recall (the minority classes are the point)

| Mood | Test support | Recall unweighted | Recall balanced | F1 unweighted | F1 balanced |
|---|---|---|---|---|---|
| Adrenaline | 731 | 0.67 | 0.59 | 0.61 | 0.60 |
| Feel-Good | 614 | 0.59 | 0.55 | 0.51 | 0.51 |
| Dark | 505 | 0.51 | 0.57 | 0.55 | 0.56 |
| Epic/Myth | 500 | 0.45 | 0.51 | 0.52 | 0.54 |
| Romantic | 497 | 0.47 | 0.53 | 0.50 | 0.51 |
| Neutral | 103 | 0.04 | 0.18 | 0.07 | 0.26 |
| Futuristic | 3 | 0.00 | 0.00 | 0.00 | 0.00 |

Note: classes with only a handful of test examples (e.g. 'Futuristic') give very noisy estimates; treat their rows as indicative, not conclusive.
