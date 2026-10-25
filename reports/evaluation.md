# Offline evaluation

Catalog: 11810 titles. Relevance proxy: genre Jaccard >= 0.5 (genres are not model features). k = 10, seed = 42.

## Overall (random sample of query movies)

| Strategy | Queries | Precision@10 | HitRate@10 | NDCG@10 | Coverage | Same-industry | Hollywood share of recs |
|---|---|---|---|---|---|---|---|
| random | 1000 | 0.111 | 0.593 | 0.165 | 0.576 | 0.719 | 0.848 |
| most popular | 1000 | 0.095 | 0.545 | 0.140 | 0.001 | 0.836 | 1.000 |
| TF-IDF cosine (no boosts) | 1000 | 0.288 | 0.861 | 0.317 | 0.457 | 0.861 | 0.829 |
| TF-IDF + /recommend boosts | 1000 | 0.345 | 0.885 | 0.365 | 0.472 | 0.900 | 0.842 |
| TF-IDF + movie-page boosts | 1000 | 0.461 | 0.925 | 0.449 | 0.462 | 0.951 | 0.837 |
| TF-IDF + movie-page boosts, no mood | 1000 | 0.286 | 0.857 | 0.316 | 0.459 | 0.959 | 0.840 |

Hollywood is 84.5% of the catalog.

Caveat: mood labels in the dataset are closely tied to genres, so boosts that use mood are partly rewarded for matching the answer key. The 'no mood' row is the non-circular comparison.

## Bias check: quality per query industry

| Industry | Titles | Queries | P@10 no boosts | P@10 movie-page boosts | Same-industry share (no boosts) | Same-industry share (boosts) |
|---|---|---|---|---|---|---|
| ANIME | 979 | 150 | 0.427 | 0.619 | 0.641 | 0.879 |
| BOLLYWOOD | 66 | 66 | 0.309 | 0.420 | 0.554 | 0.770 |
| CHINESE | 344 | 150 | 0.269 | 0.424 | 0.453 | 0.772 |
| HOLLYWOOD | 9980 | 150 | 0.278 | 0.461 | 0.912 | 0.971 |
| KOREAN | 441 | 150 | 0.281 | 0.449 | 0.404 | 0.637 |
