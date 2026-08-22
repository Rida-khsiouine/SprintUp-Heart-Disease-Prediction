# Data provenance

This directory contains exact snapshots of the four processed cohorts from the
[UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart%2Bdisease):
Cleveland, Hungary, Switzerland, and VA Long Beach.

The immutable source files live in `raw/`. `manifest.json` records their official
URLs, retrieval date, byte sizes, row counts, and SHA-256 checksums. Run
`heart-disease validate-data` to verify that the local bytes still match the
manifest. The original target `num` is retained; modeling constructs the binary
target `disease_present = (num > 0)` without modifying the raw files.

The data is licensed under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart
Disease* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C52P4X
