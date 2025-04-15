# Attempting to improve the [TUnA](https://github.com/Wang-lab-UCSD/TUnA) model

[TUnA](https://github.com/Wang-lab-UCSD/TUnA/tree/8cd8b079cae26ae6f431adaf9dcae591ba401d1a) is an interesting project that combines common sense deep learning wisdom with some seriously innovative combinations of techniques. The result is really high quality training schemes for PPI. Most models struggle to learn anything from [Bernett's dataset](https://huggingface.co/datasets/Synthyra/bernett_gold_ppi) without EXTENSIVE hyperparameter tuning, but ESM variants, TUnA arcitecture, and even GPs on top seem to stabily learn it to SOTA levels. Some of the nice tricks include
- The RFFlayer for uncertainty
- Look ahead optimization wrapper
- Random starting length sampling for sequences longer than the alloted max length

There seems to be some low hanging fruit in terms of improvements, but who knows how it will go
- [Synthyra](https://huggingface.co/Synthyra) base models for easier embedding, faster throughput, and enhanced information density
- Flash attention for the encoders for better thoughput
- I'm skeptical that spectral_norm is necessary but we'll see
- Cross attention in both directions may save on cost with O(AB) instead of O((A+B)^2)
- Longer max sequence lengths and consistent starting positions may help or hinder
- I'm going to be trying some compression schemes to see if we really need full-residue information
- token-parameter cross attention is almost always better than max pooling

## Get environment ready
```
git clone https://github.com/lhallee/TUnA.git
cd TUnA
chmod +x setup_bioenv.sh
./setup_bioenv.sh
source ~/bioenv/bin/activate
```

## To replicate the original TUnA
```
cd original_tuna
python -m process_bernett
cd results/bernett/TUnA
python -m main
```

## To run the current version
```
cd current
python -m main
```

### Change log
NA for bug or not calculated in runs yet

| Version / changes | Time / GH200 ↓ | Val AUC ↑ | Val Loss ↓ | Test AUC ↑ | Test Loss ↓ | Link |
|-|-|-|-|-|
| Reported results | NA | 0.666 |  | 0.70 |  |[paper](https://academic.oup.com/bib/article/25/5/bbae359/7720609#476124851) |
| Original tuna 1 | 7184s | 0.6605 | NA | NA | NA | [log](original_tuna/results/bernett/TUnA/base_tuna_4_12_25.txt) |
| Original tuna 2 | 7184s | 0.6605 | NA | NA | NA | [log](original_tuna/results/bernett/TUnA/base_tuna_4_12_25.txt) |
| Synthyra version of ESM2-150, improved torch Dataset | 2271s | 0.6754 | NA | NA | NA | [log](runs/1/switch_to_synthyra_4_13_25.txt) |
| Flash attn | 2139s | 0.6613 | NA | NA | NA |[log](runs/2/flash_attn_4_13_25.txt) |
| Fixed mask, rotary, swiglu, ESM2-650, batching during eval | 868s | 0.6854 | 0.6392 | 0.6978 | 0.6323 |[log](runs/3/650_4_14_25.txt) |
| bias=False, attention pooling, data collator update| 1225s | 0.6462 | 0.6635 | 0.6358 | NA | [log](runs/4/attention_pool_4_14_25.txt) |
| bias=False, back to max pool | 1177s | 0.6859 | 0.6376 | 0.6930 | NA |[log](runs/5/bias_false_4_14_25.txt) |
| ESM++ large, max length 2048| 34738s | 0.6700 | 0.6526 | NA | 0.6270 |[log](runs/6/2048_4_14_25.txt) |


### Notes
- Attention pooling looks to be worse and slower
- The collate method with `pack` may be slighly faster than the current multithreaded approach. May also just need more or less workers