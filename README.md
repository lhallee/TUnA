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

## Minimal reproduction
```
cd minimal_reproduction
py -m main
```