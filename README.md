## [ICLR2026] _i-LN_: Analyzing the Training Dynamics of Image Restoration Transformers: A Revisit to Layer Normalization

[Arxiv](https://arxiv.org/abs/2504.06629) | [OpenReview](https://openreview.net/forum?id=SbLj5hJXh6)

MinKyu Lee, Sangeek Hyun, Woojin Jun, Hyunjun Kim, Jiwoo Chung, Jae-Pil Heo* \
Sungkyunkwan University \
\*: Corresponding Author


### Abstract
> This work analyzes the training dynamics of Image Restoration (IR) Transformers and uncovers a critical yet overlooked issue: conventional LayerNorm (LN) drives feature magnitudes to diverge to a _million scale_ and collapses channel-wise entropy. We analyze this in the perspective of networks attempting to bypass LN’s constraints that conflict with IR tasks. Accordingly, we address two misalignments between LN and IR: 1) _per-token normalization_ disrupts spatial correlations, and 2) _input-independent scaling_ discards input-specific statistics. To address this, we propose Image Restoration Transformer Tailored Layer Normalization _i_-LN, a simple drop-in replacement that normalizes features holistically and adaptively rescales them per input. We provide theoretical insights and empirical evidence that this simple design effectively leads to both improved training dynamics and thereby improved performance, validated by extensive experiments.


## Acknowledgement
This project is built based on [BasicSR](https://github.com/XPixelGroup/BasicSR) and also
[SwinIR](https://github.com/cszn/KAIR/tree/master)
[HAT](https://github.com/XPixelGroup/HAT)
[DRCT](https://github.com/ming053l/drct),


## Contact
Please contact me via 2minkyulee@gmail.com for any inquiries.
