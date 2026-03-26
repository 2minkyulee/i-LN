# [ICLR 2026] _i-LN_: Analyzing the Training Dynamics of Image Restoration Transformers: A Revisit to Layer Normalization

[Arxiv](https://arxiv.org/abs/2504.06629) | [OpenReview](https://openreview.net/forum?id=SbLj5hJXh6)

MinKyu Lee, Sangeek Hyun, Woojin Jun, Hyunjun Kim, Jiwoo Chung, Jae-Pil Heo* \
Sungkyunkwan University \
\*: Corresponding Author

### Abstract
> This work analyzes the training dynamics of Image Restoration (IR) Transformers and uncovers a critical yet overlooked issue: conventional LayerNorm (LN) drives feature magnitudes to diverge to a _million scale_ and collapses channel-wise entropy. We analyze this in the perspective of networks attempting to bypass LN's constraints that conflict with IR tasks. Accordingly, we address two misalignments between LN and IR: 1) _per-token normalization_ disrupts spatial correlations, and 2) _input-independent scaling_ discards input-specific statistics. To address this, we propose Image Restoration Transformer Tailored Layer Normalization _i_-LN, a simple drop-in replacement that normalizes features holistically and adaptively rescales them per input. We provide theoretical insights and empirical evidence that this simple design effectively leads to both improved training dynamics and thereby improved performance, validated by extensive experiments.

<table align="center" width="50%" style="border-collapse: collapse; border: none; background: transparent;">
  <tr style="border: none; background: transparent;">
    <td align="center" valign="top" width="50%" style="border: none; background: transparent; padding: 0 10px;">
      <img src="https://arxiv.org/html/2504.06629v2/x1.png" width="100%" />
    </td>
    <td align="center" valign="top" width="50%" style="border: none; background: transparent; padding: 0 10px;">
      <img src="https://arxiv.org/html/2504.06629v2/x2.png" width="100%" />
    </td>
  </tr>
  <tr style="border: none; background: transparent;">
    <td align="center" style="border: none; background: transparent; padding-top: 4px;">
      <sub>(a) Feature Magnitudes</sub>
    </td>
    <td align="center" style="border: none; background: transparent; padding-top: 4px;">
      <sub>(b) Channel-wise Entropy</sub>
    </td>
  </tr>
</table>

<p align="center">
  <em><b>Figure 1.</b> Visualization of feature magnitudes and channel entropy during training of an Image Restoration (IR) Transformer.</em>
</p>

## Repository Layout

- [`iLN/`](iLN): full project code, configs, scripts, and assets
- [`iLN/README.md`](iLN/README.md): project-local README

All commands below assume you are inside `iLN/`.

## Environment Setup

```bash
cd iLN
bash _custom_setup.sh
```

## Naming

- `HAT-mini`: referred to as `HAT_1` in the paper; smaller than `HAT-S`
- `HAT-dagger`: the full-sized HAT model

## Train

### HAT-mini baseline

```bash
cd iLN
python basicsr/train.py -opt options/train/HAT-mini/SRx2_HAT-mini_baseline.yml
python basicsr/train.py -opt options/train/HAT-mini/SRx4_HAT-mini_baseline.yml
```

### HAT-mini i-LN

```bash
cd iLN
python basicsr/train.py -opt options/train/HAT-mini/SRx2_HAT-mini_iLN.yml
python basicsr/train.py -opt options/train/HAT-mini/SRx4_HAT-mini_iLN.yml
```

### HAT-dagger i-LN

```bash
cd iLN
python basicsr/train.py -opt options/train/HAT-dagger/SRx2_HAT-dagger_iLN.yml
python basicsr/train.py -opt options/train/HAT-dagger/SRx4_HAT-dagger_iLN.yml
```

## Test

### HAT-mini baseline

```bash
cd iLN
python basicsr/test.py -opt options/test/HAT-mini/SRx2_HAT-mini_baseline.yml
python basicsr/test.py -opt options/test/HAT-mini/SRx4_HAT-mini_baseline.yml
```

### HAT-mini i-LN

```bash
cd iLN
python basicsr/test.py -opt options/test/HAT-mini/SRx2_HAT-mini_iLN.yml
python basicsr/test.py -opt options/test/HAT-mini/SRx4_HAT-mini_iLN.yml
```

### HAT-dagger i-LN

```bash
cd iLN
python basicsr/test.py -opt options/test/HAT-dagger/SRx2_HAT-dagger_iLN.yml
python basicsr/test.py -opt options/test/HAT-dagger/SRx4_HAT-dagger_iLN.yml
```

## Status

- [x] Code release
- [ ] Model checkpoint release

---

## Acknowledgement

This project is built based on:

- [BasicSR](https://github.com/XPixelGroup/BasicSR)
- [SwinIR](https://github.com/cszn/KAIR/tree/master)
- [HAT](https://github.com/XPixelGroup/HAT)
- [DRCT](https://github.com/ming053l/drct)

## Contact

Please contact me via 2minkyulee@gmail.com for any inquiries.
