

## Environment setup:

```bash
bash _custom_setup.sh
```

## Naming
This repository uses the following model names:

- `HAT-mini`: referred to as `HAT_1` in the paper; smaller than `HAT-S`
- `HAT-dagger`: the full-sized HAT model



## Train

### HAT-mini baseline

```bash
python basicsr/train.py -opt options/train/HAT-mini/SRx2_HAT-mini_baseline.yml
python basicsr/train.py -opt options/train/HAT-mini/SRx4_HAT-mini_baseline.yml
```

### HAT-mini i-LN

```bash
python basicsr/train.py -opt options/train/HAT-mini/SRx2_HAT-mini_iLN.yml
python basicsr/train.py -opt options/train/HAT-mini/SRx4_HAT-mini_iLN.yml
```

### HAT-dagger i-LN

```bash
python basicsr/train.py -opt options/train/HAT-dagger/SRx2_HAT-dagger_iLN.yml
python basicsr/train.py -opt options/train/HAT-dagger/SRx4_HAT-dagger_iLN.yml
```

## Test

### HAT-mini baseline

```bash
python basicsr/test.py -opt options/test/HAT-mini/SRx2_HAT-mini_baseline.yml
python basicsr/test.py -opt options/test/HAT-mini/SRx4_HAT-mini_baseline.yml
```

### HAT-mini i-LN

```bash
python basicsr/test.py -opt options/test/HAT-mini/SRx2_HAT-mini_iLN.yml
python basicsr/test.py -opt options/test/HAT-mini/SRx4_HAT-mini_iLN.yml
```

### HAT-dagger i-LN

```bash
python basicsr/test.py -opt options/test/HAT-dagger/SRx2_HAT-dagger_iLN.yml
python basicsr/test.py -opt options/test/HAT-dagger/SRx4_HAT-dagger_iLN.yml
```
