# ASR
```
@inproceedings{lim2026and,
  title={When and Where to Reset Matters for Long-Term Test-Time Adaptation},
  author={Lim, Taejun and Hwang, Joong-Won and Lee, Kibok},
  booktitle={ICLR},
  year={2026}
}
```

## Setup
```
conda env create -f environment.yml
conda activate asr
pip install -r requirements.txt
```

## Datasets

## Code Replication

### Table 28, 29, 30
```
# baseline: ETA
bash scripts/asr_eta_ccc.sh

# baseline: ROID
bash scripts/asr_roid_ccc.sh
```
### Table 31
```
# baseline: ETA
bash scripts/asr_eta_cin_c.sh

# baseline: ROID
bash scripts/asr_roid_cin_c.sh
```
### Table 32
```
# baseline: ETA
bash scripts/asr_eta_cin_c_non_iid.sh

# baseline: ROID
bash scripts/asr_roid_cin_c_non_iid.sh
```
### Table 33
```
# baseline: ETA
bash scripts/asr_eta_in_c.sh

# baseline: ROID
bash scripts/asr_roid_in_c.sh
```
### Table 34
```
# baseline: ETA
bash scripts/asr_eta_in_d.sh

# baseline: ROID
bash scripts/asr_roid_in_d.sh
```
### Table 35
```
# baseline: ETA
bash scripts/asr_eta_ccc40.0_vit.sh

# baseline: ROID
bash scripts/asr_roid_ccc40.0_vit.sh
```
### Table 36
```
# baseline: ETA
bash scripts/asr_eta_ccc20.0_vit.sh

# baseline: ROID
bash scripts/asr_roid_ccc20.0_vit.sh
```
### Table 37
```
# baseline: ETA
bash scripts/asr_eta_ccc0.0_vit.sh

# baseline: ROID
bash scripts/asr_roid_ccc0.0_vit.sh
```
```
```
