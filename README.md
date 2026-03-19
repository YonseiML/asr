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
* **CCC**: [[check this github](https://github.com/oripress/CCC)] // please follow the instruction provided in the github
* **CIN-C**, **IN-C**: [[download link](https://zenodo.org/records/2235448#.Yj2RO_co_mF)]
* **IN-D109**: [[download link](https://ai.bu.edu/M3SDA/)] // please download the clean version

## Core Mechanism
```
# methods/base.py
if self.asr_on:
    self.asr_step += 1
    self.entropy = self.softmax_entropy(outputs.mean(dim=0, keepdim=True)).item()

    y_src = self.src(x[0]).max(-1)[1]
    y_tgt = outputs.max(-1)[1]
    y_flip = (sum(y_src != y_tgt) / len(y_tgt)).item()

    self.m_entropy = self.asr_m0 * y_flip + 1 - self.asr_m0
    self.p_fisher = self.asr_p0 * (y_flip ** 2)

    if self.entropy < self.cum_entropy:
        self.register_ewc_params(self.asr_step, momentum=0.9)
        self.reset()
    else:
        self.cum_entropy = self.m_entropy * self.cum_entropy + (1 - self.m_entropy) * self.entropy

def asr_reset(self):
    for model, model_state, src_state in zip(self.models, self.model_states, self.src_states):
        scope = max(self.asr_r0 - self.asr_r1 * (self.cum_entropy - self.entropy), 0)
        new_states = []
        for i, (param, nm) in enumerate(zip(self.params, self.param_names)):
            if i < scope * len(self.params):
                new_states.append([nm, param])
            else:
                new_states.append([nm, src_state[nm]])
        model_state.update(deepcopy(dict(new_states)))
        model.load_state_dict(model_state, strict=False)
    self.optimizer.load_state_dict(self.optimizer_state)
    logger.info(f"[reset] scope={scope:.2f}, asr_step={self.asr_step}, entropy={self.entropy:.3f}, cum_entropy={self.cum_entropy:.3f}")
    self.cum_entropy = math.log(self.num_classes * self.asr_a0)
    self.asr_step = 0
```

## Code Replication

### Tables 1, 28, 29, 30
```
# baseline: ETA
bash scripts/asr_eta_ccc.sh

# baseline: ROID
bash scripts/asr_roid_ccc.sh
```
### Tables 3, 35
```
# baseline: ETA
bash scripts/asr_eta_ccc40.0_vit.sh

# baseline: ROID
bash scripts/asr_roid_ccc40.0_vit.sh
```
### Tables 3, 36
```
# baseline: ETA
bash scripts/asr_eta_ccc20.0_vit.sh

# baseline: ROID
bash scripts/asr_roid_ccc20.0_vit.sh
```
### Tables 3, 37
```
# baseline: ETA
bash scripts/asr_eta_ccc0.0_vit.sh

# baseline: ROID
bash scripts/asr_roid_ccc0.0_vit.sh
```
### Tables 4, 31
```
# baseline: ETA
bash scripts/asr_eta_cin_c.sh

# baseline: ROID
bash scripts/asr_roid_cin_c.sh
```
### Tables 4, 32
```
# baseline: ETA
bash scripts/asr_eta_cin_c_non_iid.sh

# baseline: ROID
bash scripts/asr_roid_cin_c_non_iid.sh
```
### Tables 4, 33
```
# baseline: ETA
bash scripts/asr_eta_in_c.sh

# baseline: ROID
bash scripts/asr_roid_in_c.sh
```
### Tables 4, 34
```
# baseline: ETA
bash scripts/asr_eta_in_d.sh

# baseline: ROID
bash scripts/asr_roid_in_d.sh
```
