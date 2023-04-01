# LorentzGCN

## Env
```shell
conda create -n lorentz python=3.10
conda activate lorentz
pip install -r requirements.txt
```
requirements:
```
numpy==1.23.5
scipy==1.10.0
torch==1.13.0
torch_geometric==2.2.0
tqdm==4.64.1
wandb==0.14.0
```

## Run
```shell
chmod +x ./scripts/lorentz_gcn.sh
./scripts/lorentz_gcn.sh
```
or
```shell
python main.py
```