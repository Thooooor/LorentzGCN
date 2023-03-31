from .light_gcn import LightGCN
from .hgcf import HGCF
from .lorentz_gcn import LorentzGCN


ALL_MODELS = {
    "light_gcn": LightGCN,
    "hgcf": HGCF,
    "lorentz_gcn": LorentzGCN,
}
