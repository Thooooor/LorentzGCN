from .rsgd import RiemannianSGD
import torch.optim as optim

ALL_OPTIMIZERS = {
    "rsgd": RiemannianSGD,
    "adam": optim.Adam,
    "sgd": optim.SGD,
}
