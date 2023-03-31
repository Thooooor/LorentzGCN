from .light_layer import LightLayer
from .lorentz_layer import HyperbolicLayer, LorentzLayer
from .stack_layers import PlainLayers, ResSumLayers, ResAddLayers, DenseLayers


ALL_STACK_LAYERS = {
    "plain": PlainLayers,
    "res_sum": ResSumLayers,
    "res_add": ResAddLayers,
    "dense": DenseLayers,
}
