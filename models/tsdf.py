import flax.linen as nn

from encoders.frequency import PositionalEncodingNeRF
from models.siren import Siren


class TruncatedDistanceFunction(nn.Module):
    num_hidden_layers: int = 3
    apply_position_encoding: bool = True
    truncation_distance: float = 0.01

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        """
         return the probability density of ray hitting the surface at the location
         `input`. This should ideally be a uni/multi-modal delta function. The
         expected depth value along the ray i.e. int(this_function(r(t)) z(r(t)), t)
         should be close to the ground truth from the depth image. For background pixels,
         this function should return 0 everywhere.
         """
        # TODO: just use SirenNeRFModel
        x = PositionalEncodingNeRF()(inputs) if self.apply_position_encoding else inputs
        x = Siren(hidden_layers=self.num_hidden_layers, out_features=1)(x)
        # TODO: re-enable it.
        # x = x / self.truncation_distance
        # x = nn.sigmoid(x) * nn.sigmoid(-x)
        return x
