from .v1 import MuseVPredictorV1, MuseVImg2VidV1
from .utils import AnimationZoom, ImageSelector

NODE_CLASS_MAPPINGS = {
    "MuseVPredictor V1 (comfyui_musev_evolved)": MuseVPredictorV1,
    "MuseVImg2Vid V1 (comfyui_musev_evolved)": MuseVImg2VidV1,
    "AnimationZoom (comfyui_musev_evolved)": AnimationZoom,
    "ImageSelector (comfyui_musev_evolved)": ImageSelector,
}
