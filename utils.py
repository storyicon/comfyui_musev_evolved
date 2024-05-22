import numpy as np
from PIL import Image
import torch

class AnimationZoom:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "scale": ("FLOAT", {"default":0.5, "round": False, "step":0.01}),
                "frame": ("INT", {"default": 10}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "MuseV Evolved"
    def main(self, image, scale, frame):
        pil_image = Image.fromarray((image* 255).byte().numpy()[0]).convert('RGB')
        w, h = pil_image.size
        scales = np.linspace(1, scale, frame)
        frames = []
        for scale in scales:
            res = Image.new("RGB", (w, h), color="white")
            offset_x = int(w/2 - int(w * scale / 2))
            offset_y = int(h/2 - int(h * scale / 2))
            res.paste(pil_image.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR), (offset_x, offset_y))
            tensor = torch.from_numpy(np.array(res)).float().div(255) 
            frames.append(tensor)
        return (torch.stack(frames), )

class ImageSelector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "selected_indexes": ("STRING", {
                    "multiline": False,
                    "default": "1,2,3"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", )

    FUNCTION = "run"

    OUTPUT_NODE = False

    CATEGORY = "MuseV Evolved"

    def run(self, images: torch.Tensor, selected_indexes: str):
        shape = images.shape
        len_first_dim = shape[0]

        selected_index: list[int] = []
        total_indexes: list[int] = list(range(len_first_dim))
        for s in selected_indexes.strip().split(','):
            try:
                if ":" in s:
                    _li = s.strip().split(':', maxsplit=1)
                    _start = _li[0]
                    _end = _li[1]
                    if _start and _end:
                        selected_index.extend(
                            total_indexes[int(_start):int(_end)]
                        )
                    elif _start:
                        selected_index.extend(
                            total_indexes[int(_start):]
                        )
                    elif _end:
                        selected_index.extend(
                            total_indexes[:int(_end)]
                        )
                else:
                    x: int = int(s.strip())
                    if x < len_first_dim:
                        selected_index.append(x)
            except:
                pass

        if selected_index:
            print(f"ImageSelector: selected: {len(selected_index)} images")
            return (images[selected_index, :, :, :], )

        print(f"ImageSelector: selected no images, passthrough")
        return (images, )
