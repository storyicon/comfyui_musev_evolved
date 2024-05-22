import os
import folder_paths
import numpy as np
from PIL import Image
import torch

diffusers_path = folder_paths.get_folder_paths("diffusers")[0]

MuseVCheckPointDir = os.path.join(
    diffusers_path, "TMElyralab/MuseV"
)

current_dir = os.path.dirname(__file__)

import sys
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "MMCM"))
sys.path.insert(0, os.path.join(current_dir, "diffusers/src"))
sys.path.insert(0, os.path.join(current_dir, "controlnet_aux/src"))

from einops import repeat
from .MMCM.mmcm.utils.seed_util import set_all_seed
from .MMCM.mmcm.utils.task_util import fiss_tasks, generate_tasks as generate_tasks_from_table
from musev.pipelines.pipeline_controlnet_predictor import (
    DiffusersPipelinePredictor,
)
from musev.models.unet_loader import load_unet_by_name
from musev import logger

logger.setLevel("INFO")

file_dir = os.path.dirname(__file__)
PROJECT_DIR = file_dir
DATA_DIR = os.path.join(PROJECT_DIR, "data")

class MuseVPredictorV1:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {}
        }

    RETURN_TYPES = ("MUSEV_PREDICTOR",)
    FUNCTION = "main"
    CATEGORY = "MuseV Evolved"
    def main(self):
        sd_model_path = os.path.join(folder_paths.models_dir, 'diffusers/TMElyralab/MuseV/t2i/sd1.5/majicmixRealv6Fp16')
        sd_unet_model = os.path.join(folder_paths.models_dir, 'diffusers/TMElyralab/MuseV/motion/musev_referencenet')
        vae_path = os.path.join(folder_paths.models_dir, 'diffusers/TMElyralab/MuseV/vae/sd-vae-ft-mse')
        negative_embedding = [
            [
                os.path.join(folder_paths.models_dir, "diffusers/TMElyralab/MuseV/embedding/badhandv4.pt"),
                "badhandv4"
            ],
            [
                os.path.join(folder_paths.models_dir, "diffusers/TMElyralab/MuseV/embedding/ng_deepnegative_v1_75t.pt"),
                "ng_deepnegative_v1_75t"
            ],
            [
                os.path.join(folder_paths.models_dir, "diffusers/TMElyralab/MuseV/embedding/EasyNegativeV2.safetensors"),
                "EasyNegativeV2"
            ],
            [
                os.path.join(folder_paths.models_dir, "diffusers/TMElyralab/MuseV/embedding/bad_prompt_version2-neg.pt"),
                "bad_prompt_version2-neg"
            ]
        ]
        unet = load_unet_by_name(
            model_name='musev_referencenet',
            sd_unet_model=sd_unet_model,
            sd_model=sd_model_path,
            cross_attention_dim=768,
            need_t2i_facein=False,
            strict=True,
            need_t2i_ip_adapter_face=False,
        )
        sd_predictor = DiffusersPipelinePredictor(
            sd_model_path=sd_model_path,
            unet=unet,
            lora_dict=None,
            lcm_lora_dct=None,
            device='cuda',
            dtype=torch.float16,
            negative_embedding=negative_embedding,
            referencenet=None,
            ip_adapter_image_proj=None,
            vision_clip_extractor=None,
            facein_image_proj=None,
            face_emb_extractor=None,
            vae_model=vae_path,
            ip_adapter_face_emb_extractor=None,
            ip_adapter_face_image_proj=None,
        )
        return (sd_predictor, )

class MuseVImg2VidV1:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "musev_predictor": ("MUSEV_PREDICTOR", ),
                "time_size": ("INT", {"default":12}),
                "seed": ("INT", {"default":1234}),
                "video_num_inference_steps": ("INT", {"default":10}),
                "video_guidance_scale": ("FLOAT", {"default":3.5, "round": False, "step":0.01}),
                "w_ind_noise": ("FLOAT", {"default":0.5, "round": False, "step":0.01}),
                "image_weight": ("FLOAT", {"default":0.001, "round": False, "step":0.001}),
                "motion_speed": ("FLOAT", {"default":8.0, "round": False, "step":0.01}),
                "context_frames": ("INT", {"default":12}),
                "context_stride": ("INT", {"default":1}),
                "context_overlap": ("INT", {"default":4}),
                "output_shift_first_frame": ("BOOLEAN", {"default":True}),
                "positive_prompt": ("STRING", {"multiline": True, "default": "(masterpiece, best quality, highres:1),(1girl, solo:1),(beautiful face, soft skin, costume:1),(eye blinks:1.8),(head wave:1.3)"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "badhandv4, ng_deepnegative_v1_75t, (((multiple heads))), (((bad body))), (((two people))), ((extra arms)), ((deformed body)), (((sexy))), paintings,(((two heads))), ((big head)),sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, (((nsfw))), nipples, extra fingers, (extra legs), (long neck), mutated hands, (fused fingers), (too many fingers)"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "MuseV Evolved"

    def main(
        self,
        image,
        musev_predictor,
        time_size,
        seed,
        w_ind_noise,
        context_frames,
        context_stride,
        context_overlap,
        video_num_inference_steps,
        video_guidance_scale,
        output_shift_first_frame,
        positive_prompt,
        negative_prompt,
        image_weight,
        motion_speed,
    ):
        cpu_generator, gpu_generator = set_all_seed(seed)
        condition_image = 255.0 * image[0].cpu().numpy()
        condition_image = np.clip(condition_image, 0, 255).astype(np.uint8)
        condition_image = repeat(condition_image, "h w c-> b c t h w", b=1, t=1)
        width = condition_image.shape[4]
        height = condition_image.shape[3]
        out_videos = musev_predictor.run_pipe_text2video(
            video_length=time_size,
            prompt=positive_prompt,
            width=width,
            height=height,
            generator=gpu_generator,
            noise_type='video_fusion',
            negative_prompt=negative_prompt,
            video_negative_prompt=negative_prompt,
            max_batch_num=1,
            strength=0.8,
            need_img_based_video_noise=True,
            video_num_inference_steps=video_num_inference_steps,
            condition_images=condition_image,
            fix_condition_images=False,
            video_guidance_scale=video_guidance_scale,
            guidance_scale=7.5,
            num_inference_steps=30,
            redraw_condition_image=False,
            img_weight=image_weight,
            w_ind_noise=w_ind_noise,
            n_vision_condition=1,
            motion_speed=motion_speed,
            need_hist_match=False,
            video_guidance_scale_end=None,
            video_guidance_scale_method='linear',
            vision_condition_latent_index=None,
            refer_image=None,
            fixed_refer_image=True,
            redraw_condition_image_with_referencenet=True,
            ip_adapter_image=None,
            refer_face_image=None,
            fixed_refer_face_image=True,
            facein_scale=1.0,
            redraw_condition_image_with_facein=True,
            ip_adapter_face_scale=1.0,
            redraw_condition_image_with_ip_adapter_face=True,
            fixed_ip_adapter_image=True,
            ip_adapter_scale=1.0,
            redraw_condition_image_with_ipdapter=True,
            prompt_only_use_image_prompt=False,
            # serial_denoise parameter start
            record_mid_video_noises=False,
            record_mid_video_latents=False,
            video_overlap=1,
            # serial_denoise parameter end
            # parallel_denoise parameter start
            context_schedule='uniform_v2',
            context_frames=context_frames,
            context_stride=context_stride,
            context_overlap=context_overlap,
            context_batch_size=1,
            interpolation_factor=1,
            # parallel_denoise parameter end
        )
        video = torch.from_numpy(out_videos).permute(0,2,3,4,1)
        if output_shift_first_frame:
            video = video[:, 1:, :, :, :]
        return video
