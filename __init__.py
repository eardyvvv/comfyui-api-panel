import torch
import numpy as np
from PIL import Image
from transformers import pipeline, AutoImageProcessor

class API_Input_Panel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "img_url": ("STRING", {"default": "https://example.com/image.jpg", "multiline": False}),
                "video_url": ("STRING", {"default": "https://example.com/video.mp4", "multiline": False}),
                "total_frames": ("INT", {"default": 100, "min": 1, "max": 10000, "step": 1}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "mask_width": ("INT", {"default": 512, "min": 10, "max": 4096, "step": 1}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 999999999999999, "step": 1}),
                "face_only": ("BOOLEAN", {"default": False}),
                "width": ("INT", {"default": 480, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 856, "min": 1, "max": 8192, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "FLOAT", "STRING", "STRING", "INT", "BOOLEAN", "INT", "INT")
    RETURN_NAMES = ("img_url", "video_url", "total_frames", "skip_first_frames", "mask_width", "frame_rate", "positive_prompt", "negative_prompt", "seed", "face_only", "width", "height")
    FUNCTION = "get_values"
    CATEGORY = "API"

    def get_values(self, img_url, video_url, total_frames, skip_first_frames, mask_width, frame_rate, positive_prompt, negative_prompt, seed, face_only, width, height):
        return (img_url, video_url, total_frames, skip_first_frames, mask_width, frame_rate, positive_prompt, negative_prompt, seed, face_only, width, height)


class API_BBox_Switch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes_body": ("BBOX",),
                "bboxes_face": ("BBOX",),
                "face_only": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("BBOX",)
    RETURN_NAMES = ("bboxes",)
    FUNCTION = "switch"
    CATEGORY = "API"

    def switch(self, bboxes_body, bboxes_face, face_only):
        if face_only:
            return (bboxes_face,)
        return (bboxes_body,)


_classifier = None

class NSFW_Image_Checker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sexy_threshold": ("FLOAT", {"default": 0.98, "min": 0.0, "max": 1.0, "step": 0.01}),
                "minor_threshold": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "check_result")
    FUNCTION = "check_nsfw"
    CATEGORY = "API"
    OUTPUT_NODE = True

    def check_nsfw(self, image, threshold, sexy_threshold, minor_threshold):
        global _classifier
        if _classifier is None:
            _classifier = pipeline("zero-shot-image-classification", model="/workspace/ComfyUI/models/siglip")
            
        total_frames = image.shape[0]
        if total_frames > 1:
            step = (total_frames - 1) / 9
            frames_to_check = [int(step * i) for i in range(9)] + [total_frames - 1]
            frames_to_check = sorted(list(set(frames_to_check)))
        else:
            frames_to_check = [0]
            
        images_to_process = []
        for frame_idx in frames_to_check:
            i = 255. * image[frame_idx].cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            images_to_process.append(img)
            
        labels = [
            "portrait, everyday scene, clothed person",
            "revealing clothing, deep cleavage",
            "bikini, swimwear, lingerie",
            "explicit nude pornography, exposed genitals",
            "child, baby",
            "sports, fitness, yoga",
            "close-up photo of skin",
            "classical art, painting, statue"
        ]
        results_batch = _classifier(images_to_process, candidate_labels=labels, multi_label=True)
        
        if not isinstance(results_batch[0], list):
            results_batch = [results_batch]
            
        log_messages = []
        bad_frames = 0
        limit = 5 if total_frames > 1 else 1
        
        for frame_idx, results in zip(frames_to_check, results_batch):
            s = {r['label']: r['score'] for r in results}
            safe_val = s.get(labels[0], 0)
            mild_sexy_val = s.get(labels[1], 0)
            sexy_val = s.get(labels[2], 0)
            porn_val = s.get(labels[3], 0)
            minor_val = s.get(labels[4], 0)
            
            frame_log = f"F{frame_idx}: safe:{safe_val:.4f}, mild_sexy:{mild_sexy_val:.4f}, sexy:{sexy_val:.4f}, porn:{porn_val:.4f}, minor:{minor_val:.4f}"
            log_messages.append(frame_log)
            
            is_bad = False
            if porn_val > threshold:
                is_bad = True
            if minor_val > minor_threshold:
                is_bad = True
            if sexy_val > sexy_threshold:
                is_bad = True
                
            if is_bad:
                bad_frames += 1
                if bad_frames >= limit:
                    raise ValueError(f"Blocked: NSFW limit reached ({bad_frames} frames). Last trigger: {frame_log}")

        final_log = "\n".join(log_messages)
        return (image, final_log)

NODE_CLASS_MAPPINGS = {
    "API_Input_Panel": API_Input_Panel,
    "API_BBox_Switch": API_BBox_Switch,
    "NSFW_Image_Checker": NSFW_Image_Checker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "API_Input_Panel": "Input Panel",
    "API_BBox_Switch": "BBoxes Switch",
    "NSFW_Image_Checker": "NSFW Image Checker"
}