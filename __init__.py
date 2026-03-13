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
                "threshold": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "source_name": ("STRING", {"default": "Media_Input"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "check_result")
    FUNCTION = "check_nsfw"
    CATEGORY = "API"
    OUTPUT_NODE = True

    def check_nsfw(self, image, threshold, source_name):
        global _classifier
        if _classifier is None:
            _classifier = pipeline("image-classification", model="Freepik/nsfw_image_detector", device=0)
            
        total_frames = image.shape[0]
        num_checked = min(50, total_frames)
        if total_frames > 1:
            step = (total_frames - 1) / (num_checked - 1)
            frames_to_check = [int(step * i) for i in range(num_checked - 1)] + [total_frames - 1]
            frames_to_check = sorted(list(set(frames_to_check)))
        else:
            frames_to_check = [0]
            
        images_to_process = []
        for frame_idx in frames_to_check:
            i = 255. * image[frame_idx].cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            images_to_process.append(img)
            
        results_batch = _classifier(images_to_process)
        
        if not isinstance(results_batch[0], list):
            results_batch = [results_batch]
            
        porn_scores = []
        bad_frames = 0
        s_frames = 0
        consecutive_sniper = 0
        max_consecutive_sniper = 0
        
        num_checked = len(frames_to_check)
        limit = max(1, int(num_checked * 0.3))
        
        for frame_idx, results in zip(frames_to_check, results_batch):
            s = {str(r['label']).lower(): r['score'] for r in results}
            
            sexy_val = s.get('medium', s.get('sexy', s.get('questionable', 0)))
            porn_val = s.get('high', s.get('porn', s.get('nsfw', s.get('unsafe', s.get('hentai', 0)))))
            
            porn_scores.append(porn_val)
            
            if sexy_val > 0.95:
                s_frames += 1
                
            if porn_val > threshold:
                bad_frames += 1
                
            if porn_val > 0.98:
                consecutive_sniper += 1
                if consecutive_sniper > max_consecutive_sniper:
                    max_consecutive_sniper = consecutive_sniper
            else:
                consecutive_sniper = 0

        avg_porn = sum(porn_scores) / len(porn_scores) if porn_scores else 0
        max_porn = max(porn_scores) if porn_scores else 0
        
        is_blocked = False
        block_reason = ""
        
        if avg_porn > 0.50:
            is_blocked = True
            block_reason = "Trigger 1 (Avg > 0.50)"
        elif bad_frames >= limit:
            is_blocked = True
            block_reason = f"Trigger 2 (Density > {threshold})"
        elif max_consecutive_sniper >= 5:
            is_blocked = True
            block_reason = "Trigger 3 (Sniper > 0.98)"
        
        status = "BLOCKED" if is_blocked else "PASSED"
        
        t1_status = " [TRIGGERED]" if avg_porn > 0.50 else ""
        t2_status = " [TRIGGERED]" if bad_frames >= limit else ""
        t3_status = " [TRIGGERED]" if max_consecutive_sniper >= 5 else ""
        
        final_log = f"[{status}] Source: {source_name} | Frames: {num_checked}\n"
        final_log += f"P-score avg: {avg_porn:.2f} | P-score max: {max_porn:.2f}\n"
        final_log += f"S-frames (>0.95): {s_frames}/{num_checked}\n\n"
        final_log += f"Trigger 1 (Avg > 0.50): {avg_porn:.2f}{t1_status}\n"
        final_log += f"Trigger 2 (Density > {threshold}): {bad_frames}/{limit}{t2_status}\n"
        final_log += f"Trigger 3 (Sniper > 0.98): {max_consecutive_sniper}/5{t3_status}\n"
        
        if is_blocked:
            final_log += f"Reason: {block_reason}"
            raise ValueError(final_log)

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