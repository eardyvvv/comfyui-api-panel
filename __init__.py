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


NODE_CLASS_MAPPINGS = {
    "API_Input_Panel": API_Input_Panel,
    "API_BBox_Switch": API_BBox_Switch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "API_Input_Panel": "Input Panel",
    "API_BBox_Switch": "BBoxes Switch"
}