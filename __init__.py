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
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("img_url", "video_url", "total_frames", "skip_first_frames", "mask_width", "frame_rate", "positive_prompt", "negative_prompt")
    FUNCTION = "get_values"
    CATEGORY = "API"

    def get_values(self, img_url, video_url, total_frames, skip_first_frames, mask_width, frame_rate, positive_prompt, negative_prompt):
        return (img_url, video_url, total_frames, skip_first_frames, mask_width, frame_rate, positive_prompt, negative_prompt)


NODE_CLASS_MAPPINGS = {
    "API_Input_Panel": API_Input_Panel
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "API_Input_Panel": "🛠️ API Input Panel"
}