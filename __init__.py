import os
import requests
import uuid
import folder_paths
import time
import glob
import subprocess
import json
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
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
        m_frames = 0
        l_frames = 0
        n_frames = 0
        p_frames = 0
        consecutive_sniper = 0
        max_consecutive_sniper = 0
        
        num_checked = len(frames_to_check)
        limit = max(1, int(num_checked * 0.3))
        
        for frame_idx, results in zip(frames_to_check, results_batch):
            s = {str(r['label']).lower(): r['score'] for r in results}
            
            safe_val = s.get('neutral', s.get('safe', s.get('sfw', s.get('normal', 0))))
            mild_sexy_val = s.get('low', s.get('drawings', 0))
            sexy_val = s.get('medium', s.get('sexy', s.get('questionable', 0)))
            porn_val = s.get('high', s.get('porn', s.get('nsfw', s.get('unsafe', s.get('hentai', 0)))))
            
            porn_scores.append(porn_val)
            
            max_val = max(safe_val, mild_sexy_val, sexy_val, porn_val)
            if max_val == safe_val:
                n_frames += 1
            elif max_val == mild_sexy_val:
                l_frames += 1
            elif max_val == sexy_val:
                m_frames += 1
            else:
                p_frames += 1
                
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
        
        if avg_porn > 0.50:
            is_blocked = True
        elif bad_frames >= limit:
            is_blocked = True
        elif max_consecutive_sniper >= 5:
            is_blocked = True
        
        status = "BLOCKED" if is_blocked else "PASSED"
        
        t1_status = " [TRIGGERED]" if avg_porn > 0.50 else ""
        t2_status = " [TRIGGERED]" if bad_frames >= limit else ""
        t3_status = " [TRIGGERED]" if max_consecutive_sniper >= 5 else ""
        
        final_log = f"[{status}] Source: {source_name}, Frames: {num_checked}\n"
        final_log += f"P-score avg: {avg_porn:.2f}, P-score max: {max_porn:.2f}\n"
        final_log += f"N-frames: {n_frames}/{num_checked}, L-frames: {l_frames}/{num_checked}, M-frames: {m_frames}/{num_checked}, P-frames: {p_frames}/{num_checked}\n"
        final_log += f"Trigger 1 (Avg > 0.50): {avg_porn:.2f}{t1_status}\n"
        final_log += f"Trigger 2 (Density > {threshold}): {bad_frames}/{limit}{t2_status}\n"
        final_log += f"Trigger 3 (Sniper > 0.98): {max_consecutive_sniper}/5{t3_status}"
        
        if is_blocked:
            raise ValueError(final_log)

        return (image, final_log)

class API_Video_Downloader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"forceInput": True})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("local_video",)
    FUNCTION = "download_video"
    CATEGORY = "API"

    def download_video(self, url):
        # If it's already a local file, just pass it through
        if not url.startswith("http"):
            print(f"[Video Downloader] Local file detected, skipping network request: {url}")
            return (url,)

        # Create a unique filename in the ComfyUI input directory
        filename = f"dl_{uuid.uuid4().hex[:8]}.mp4"
        filepath = os.path.join(folder_paths.get_input_directory(), filename)

        # Configure robust retries
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        print(f"[Video Downloader] Starting download from Linode...")
        print(f"[Video Downloader] Target file: {filename}")

        # Download the file in chunks with logging
        with session.get(url, stream=True) as r:
            r.raise_for_status()
            
            # Extract file size for percentage calculation
            total_size = int(r.headers.get('content-length', 0))
            downloaded_size = 0
            last_printed_mb = 0
            
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # Log progress every 5MB to avoid spamming the console
                    current_mb = downloaded_size // (1024 * 1024 * 5)
                    if current_mb > last_printed_mb:
                        last_printed_mb = current_mb
                        if total_size > 0:
                            percent = (downloaded_size / total_size) * 100
                            print(f"[Video Downloader] Progress: {downloaded_size / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({percent:.1f}%)")
                        else:
                            print(f"[Video Downloader] Downloaded: {downloaded_size / (1024*1024):.1f}MB...")

        print(f"[Video Downloader] Success! Video fully saved to {filepath}")

        # Auto-Cleanup: Delete downloaded videos older than 1 hour
        try:
            current_time = time.time()
            search_pattern = os.path.join(folder_paths.get_input_directory(), "dl_*.mp4")
            for old_file in glob.glob(search_pattern):
                if current_time - os.path.getmtime(old_file) > 3600:
                    os.remove(old_file)
                    print(f"[Video Downloader] Auto-cleaned old file: {os.path.basename(old_file)}")
        except Exception as e:
            print(f"[Video Downloader] Cleanup warning: {e}")

        return (f"input/{filename}",)

class API_Frames_Calculator:
    """
    Computes a safe `total_frames` value for WanVideo Animate workflows.
    
    Reads the source video's actual duration, then returns the largest valid
    Wan VAE length (4n+1) that does NOT exceed:
      1. The user's requested frame count
      2. What the video can physically supply at the target fps
    
    Logs every step for debugging.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "requested_frames": ("INT", {"default": 241, "min": 1, "max": 10000, "step": 1}),
                "target_fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 120.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("total_frames", "info")
    FUNCTION = "calculate"
    CATEGORY = "API"

    def _resolve_path(self, video_path):
        # Absolute or relative-to-CWD that already exists
        if os.path.isfile(video_path):
            return video_path
        # Relative to ComfyUI base path
        try:
            base_path = folder_paths.base_path
            candidate = os.path.join(base_path, video_path)
            if os.path.isfile(candidate):
                return candidate
        except AttributeError:
            pass
        # "input/..." prefix → input directory
        if video_path.startswith("input/"):
            candidate = os.path.join(
                folder_paths.get_input_directory(),
                video_path[len("input/"):],
            )
            if os.path.isfile(candidate):
                return candidate
        # Just basename in input dir
        candidate = os.path.join(
            folder_paths.get_input_directory(),
            os.path.basename(video_path),
        )
        if os.path.isfile(candidate):
            return candidate
        return None

    def _probe_ffprobe(self, path):
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate,nb_frames,duration:format=duration",
                "-of", "json",
                path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode != 0:
                print(f"[Frames Calculator] ffprobe rc={result.returncode}: {result.stderr.strip()}")
                return None, None, None
            data = json.loads(result.stdout)

            duration = None
            source_fps = None
            nb_frames = None

            streams = data.get("streams", [])
            if streams:
                st = streams[0]
                d = st.get("duration")
                if d:
                    try: duration = float(d)
                    except (TypeError, ValueError): pass
                fr = st.get("avg_frame_rate")
                if fr and "/" in fr:
                    try:
                        num, den = fr.split("/")
                        n, dn = float(num), float(den)
                        if dn > 0: source_fps = n / dn
                    except (ValueError, ZeroDivisionError): pass
                nb = st.get("nb_frames")
                if nb:
                    try: nb_frames = int(nb)
                    except (TypeError, ValueError): pass

            if duration is None:
                d = data.get("format", {}).get("duration")
                if d:
                    try: duration = float(d)
                    except (TypeError, ValueError): pass

            return duration, source_fps, nb_frames
        except FileNotFoundError:
            print("[Frames Calculator] ffprobe not found in PATH")
            return None, None, None
        except Exception as e:
            print(f"[Frames Calculator] ffprobe error: {type(e).__name__}: {e}")
            return None, None, None

    def _probe_cv2(self, path):
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"[Frames Calculator] cv2 cannot open {path}")
                return None, None, None
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            source_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if source_fps <= 0 or source_count <= 0:
                return None, None, None
            duration = source_count / source_fps
            return duration, source_fps, source_count
        except Exception as e:
            print(f"[Frames Calculator] cv2 error: {type(e).__name__}: {e}")
            return None, None, None

    def calculate(self, video_path, requested_frames, target_fps):
        log_lines = []
        def log(msg):
            print(f"[Frames Calculator] {msg}")
            log_lines.append(msg)

        log("=" * 50)
        log("Starting calculation")
        log(f"Inputs: video_path='{video_path}', requested_frames={requested_frames}, target_fps={target_fps}")

        if requested_frames < 1:
            err = f"ERROR: requested_frames must be >= 1 (got {requested_frames})"
            log(err); raise ValueError(err)
        if target_fps <= 0:
            err = f"ERROR: target_fps must be > 0 (got {target_fps})"
            log(err); raise ValueError(err)

        actual_path = self._resolve_path(video_path)
        if actual_path is None:
            err = f"ERROR: Could not resolve video path '{video_path}'"
            log(err); raise FileNotFoundError(err)
        log(f"Resolved path: {actual_path}")

        duration, source_fps, source_count = self._probe_ffprobe(actual_path)
        probe_source = "ffprobe"
        if duration is None:
            log("ffprobe unavailable/failed; falling back to cv2")
            duration, source_fps, source_count = self._probe_cv2(actual_path)
            probe_source = "cv2"

        if duration is None or duration <= 0:
            err = "ERROR: Could not determine video duration via ffprobe or cv2"
            log(err); raise RuntimeError(err)

        log(f"Probe via {probe_source}: duration={duration:.3f}s, "
            f"source_fps={source_fps if source_fps is None else f'{source_fps:.3f}'}, "
            f"source_frames={source_count}")

        # How many frames at TARGET fps the video can supply (floor — partial frames don't count)
        source_frames_at_target = int(duration * target_fps)
        log(f"At target_fps={target_fps}, video supplies {source_frames_at_target} frames "
            f"(= floor({duration:.3f} × {target_fps}))")

        target = min(requested_frames, source_frames_at_target)
        log(f"target = min(requested={requested_frames}, available={source_frames_at_target}) = {target}")

        if target < 5:
            err = (f"ERROR: target frames = {target}, below Wan VAE minimum (5). "
                   f"Source video too short for any usable generation.")
            log(err); raise ValueError(err)

        # Round DOWN to nearest 4n+1 (Wan VAE valid length)
        n = (target - 1) // 4
        corrected = 4 * n + 1
        result_duration = corrected / target_fps

        log(f"Round-down to 4n+1: n=(target-1)//4={n}, corrected=4n+1={corrected}")
        log(f"Result duration: {corrected}/{target_fps} = {result_duration:.3f}s")

        if corrected == requested_frames:
            log(f"OK: full requested length achievable ({corrected} frames, {result_duration:.3f}s)")
        else:
            reasons = []
            if source_frames_at_target < requested_frames:
                reasons.append(f"video supplies only {source_frames_at_target} frames at target fps")
            if (requested_frames - 1) % 4 != 0:
                reasons.append("requested length is not 4n+1")
            log(f"NOTE: reduced from {requested_frames} → {corrected} frames "
                f"({result_duration:.3f}s). Reason: {'; '.join(reasons) or 'rounded down to 4n+1'}")

        log("=" * 50)
        return (corrected, "\n".join(log_lines))

NODE_CLASS_MAPPINGS = {
    "API_Input_Panel": API_Input_Panel,
    "API_BBox_Switch": API_BBox_Switch,
    "NSFW_Image_Checker": NSFW_Image_Checker,
    "API_Video_Downloader": API_Video_Downloader,
    "API_Frames_Calculator": API_Frames_Calculator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "API_Input_Panel": "Input Panel",
    "API_BBox_Switch": "BBoxes Switch",
    "NSFW_Image_Checker": "NSFW Image Checker",
    "API_Video_Downloader": "Video Downloader",
    "API_Frames_Calculator": "Frames Calculator",
}