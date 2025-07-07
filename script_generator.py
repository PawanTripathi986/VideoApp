import os
import cv2
import uuid
import textwrap
import numpy as np
import subprocess
from gtts import gTTS
from ai_video_generator import AIVideoGenerator
import tempfile

def cartoonize_frame(img):
    # Convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply median blur
    gray_blur = cv2.medianBlur(gray, 7)
    # Detect edges using adaptive threshold
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    # Apply bilateral filter to smooth colors while preserving edges
    color = cv2.bilateralFilter(img, 9, 300, 300)
    # Combine edges and color
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    # Convert edges to 3 channel
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # Enhance edges by combining with color
    cartoon = cv2.bitwise_and(color, edges_colored)
    # Additional sharpening filter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    cartoon = cv2.morphologyEx(cartoon, cv2.MORPH_CLOSE, kernel)
    # Increase contrast
    lab = cv2.cvtColor(cartoon, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    cartoon = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return cartoon

def convert_avi_to_mp4(avi_path, mp4_path):
    command = ['ffmpeg', '-y', '-i', avi_path, '-vcodec', 'libx264', '-acodec', 'aac', mp4_path]
    subprocess.run(command, check=True)

def generate_audio_from_script(script, audio_path):
    tts = gTTS(text=script, lang='en')
    tts.save(audio_path)

def merge_audio_video(video_path, audio_path, output_path):
    command = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        output_path
    ]
    subprocess.run(command, check=True)

import ast

def generate_individual_character_video(character, video_id):
    import cv2
    import logging
    output_dir = os.path.join("output")
    os.makedirs(output_dir, exist_ok=True)
    avi_path = os.path.join(output_dir, f"{video_id}.avi")
    mp4_path = os.path.join(output_dir, f"{video_id}.mp4")
    audio_path = os.path.join(output_dir, f"{video_id}.mp3")
    final_output_path = os.path.join(output_dir, f"{video_id}_final.mp4")

    width, height, fps = 640, 480, 10

    # Generate speech audio from character script
    script_text = character.get("script", "")
    generate_audio_from_script(script_text, audio_path)

    # Get audio duration using ffprobe
    def get_audio_duration(path):
        import subprocess
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries',
             'format=duration', '-of',
             'default=noprint_wrappers=1:nokey=1', path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        return float(result.stdout)

    audio_duration = get_audio_duration(audio_path)
    total_frames = int(fps * audio_duration)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))

    # Parse and validate size
    raw_size = character.get("size", 40)
    try:
        radius = int(raw_size)
    except Exception:
        radius = 40

    # Load avatar image
    avatar_path = character.get("avatar_path")

    # Initialize character state
        if not os.path.isabs(avatar_path):
            avatar_path = os.path.abspath(avatar_path)
        if os.path.exists(avatar_path):
            avatar_img = cv2.imread(avatar_path, cv2.IMREAD_UNCHANGED)
            if avatar_img is None:
                logging.warning(f"Failed to load avatar image at {avatar_path}")
                avatar_img = None
            else:
                logging.info(f"Successfully loaded avatar image at {avatar_path}")
        else:
            logging.warning(f"Avatar image path does not exist: {avatar_path}")
            avatar_img = None
    else:
        logging.info("No avatar_path provided for character")
        avatar_img = None
    state = {
        "name": character.get("name", "char"),
        "pos_x": 0,
        "pos_y": height // 2,
        "speed": width / total_frames if total_frames > 0 else width // (fps * 5),
        "radius": radius,
        "script": script_text
    }

    for frame_num in range(total_frames):
        # Create a colorful gradient background
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            color = (int(255 * y / height), int(128 * y / height), int(255 * (height - y) / height))
            img[y, :] = color

        # Calculate character position with looping movement
        center_x = int((state["pos_x"] + state["speed"] * frame_num) % (width + state["radius"]))
        center_y = state["pos_y"]

        if avatar_img is not None:
            # Resize avatar to fit radius
            scale = (2 * state["radius"]) / max(avatar_img.shape[0], avatar_img.shape[1])
            avatar_resized = cv2.resize(avatar_img, (0, 0), fx=scale, fy=scale)

            # Calculate top-left corner for avatar placement
            top_left_x = center_x - avatar_resized.shape[1] // 2
            top_left_y = center_y - avatar_resized.shape[0] // 2

            # Overlay avatar on background with alpha channel if present
            y1, y2 = max(0, top_left_y), min(height, top_left_y + avatar_resized.shape[0])
            x1, x2 = max(0, top_left_x), min(width, top_left_x + avatar_resized.shape[1])

            avatar_y1 = max(0, -top_left_y)
            avatar_y2 = avatar_y1 + (y2 - y1)
            avatar_x1 = max(0, -top_left_x)
            avatar_x2 = avatar_x1 + (x2 - x1)

            if avatar_resized.shape[2] == 4:
                alpha_avatar = avatar_resized[avatar_y1:avatar_y2, avatar_x1:avatar_x2, 3] / 255.0
                alpha_background = 1.0 - alpha_avatar

                for c in range(3):
                    img[y1:y2, x1:x2, c] = (alpha_avatar * avatar_resized[avatar_y1:avatar_y2, avatar_x1:avatar_x2, c] +
                                            alpha_background * img[y1:y2, x1:x2, c])
            else:
                img[y1:y2, x1:x2] = avatar_resized[avatar_y1:avatar_y2, avatar_x1:avatar_x2]
        else:
            # Draw simple cartoon character (smiley face) with customizable features
            face_color = (0, 255, 255)
            eye_color = (0, 0, 0)
            mouth_color = (0, 0, 0)
            eye_radius = state["radius"] // 5
            mouth_width = state["radius"] // 2
            mouth_height = state["radius"] // 4

            cv2.circle(img, (center_x, center_y), state["radius"], face_color, -1)  # Face
            cv2.circle(img, (center_x - state["radius"]//3, center_y - state["radius"]//4), eye_radius, eye_color, -1)  # Left eye
            cv2.circle(img, (center_x + state["radius"]//3, center_y - state["radius"]//4), eye_radius, eye_color, -1)  # Right eye
            cv2.ellipse(img, (center_x, center_y + state["radius"]//4), (mouth_width, mouth_height), 0, 0, 180, mouth_color, 3)  # Mouth

        cartoon = cartoonize_frame(img)
        out.write(cartoon)
    out.release()

    convert_avi_to_mp4(avi_path, mp4_path)
    merge_audio_video(mp4_path, audio_path, final_output_path)

    os.remove(avi_path)
    os.remove(mp4_path)
    os.remove(audio_path)

    return final_output_path

def merge_character_videos(video_paths, output_path):
    import logging
    # Build ffmpeg input arguments for filter_complex concat
    input_args = []
    for video_path in video_paths:
        input_args.extend(['-i', video_path])

    # Build filter_complex concat string
    filter_complex = f"{''.join([f'[{i}:v:0][{i}:a:0]' for i in range(len(video_paths))])}concat=n={len(video_paths)}:v=1:a=1[outv][outa]"

    command = [
        'ffmpeg', '-y',
        *input_args,
        '-filter_complex', filter_complex,
        '-map', '[outv]',
        '-map', '[outa]',
        '-c:v', 'libx264',
        '-c:a', 'aac',
        output_path
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(f"FFmpeg concat succeeded: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg concat failed with error: {e.stderr}")
        raise e


def generate_and_combine_synthesia_video(script, other_video_paths, output_path):
    """
    Generate a video using Synthesia AI video generator and combine it with other videos using ffmpeg.

    :param script: Text script for Synthesia video generation
    :param other_video_paths: List of file paths to other videos to combine
    :param output_path: Path to save the combined output video
    :return: Path to the combined output video
    """
    generator = AIVideoGenerator()
    synthesia_video_path = generator.generate_video(script)

    # Create a temporary file list for ffmpeg concat
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as file_list:
        # Add Synthesia video first
        file_list.write(f"file '{synthesia_video_path}'\n")
        # Add other videos
        for video_path in other_video_paths:
            file_list.write(f"file '{video_path}'\n")
        file_list_path = file_list.name

    # Run ffmpeg concat command
    command = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', file_list_path,
        '-c', 'copy',
        output_path
    ]
    subprocess.run(command, check=True)

    # Remove temporary file list
    os.remove(file_list_path)

    return output_path

def generate_synthesia_video_with_audio_merge(script, audio_path, output_path):
    """
    Generate a video using Synthesia AI video generator and merge it with provided audio using ffmpeg.

    :param script: Text script for Synthesia video generation
    :param audio_path: Path to the audio file to merge
    :param output_path: Path to save the merged output video
    :return: Path to the merged output video
    """
    generator = AIVideoGenerator()
    synthesia_video_path = generator.generate_video(script)

    merge_audio_video(synthesia_video_path, audio_path, output_path)

    return output_path

def generate_video_conditional(script, use_ai=True, video_id=None):
    """
    Generate a video either using AI video generation (Synthesia) or normal cartoon video using Python library.

    :param script: Text script for video generation
    :param use_ai: Boolean flag to choose AI video generation or normal cartoon video
    :param video_id: Optional video ID for naming output files in normal cartoon video generation
    :return: Path to the generated video
    """
    if use_ai:
        generator = AIVideoGenerator()
        return generator.generate_video(script)
    else:
        if video_id is None:
            raise ValueError("video_id must be provided for normal cartoon video generation")
        return generate_from_script(script, video_id)

def generate_from_script(script, video_id):
    """
    Basic implementation to generate a simple cartoon video from a script.
    This example generates a video with a moving colored circle and overlays the script text as subtitles.
    """
    import cv2
    import numpy as np
    import os

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, f"{video_id}_basic.mp4")

    width, height, fps, duration = 640, 480, 10, 5
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)
    thickness = 2
    line_type = cv2.LINE_AA

    # Split script into lines for subtitles
    lines = script.split('. ')
    lines = [line.strip() for line in lines if line.strip()]
    total_frames = fps * duration
    frames_per_line = max(1, total_frames // max(len(lines), 1))

    for frame_num in range(total_frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Moving circle position
        center_x = int((frame_num / total_frames) * width)
        center_y = height // 2
        radius = 40
        color = (0, 255, 255)

        cv2.circle(img, (center_x, center_y), radius, color, -1)

        # Determine which subtitle line to show
        line_index = min(frame_num // frames_per_line, len(lines) - 1) if lines else -1
        if line_index >= 0:
            text = lines[line_index]
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 50
            cv2.putText(img, text, (text_x, text_y), font, font_scale, font_color, thickness, line_type)

        out.write(img)

    out.release()
    return video_path

def generate_video_from_characters(characters, use_ai=True, video_id=None):
    """
    Unified function to generate video from character scripts using either AI or normal cartoon video generation.

    :param characters: List of character dicts with keys like 'name', 'color', 'size', 'script'
    :param use_ai: Boolean flag to choose AI video generation or normal cartoon video
    :param video_id: Optional video ID for naming output files in normal cartoon video generation
    :return: Path to the generated video file
    """
    combined_script = " ".join([char.get("script", "") for char in characters])
    if use_ai:
        generator = AIVideoGenerator()
        return generator.generate_video(combined_script)
    else:
        if video_id is None:
            raise ValueError("video_id must be provided for normal cartoon video generation")
        # Call the function that generates normal cartoon video from characters
        return generate_video_from_characters_normal(characters, use_ai=False, video_id=video_id)

def generate_video_from_characters_normal(characters, use_ai=False, video_id=None):
    """
    Actual implementation of normal cartoon video generation from characters.

    :param characters: List of character dicts
    :param use_ai: Should be False for normal cartoon video generation
    :param video_id: Video ID for output file naming
    :return: Path to generated video file
    """
    import uuid
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    video_paths = []
    for char in characters:
        unique_vid_id = str(uuid.uuid4())
        vid_path = generate_individual_character_video(char, unique_vid_id)
        video_paths.append(vid_path)
    merged_video_path = os.path.join(output_dir, f"{video_id}_merged.mp4")
    # Use new function to create tiled video layout
    tiled_video_path = os.path.join(output_dir, f"{video_id}_tiled.mp4")
    create_tiled_video(video_paths, tiled_video_path)
    return tiled_video_path

def create_tiled_video(input_videos, output_path):
    """
    Create a tiled layout video from multiple input videos using ffmpeg.

    :param input_videos: List of input video file paths
    :param output_path: Output video file path
    """
    import math
    import subprocess
    import logging

    num_videos = len(input_videos)
    if num_videos == 0:
        raise ValueError("No input videos provided for tiling")

    # Calculate grid size (rows and cols)
    cols = math.ceil(math.sqrt(num_videos))
    rows = math.ceil(num_videos / cols)

    # Prepare ffmpeg input arguments
    input_args = []
    for video in input_videos:
        input_args.extend(['-i', video])

    # Build filter_complex for tile layout
    filter_complex_parts = []
    for i in range(num_videos):
        filter_complex_parts.append(f"[{i}:v] setpts=PTS-STARTPTS, scale=320x240 [v{i}];")
    filter_complex_parts.append(f"{''.join([f'[v{i}]' for i in range(num_videos)])} xstack=inputs={num_videos}:layout=")

    layout = ""
    for i in range(num_videos):
        x = (i % cols) * 320
        y = (i // cols) * 240
        layout += f"{x}_{y}|"
    layout = layout.rstrip('|')

    filter_complex_parts[-1] += layout + " [v]"

    filter_complex = "".join(filter_complex_parts)

    command = [
        'ffmpeg', '-y',
        *input_args,
        '-filter_complex', filter_complex,
        '-map', '[v]',
        '-c:v', 'libx264',
        '-crf', '23',
        '-preset', 'veryfast',
        output_path
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(f"FFmpeg tiled video succeeded: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg tiled video failed with error: {e.stderr}")
        raise e
