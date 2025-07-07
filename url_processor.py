import os, uuid, cv2
from yt_dlp import YoutubeDL
from script_generator import cartoonize_frame

def download_video(video_url, video_id):
    ydl_opts = {
        'outtmpl': f'output/{video_id}.mp4',
        'format': 'best[ext=mp4]',
        'quiet': True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    return f"output/{video_id}.mp4"

def cartoonize_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret: break
        cartoon = cartoonize_frame(frame)
        out.write(cartoon)
    cap.release()
    out.release()

def process_url(video_url, video_id):
    downloaded_path = download_video(video_url, video_id)
    cartoon_path = f"output/{video_id}_cartoon.mp4"
    cartoonize_video(downloaded_path, cartoon_path)
    return cartoon_path
