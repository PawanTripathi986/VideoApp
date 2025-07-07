from ai_video_generator import AIVideoGenerator

def main():
    script = "A short video showing a futuristic cityscape with flying cars and neon lights."
    generator = AIVideoGenerator()
    try:
        video_path = generator.generate_video(script)
        print(f"Video generated and saved to: {video_path}")
    except Exception as e:
        print(f"Error generating video: {e}")

if __name__ == "__main__":
    main()
