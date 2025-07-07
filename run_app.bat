@echo off
REM Run this script to set up environment and start the cartoonizer app

@REM echo Setting up ffmpeg path...
@REM call install_ffmpeg_and_set_path.bat

@REM echo Setting OpenAI API key...
@REM call set_openai_api_key.bat

echo Installing Python dependencies...
pip install -r requirements.txt

echo Starting the Flask app...
python app.py

pause
