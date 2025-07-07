@echo off
REM Download ffmpeg Windows build zip file using PowerShell
powershell -Command "Invoke-WebRequest -Uri https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip -OutFile %USERPROFILE%\Downloads\ffmpeg-release-essentials.zip"

REM Extract the zip file to C:\ffmpeg
powershell -Command "Expand-Archive -Path %USERPROFILE%\Downloads\ffmpeg-release-essentials.zip -DestinationPath C:\ffmpeg -Force"

REM Add ffmpeg bin folder to system PATH (requires admin privileges)
setx /M PATH "%PATH%;C:\ffmpeg\ffmpeg-release-essentials\bin"

echo ffmpeg downloaded and PATH updated. Please restart your command prompt or PC for changes to take effect.
pause
