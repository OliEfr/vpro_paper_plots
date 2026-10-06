#!/bin/bash
# Submission-size version of the companion video: H.264, two-pass, sized to fit under 10 MB.
# Keeps 1920x1080 @ 30 fps (the content is mostly white canvas, which costs nothing).
#   ./make_submission.sh [target_MB=9.5]
set -euo pipefail
cd "$(dirname "$0")"
IN=slides/paper_video.mp4
OUT=slides/paper_video_submission.mp4
TARGET_MB=${1:-9.5}
DUR=$(/usr/bin/ffprobe -v error -show_entries format=duration -of csv=p=0 "$IN")
# total kbit/s for the target size (no audio track)
KBPS=$(python3 -c "print(int($TARGET_MB*8*1024*1024/1000/$DUR))")
echo "duration ${DUR}s -> ${KBPS} kbit/s"
/usr/bin/ffmpeg -v error -y -i "$IN" -an -c:v libx264 -preset slow -b:v ${KBPS}k -pass 1 -passlogfile /tmp/paper_video_2pass -pix_fmt yuv420p -f mp4 /dev/null
/usr/bin/ffmpeg -v error -y -i "$IN" -an -c:v libx264 -preset slow -b:v ${KBPS}k -pass 2 -passlogfile /tmp/paper_video_2pass -pix_fmt yuv420p -movflags +faststart "$OUT"
rm -f /tmp/paper_video_2pass*
ls -la "$OUT"
