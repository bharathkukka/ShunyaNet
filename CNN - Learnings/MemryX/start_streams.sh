#!/bin/bash
SERVER="rtsp://localhost:8554"
NUM=10

echo "Starting $NUM publishers in the background..."

for i in $(seq 1 $NUM); do
  ffmpeg -re -f lavfi -i "testsrc=size=640x480:rate=8" \
    -c:v libx264 -preset ultrafast -pix_fmt yuv420p -g 16 \
    -rtsp_transport tcp \
    -f rtsp "${SERVER}/stream${i}" \
    > /tmp/ffmpeg_stream${i}.log 2>&1 &

  sleep 0.05
done

echo "All $NUM publishers are running in the background."
