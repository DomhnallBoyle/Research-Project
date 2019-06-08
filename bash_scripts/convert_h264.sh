# script to convert .h264 files to mp4 video files

ffmpeg -r $1 -i pi1.h264 -c copy pi1.mp4
ffmpeg -r $1 -i pi2.h264 -c copy pi2.mp4
ffmpeg -r $1 -i pi3.h264 -c copy pi3.mp4