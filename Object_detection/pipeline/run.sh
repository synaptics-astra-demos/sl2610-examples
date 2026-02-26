#!/bin/bash

echo "Run the object detection pipeline!"

export XDG_RUNTIME_DIR=/var/run/user/0
export WESTON_DISABLE_GBM_MODIFIERS=true
export WAYLAND_DISPLAY=wayland-1
export QT_QPA_PLATFORM=wayland


# Define the name to look for 
# Most USB cameras use the 'uvcvideo' driver.
TARGET_DRIVER="uvcvideo"

# Iterate through all video devices in sysfs to find the match
for dev in /sys/class/video4linux/video*; do
    if [ -e "$dev/device/driver" ]; then
        DRIVER_NAME=$(basename $(readlink "$dev/device/driver"))
        
        if [ "$DRIVER_NAME" = "$TARGET_DRIVER" ]; then
            # Get the device node name (e.g., video0)
            NODE_NAME=$(basename "$dev")
            export CAMERA_DEV="/dev/$NODE_NAME"
            break
        fi
    fi
done

# Check if we found it
if [ -z "$CAMERA_DEV" ]; then
    echo "Error: No USB camera found with driver $TARGET_DRIVER" >&2
    exit 1
else
    echo "Found USB camera at: $CAMERA_DEV"
    # To use this in your current shell, you must 'source' the script
fi


gst-launch-1.0 v4l2src device=$CAMERA_DEV ! video/x-raw,framerate=30/1, format=YUY2, width=640, height=480 ! tee name=t_data t_data. ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! synavideoconvertscale ! video/x-raw,width=320,height=320,format=RGB ! synapinfer model=/home/root/model.synap mode=detector frameinterval=3 ! overlay.inference_sink t_data. ! queue ! synavideoconvertscale ! synapoverlay name=overlay label=/usr/share/synap/models/object_detection/coco/info.json ! waylandsink


