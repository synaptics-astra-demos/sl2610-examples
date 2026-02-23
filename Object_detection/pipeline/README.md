# Synaptics Astra SL2610 Object Detection Pipeline Simple Example

Create a simple vison AI application using Gstreamer. Compile the model with Torq. Wrap in SyNAP format to use the GStreamer plugin to integrate into the Vision Application pipeline.

Step 1: Download Quantized Model 

Example: YOLOv8n model, Integer, 320x320 from https://huggingface.co/Synaptics/yolo/tree/main 


Step 2: Get sample Synap model structure

See `sample` folder. 

Step 3: Enter Docker Container

See Torq documentation for details. 

```
docker run --rm -it -v $(pwd):$(pwd) -w $(pwd) -u $(id -u):$(id -g) ghcr.io/synaptics-torq/torq-compiler/compiler:main
```

Step 4: Convert Model (TFLite → TOSA) 

```
iree-import-tflite yolov8n_full_integer_quant_320_od.tflite -o yolov8-od.tosa  
```

Step 5: Convert binary MLIR to text MLIR (Optional)

```
iree-opt yolov8-od.tosa -o yolov8-od.mlir
```

Step 6: Compile TOSA → Torq Model

```
torq-compile yolov8-od.mlir -o yolov8-od.vmfb --torq-hw=SL2610 \
--torq-target-host-cpu=generic \
--torq-target-host-cpu-features="+neon,+crypto,+crc,+dotprod,+rdm,+rcpc,+lse" \
--torq-target-host-triple=aarch64-unknown-linux-gnu \
--torq-convert-table-to-gather \
--torq-mul-cast-i32-to-i16 \
--torq-disable-css \
--torq-disable-slices
```

Step 6: Create Synap Model

```
cp -r sl2610-examples/Object_detection/pipeline/sample  sample
cp yolov8-od.vmfb sample/0/subgraph_0.vmfb 
cd sample
zip -r model.synap 0 bundle.json
mv model.synap ../
exit
```

Step 7: Copy model.synap to RDK


Step 8: Set Environment Variables for Wayland

In a terminal on the Astra.

```
export XDG_RUNTIME_DIR=/var/run/user/0
export WESTON_DISABLE_GBM_MODIFIERS=true
export WAYLAND_DISPLAY=wayland-1
export QT_QPA_PLATFORM=wayland
```

Step 9: Check your video device number


Step 10: Use this GStreamer pipeline

```
root@sl2619:~# gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw,framerate=30/1, format=YUY2, width=640, height=480 ! tee name=t_data t_data. ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! synavideoconvertscale ! video/x-raw,width=320,height=320,format=RGB ! synapinfer model=/home/root/model.synap mode=detector frameinterval=3 ! overlay.inference_sink t_data. ! queue ! synavideoconvertscale ! synapoverlay name=overlay label=/usr/share/synap/models/object_detection/coco/info.json ! waylandsink
```

You should see a window with the camera video and bounding boxes over detected objects. 