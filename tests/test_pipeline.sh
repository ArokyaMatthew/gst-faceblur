#!/bin/bash

# Check if model exists
if [ ! -f "version-RFB-320.onnx" ]; then
    echo "Downloading model..."
    wget https://github.com/onnx/models/raw/main/validated/vision/body_analysis/ultraface/models/version-RFB-320.onnx
fi

export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:../builddir/src

echo "Running test pipeline..."
gst-launch-1.0 videotestsrc num-buffers=30 ! video/x-raw,format=RGBA ! \
    faceblur model-location=version-RFB-320.onnx ! \
    videoconvert ! fakesink

if [ $? -eq 0 ]; then
    echo "SUCCESS: Pipeline ran successfully"
else
    echo "FAILURE: Pipeline failed"
    exit 1
fi
