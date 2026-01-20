#!/bin/bash

# Configuration
PLUGIN_PATH="../builddir/src"
export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PLUGIN_PATH

echo "Testing plugin load..."
gst-inspect-1.0 faceblur

if [ $? -eq 0 ]; then
    echo "SUCCESS: Plugin loaded successfully"
else
    echo "FAILURE: Could not load plugin"
    exit 1
fi
