# 🎭 GStreamer Face Blur Plugin

**Real-time face detection and privacy-grade pixelation for GStreamer using ONNX Runtime**

[![License: LGPL v2.1](https://img.shields.io/badge/License-LGPL_v2.1-blue.svg)](https://www.gnu.org/licenses/lgpl-2.1)
[![GStreamer](https://img.shields.io/badge/GStreamer-1.22+-brightgreen.svg)](https://gstreamer.freedesktop.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20|%20Windows-lightgrey.svg)]()

---

## 📖 About This Project

The **GStreamer Face Blur Plugin** is a privacy-focused video processing element that automatically detects human faces in video streams and applies a mosaic (pixelation) effect to protect identities. 

This plugin is perfect for:
- 🎥 **Content creators** who need to blur faces in street footage
- 🏢 **Businesses** processing CCTV or surveillance footage for GDPR compliance
- 📱 **App developers** building privacy-aware video applications
- 🔬 **Researchers** anonymizing video datasets

### How It Works

1. Each video frame is preprocessed and resized for the AI model
2. The **UltraFace-RFB-320** neural network detects all faces in the frame
3. **Non-Maximum Suppression (NMS)** removes duplicate detections
4. Detected face regions are pixelated with a configurable mosaic effect
5. The processed frame continues through your GStreamer pipeline

---

## 🎬 Demo

https://github.com/user-attachments/assets/3c9f24d1-26d9-4130-bdf0-bbffb52866c9

> *The plugin processes video in real-time, detecting and blurring all faces automatically.*

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🚀 Real-time Processing** | Optimized for live video streams and file processing |
| **🎯 Accurate Detection** | Uses UltraFace-RFB-320 ONNX model (1.2MB, fast inference) |
| **🔒 Privacy Protection** | Configurable mosaic block size (4-64 pixels) |
| **🧠 Smart NMS** | Non-Maximum Suppression eliminates overlapping detections |
| **🧵 Thread-Safe** | Safe for use in complex multi-threaded pipelines |
| **🎨 Multiple Formats** | Supports RGBA, RGBx, BGRA, BGRx, ARGB, ABGR |

---

## 📋 Requirements

Before building, make sure you have these installed:

| Dependency | Minimum Version | Purpose |
|------------|-----------------|---------|
| GStreamer | 1.22+ | Multimedia framework |
| GStreamer Plugins Base | 1.22+ | Video processing support |
| ONNX Runtime | 1.10+ | AI inference engine |
| Meson | 0.60+ | Build system |
| Ninja | 1.10+ | Build backend |
| C Compiler | GCC 9+ / Clang 10+ / MSVC 2019+ | Compilation |

---

## 🔧 Installation

### Step 1: Install System Dependencies

<details open>
<summary><b>🐧 Ubuntu / Debian Linux</b></summary>

```bash
# Update package list
sudo apt update

# Install GStreamer development libraries
sudo apt install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-tools

# Install build tools
sudo apt install -y meson ninja-build build-essential wget
```

</details>

<details>
<summary><b>🎩 Fedora / RHEL / CentOS</b></summary>

```bash
# Install GStreamer development libraries
sudo dnf install -y \
    gstreamer1-devel \
    gstreamer1-plugins-base-devel \
    gstreamer1-plugins-good

# Install build tools
sudo dnf install -y meson ninja-build gcc wget
```

</details>

<details>
<summary><b>🪟 Windows</b></summary>

1. **Install GStreamer:**
   - Download from [gstreamer.freedesktop.org](https://gstreamer.freedesktop.org/download/)
   - Choose **MinGW 64-bit** → Install both **Runtime** and **Development** packages
   - Add `C:\gstreamer\1.0\mingw_x86_64\bin` to your PATH

2. **Install Build Tools:**
   - Install [Python 3.8+](https://www.python.org/downloads/)
   - Run: `pip install meson ninja`

3. **Install Visual Studio Build Tools** (or MinGW-w64)

</details>

### Step 2: Install ONNX Runtime

<details open>
<summary><b>🐧 Linux</b></summary>

```bash
# Download ONNX Runtime (adjust version as needed)
cd ~
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz

# Extract
tar -xzf onnxruntime-linux-x64-1.16.3.tgz

# Set environment variable (add to ~/.bashrc for persistence)
export ONNX_RUNTIME_DIR=$HOME/onnxruntime-linux-x64-1.16.3
```

</details>

<details>
<summary><b>🪟 Windows</b></summary>

1. Download from [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases)
2. Extract to a folder like `C:\onnxruntime`
3. Note the path for the build step

</details>

### Step 3: Build the Plugin

```bash
# Clone the repository
git clone https://github.com/ArokyaMatthew/gst-faceblur.git
cd gst-faceblur

# Configure the build (Linux)
meson setup builddir -Donnxruntime_path=$ONNX_RUNTIME_DIR

# Or on Windows:
# meson setup builddir -Donnxruntime_path=C:\onnxruntime

# Compile
meson compile -C builddir
```

### Step 4: Download the Face Detection Model

```bash
# Download UltraFace-RFB-320 model
wget https://github.com/onnx/models/raw/main/validated/vision/body_analysis/ultraface/models/version-RFB-320.onnx
```

### Step 5: Test the Installation

```bash
# Set plugin path
export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$(pwd)/builddir/src

# Verify plugin is loaded
gst-inspect-1.0 faceblur
```

If successful, you'll see the plugin properties and capabilities printed to the terminal.

---

## 🚀 Usage Examples

### Basic: Display Video with Blurred Faces

```bash
# Set plugin path (required each session, or add to ~/.bashrc)
export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:/path/to/gst-faceblur/builddir/src

# Process a video file and display
gst-launch-1.0 filesrc location=input.mp4 ! decodebin ! videoconvert ! \
    video/x-raw,format=RGBA ! \
    faceblur model-location=/path/to/version-RFB-320.onnx ! \
    videoconvert ! autovideosink
```

### Save Processed Video to File

```bash
gst-launch-1.0 filesrc location=input.mp4 ! decodebin ! videoconvert ! \
    video/x-raw,format=RGBA ! \
    faceblur model-location=version-RFB-320.onnx ! \
    videoconvert ! x264enc ! mp4mux ! filesink location=output_blurred.mp4
```

### Live Webcam with Face Blur

```bash
# Linux (V4L2)
gst-launch-1.0 v4l2src ! videoconvert ! video/x-raw,format=RGBA ! \
    faceblur model-location=version-RFB-320.onnx ! \
    videoconvert ! autovideosink

# Windows (DirectShow)
gst-launch-1.0 ksvideosrc ! videoconvert ! video/x-raw,format=RGBA ! \
    faceblur model-location=version-RFB-320.onnx ! \
    videoconvert ! autovideosink
```

### Customize Blur Strength

```bash
# Stronger pixelation (larger blocks)
faceblur model-location=model.onnx mosaic-size=24

# Subtle pixelation (smaller blocks)
faceblur model-location=model.onnx mosaic-size=6
```

### Adjust Detection Sensitivity

```bash
# High precision (fewer false positives, might miss some faces)
faceblur model-location=model.onnx confidence-threshold=0.85

# High recall (catches more faces, might have false positives)
faceblur model-location=model.onnx confidence-threshold=0.5
```

---

## ⚙️ Plugin Properties & Customization

The faceblur plugin has **three configurable properties** that you can adjust to fine-tune the blur effect:

### Properties Reference

| Property | Type | Default | Range | Description |
|----------|------|---------|-------|-------------|
| `model-location` | string | *(required)* | — | Path to the ONNX face detection model file |
| `mosaic-size` | integer | **12** | 4 – 64 | Size of pixelation blocks (larger = stronger blur) |
| `confidence-threshold` | float | **0.7** | 0.0 – 1.0 | Detection sensitivity (lower = detects more faces) |

---

### 🎨 Customizing the Blur Effect

#### Mosaic Size (Pixelation Strength)

The `mosaic-size` property controls how "blocky" the pixelation appears:

| Value | Effect | Best For |
|-------|--------|----------|
| `4-8` | Subtle blur, face shape still visible | Light anonymization |
| `12` (default) | Balanced - good privacy protection | General use |
| `16-24` | Strong blur, face unrecognizable | High privacy needs |
| `32-64` | Very heavy pixelation | Maximum anonymization |

**Examples:**
```bash
# Light blur - preserves some detail
gst-launch-1.0 ... ! faceblur model-location=model.onnx mosaic-size=6 ! ...

# Strong blur - complete anonymization
gst-launch-1.0 ... ! faceblur model-location=model.onnx mosaic-size=24 ! ...

# Maximum blur - very blocky
gst-launch-1.0 ... ! faceblur model-location=model.onnx mosaic-size=48 ! ...
```

---

#### Confidence Threshold (Detection Sensitivity)

The `confidence-threshold` property controls how certain the AI must be before blurring a face:

| Value | Behavior | Use Case |
|-------|----------|----------|
| `0.3-0.5` | Detects more faces (may include false positives) | Don't miss any face |
| `0.7` (default) | Balanced detection accuracy | General use |
| `0.8-0.9` | Only very confident detections | Reduce false positives |

**Examples:**
```bash
# Catch all possible faces (more aggressive)
gst-launch-1.0 ... ! faceblur model-location=model.onnx confidence-threshold=0.5 ! ...

# Only blur faces AI is very sure about
gst-launch-1.0 ... ! faceblur model-location=model.onnx confidence-threshold=0.85 ! ...
```

---

#### Combining Parameters

You can combine both parameters for fine-tuned control:

```bash
# Strong blur + aggressive detection (maximum privacy)
gst-launch-1.0 filesrc location=video.mp4 ! decodebin ! videoconvert ! \
    video/x-raw,format=RGBA ! \
    faceblur model-location=version-RFB-320.onnx \
             mosaic-size=24 \
             confidence-threshold=0.5 ! \
    videoconvert ! x264enc ! mp4mux ! filesink location=output.mp4

# Light blur + precise detection (subtle effect)
gst-launch-1.0 filesrc location=video.mp4 ! decodebin ! videoconvert ! \
    video/x-raw,format=RGBA ! \
    faceblur model-location=version-RFB-320.onnx \
             mosaic-size=8 \
             confidence-threshold=0.8 ! \
    videoconvert ! x264enc ! mp4mux ! filesink location=output.mp4
```

---

## 📁 Project Structure

```
gst-faceblur/
├── src/
│   ├── gstfaceblur.c       # Main plugin implementation (1200+ lines)
│   ├── gstfaceblur.h       # Public header with type definitions
│   └── meson.build         # Source build configuration
├── tests/
│   ├── test_plugin_load.sh     # Linux: verify plugin loads
│   ├── test_plugin_load.ps1    # Windows: verify plugin loads
│   └── test_pipeline.sh        # Test a simple pipeline
├── demo/
│   └── people_blurred.mp4      # Example output video
├── meson.build             # Main build configuration
├── meson_options.txt       # Build options (ONNX Runtime path)
├── LICENSE                 # LGPL-2.1 license
└── README.md               # This file
```

---

## 🧪 Running Tests

```bash
# Linux
cd tests
chmod +x *.sh
./test_plugin_load.sh

# Windows PowerShell
cd tests
.\test_plugin_load.ps1
```

---

## 🔬 Technical Details

### Face Detection Model

- **Model**: [UltraFace-RFB-320](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/ultraface)
- **Size**: 1.2 MB (lightweight, fast inference)
- **Input**: 320×240 RGB image (normalized to [-1, 1])
- **Output**: 4420 anchor boxes with confidence scores

### Processing Pipeline

1. **Preprocessing**: Resize frame to 320×240, normalize pixels, convert to CHW format
2. **Inference**: Run ONNX Runtime with the UltraFace model
3. **Postprocessing**: Filter by confidence threshold, apply NMS (IoU=0.3)
4. **Pixelation**: Average color blocks for each detected face region

### Performance

- Processes 30+ FPS on modern CPUs (tested on Intel i7, AMD Ryzen)
- GPU acceleration available via ONNX Runtime CUDA/DirectML providers (not included by default)

---

## 🐛 Troubleshooting

**Plugin not found:**
```bash
# Make sure GST_PLUGIN_PATH includes the build directory
export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:/path/to/gst-faceblur/builddir/src
```

**Model not loading:**
- Ensure the model path is absolute or relative to the working directory
- Check that `version-RFB-320.onnx` exists and is readable

**No faces detected:**
- Lower the `confidence-threshold` (e.g., 0.5)
- Ensure faces are reasonably visible and not too small

**Build errors:**
- Verify GStreamer development libraries are installed
- Check ONNX Runtime path is correct

---

## 📄 License

This project is licensed under the **GNU Lesser General Public License v2.1 (LGPL-2.1)**.

You are free to:
- Use this plugin in commercial and non-commercial projects
- Modify the source code
- Distribute the plugin

See [LICENSE](LICENSE) for full details.

---

## 👨‍💻 Author

**Arokya Matthew Nathan**

- 📧 Email: [arokyamatthewnathan@gmail.com](mailto:arokyamatthewnathan@gmail.com)
- 🔗 GitHub: [@ArokyaMatthew](https://github.com/ArokyaMatthew)

---

## 🙏 Acknowledgments

This project builds on the excellent work of:

- **[UltraFace](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)** — Lightweight face detection model
- **[ONNX Runtime](https://onnxruntime.ai/)** — High-performance machine learning inference
- **[GStreamer](https://gstreamer.freedesktop.org/)** — Powerful multimedia framework

---

## ⭐ Support

If you find this project useful, please consider:
- ⭐ Starring the repository
- 🐛 Reporting bugs or suggesting features
- 🔀 Contributing pull requests

---

*Made with ❤️ for the privacy-conscious developer community*
