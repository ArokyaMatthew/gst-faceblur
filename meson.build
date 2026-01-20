project('gst-faceblur', 'c',
  version : '1.0.0',
  license : 'LGPL',
  default_options : [
    'warning_level=2',
    'c_std=gnu11',
    'buildtype=debugoptimized'
  ]
)

cc = meson.get_compiler('c')

gst_version = '>= 1.22.0'
gst_dep = dependency('gstreamer-1.0', version : gst_version,
  required : true, fallback : ['gstreamer', 'gst_dep'])
gstbase_dep = dependency('gstreamer-base-1.0', version : gst_version,
  required : true, fallback : ['gstreamer', 'gst_base_dep'])
gstvideo_dep = dependency('gstreamer-video-1.0', version : gst_version,
  required : true, fallback : ['gst-plugins-base', 'gst_video_dep'])

# ONNX Runtime dependency
onnxruntime_path = get_option('onnxruntime_path')
if onnxruntime_path != ''
  # Custom ONNX Runtime location
  onnx_inc = include_directories(onnxruntime_path / 'include')
  
  # Check for different library naming conventions (windows vs linux vs mac)
  onnx_lib = cc.find_library('onnxruntime', dirs : onnxruntime_path / 'lib', required : false)
  if not onnx_lib.found()
    onnx_lib = cc.find_library('onnxruntime.dll', dirs : onnxruntime_path / 'bin', required : false)
  endif
  
  if not onnx_lib.found()
    error('Could not find onnxruntime library in ' + onnxruntime_path)
  endif
  
  onnx_dep = declare_dependency(
    include_directories : onnx_inc,
    dependencies : onnx_lib
  )
else
  # System ONNX Runtime - try pkg-config first, then fallback to library search
  onnx_dep = dependency('libonnxruntime', required : false)
  if not onnx_dep.found()
    onnx_dep = cc.find_library('onnxruntime', required : true)
  endif
endif

plugins_install_dir = join_paths(get_option('libdir'), 'gstreamer-1.0')

subdir('src')
