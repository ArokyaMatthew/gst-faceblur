plugin_c_args = ['-DHAVE_CONFIG_H']

cdata = configuration_data()
cdata.set_quoted('PACKAGE_VERSION', meson.project_version())
cdata.set_quoted('PACKAGE', 'gst-faceblur')
cdata.set_quoted('GST_LICENSE', 'LGPL')
cdata.set_quoted('GST_API_VERSION', '1.0')
cdata.set_quoted('GST_PACKAGE_NAME', 'GStreamer Face Blur Plugin')
cdata.set_quoted('GST_PACKAGE_ORIGIN', 'https://github.com/ArokyaMatthew/gst-faceblur')
configure_file(output : 'config.h', configuration : cdata)

gstfaceblur_sources = [
  'gstfaceblur.c'
]

gstfaceblur = shared_library('gstfaceblur',
  gstfaceblur_sources,
  c_args : plugin_c_args,
  dependencies : [gst_dep, gstbase_dep, gstvideo_dep, onnx_dep],
  install : true,
  install_dir : plugins_install_dir,
)
