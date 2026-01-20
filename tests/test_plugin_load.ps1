$env:GST_PLUGIN_PATH += ";../builddir/src"

Write-Host "Testing plugin load..."
gst-inspect-1.0 faceblur

if ($?) {
    Write-Host "SUCCESS: Plugin loaded successfully" -ForegroundColor Green
} else {
    Write-Host "FAILURE: Could not load plugin" -ForegroundColor Red
    exit 1
}
