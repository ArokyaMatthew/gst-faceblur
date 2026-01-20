/*
 * GStreamer Face Blur Plugin
 * Copyright (C) 2026 Arokya Matthew Nathan <arokyamatthewnathan@gmail.com>
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * Face detection and pixelation using ONNX Runtime.
 * Compatible with UltraFace-RFB-320 model.
 */

/**
 * SECTION:element-faceblur
 * @title: faceblur
 * @short_description: Real-time face detection and pixelation filter
 * @see_also: #GstVideoFilter, #GstBaseTransform
 *
 * The faceblur element performs real-time face detection using ONNX Runtime
 * and applies pixelation (mosaic) effect to detected faces for privacy
 * protection.
 *
 * The element uses the UltraFace-RFB-320 ONNX model for face detection,
 * which provides a good balance between speed and accuracy. Non-Maximum
 * Suppression (NMS) is applied to filter overlapping detections.
 *
 * ## Supported Formats
 *
 * The element works with 32-bit RGBA-family video formats:
 * RGBA, RGBx, BGRA, BGRx, ARGB, ABGR
 *
 * ## Example Pipeline
 *
 * |[
 * gst-launch-1.0 filesrc location=video.mp4 ! decodebin ! videoconvert ! \
 *     video/x-raw,format=RGBA ! \
 *     faceblur model-location=/path/to/version-RFB-320.onnx ! \
 *     videoconvert ! autovideosink
 * ]|
 *
 * This pipeline will decode a video file and apply face blur to all
 * detected faces before displaying the result.
 *
 * ## Properties
 *
 * - model-location: Path to the ONNX face detection model (required)
 * - mosaic-size: Size of pixelation blocks in pixels (4-64, default: 12)
 * - confidence-threshold: Minimum confidence for detection (0.0-1.0, default: 0.7)
 *
 * Since: 1.0
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gstfaceblur.h"
#include <onnxruntime_c_api.h>
#include <string.h>

/*      Debug Category     */

GST_DEBUG_CATEGORY_STATIC (gst_faceblur_debug);
#define GST_CAT_DEFAULT gst_faceblur_debug

/*   Model Configuration     */

/**
 * MODEL_W:
 *
 * Input width expected by the UltraFace-RFB-320 ONNX model.
 * The model was trained with 320x240 input images.
 */
#define MODEL_W 320

/**
 * MODEL_H:
 *
 * Input height expected by the UltraFace-RFB-320 ONNX model.
 */
#define MODEL_H 240

/**
 * NUM_DETECTIONS:
 *
 * Number of detection anchors output by the UltraFace model.
 * This is determined by the model architecture (4420 anchors).
 */
#define NUM_DETECTIONS 4420

/**
 * MAX_FACES:
 *
 * Maximum number of faces that can be tracked per frame.
 * This limit prevents excessive memory usage and processing time.
 */
#define MAX_FACES 64

/**
 * NMS_IOU_THRESHOLD:
 *
 * Intersection over Union (IoU) threshold for Non-Maximum Suppression.
 * Lower values result in fewer overlapping detections.
 * Range: 0.0 (aggressive suppression) to 1.0 (no suppression)
 */
#define NMS_IOU_THRESHOLD 0.3f


/*   Properties   */

/**
 * GstFaceBlurProperties:
 * @PROP_0: Reserved for GObject
 * @PROP_MODEL_LOCATION: Path to ONNX model file
 * @PROP_MOSAIC_SIZE: Pixelation block size in pixels
 * @PROP_CONFIDENCE_THRESHOLD: Minimum detection confidence
 *
 * Properties exposed by the faceblur element.
 */
enum {
  PROP_0,
  PROP_MODEL_LOCATION,
  PROP_MOSAIC_SIZE,
  PROP_CONFIDENCE_THRESHOLD
};

#define DEFAULT_MOSAIC_SIZE 12
#define DEFAULT_CONFIDENCE_THRESHOLD 0.7f

/* Static GParamSpecs for property notifications */
static GParamSpec *properties[PROP_CONFIDENCE_THRESHOLD + 1];

/*  Data Structures   */

/**
 * FaceBox:
 * @x1: Left coordinate (normalized 0.0-1.0)
 * @y1: Top coordinate (normalized 0.0-1.0)
 * @x2: Right coordinate (normalized 0.0-1.0)
 * @y2: Bottom coordinate (normalized 0.0-1.0)
 * @confidence: Detection confidence score (0.0-1.0)
 *
 * Represents a detected face bounding box with normalized coordinates.
 * Coordinates are relative to the image dimensions.
 */
typedef struct {
  gfloat x1, y1, x2, y2;
  gfloat confidence;
} FaceBox;

/**
 * GstFaceBlur:
 *
 * The opaque #GstFaceBlur data structure.
 */
typedef struct _GstFaceBlur {
  GstBaseTransform parent;

  /*< private >*/

  /* Properties */
  gchar *model_location;
  gint mosaic_size;
  gfloat confidence_threshold;

  /* Video info */
  GstVideoInfo video_info;
  gboolean video_info_set;

  /* Channel order for current format (offset in bytes) */
  gint r_offset;
  gint g_offset;
  gint b_offset;

  /* ONNX Runtime resources */
  const OrtApi *ort;
  OrtEnv *env;
  OrtSession *session;
  OrtMemoryInfo *meminfo;

  /* Input tensor */
  float *input_tensor;
  OrtValue *input_value;

  /* Detection buffer */
  FaceBox *detections;

  /* ONNX session lock for thread safety */
  GMutex ort_lock;
  gboolean model_loaded;
} GstFaceBlur;

/**
 * GstFaceBlurClass:
 *
 * The #GstFaceBlur class structure.
 */
typedef struct _GstFaceBlurClass {
  GstBaseTransformClass parent_class;
} GstFaceBlurClass;

G_DEFINE_TYPE (GstFaceBlur, gst_face_blur, GST_TYPE_BASE_TRANSFORM)

/*    Forward Declarations    */

static void gst_face_blur_set_property (GObject *obj, guint prop_id,
    const GValue *value, GParamSpec *pspec);
static void gst_face_blur_get_property (GObject *obj, guint prop_id,
    GValue *value, GParamSpec *pspec);
static void gst_face_blur_finalize (GObject *obj);
static gboolean gst_face_blur_start (GstBaseTransform *bt);
static gboolean gst_face_blur_stop (GstBaseTransform *bt);
static gboolean gst_face_blur_set_caps (GstBaseTransform *bt,
    GstCaps *incaps, GstCaps *outcaps);
static GstFlowReturn gst_face_blur_transform_ip (GstBaseTransform *bt,
    GstBuffer *buf);

/*    Format Configuration    */

/**
 * gst_face_blur_configure_format:
 * @self: the #GstFaceBlur instance
 * @format: the #GstVideoFormat to configure
 *
 * Configures RGB channel offsets based on the negotiated video format.
 * This allows the element to work with various RGBA-family formats
 * without format-specific code paths.
 *
 * Returns: %TRUE on success, %FALSE if format is unsupported
 */
static gboolean
gst_face_blur_configure_format (GstFaceBlur *self, GstVideoFormat format)
{
  switch (format) {
    case GST_VIDEO_FORMAT_RGBA:
    case GST_VIDEO_FORMAT_RGBx:
      self->r_offset = 0;
      self->g_offset = 1;
      self->b_offset = 2;
      break;
    case GST_VIDEO_FORMAT_BGRA:
    case GST_VIDEO_FORMAT_BGRx:
      self->r_offset = 2;
      self->g_offset = 1;
      self->b_offset = 0;
      break;
    case GST_VIDEO_FORMAT_ARGB:
      self->r_offset = 1;
      self->g_offset = 2;
      self->b_offset = 3;
      break;
    case GST_VIDEO_FORMAT_ABGR:
      self->r_offset = 3;
      self->g_offset = 2;
      self->b_offset = 1;
      break;
    default:
      GST_ERROR_OBJECT (self, "Unsupported video format: %s",
                        gst_video_format_to_string (format));
      return FALSE;
  }

  GST_DEBUG_OBJECT (self, "Format %s: R=%d, G=%d, B=%d",
                    gst_video_format_to_string (format),
                    self->r_offset, self->g_offset, self->b_offset);
  return TRUE;
}

/*   Non-Maximum Suppression (NMS)   */

/**
 * compute_iou:
 * @a: first bounding box
 * @b: second bounding box
 *
 * Computes Intersection over Union (IoU) between two bounding boxes.
 * IoU is a standard metric for measuring overlap between boxes and
 * is used by NMS to filter redundant detections.
 *
 * Formula: IoU = intersection_area / union_area
 *
 * Returns: IoU value between 0.0 (no overlap) and 1.0 (identical boxes)
 */
static inline gfloat
compute_iou (const FaceBox *a, const FaceBox *b)
{
  gfloat x1 = MAX (a->x1, b->x1);
  gfloat y1 = MAX (a->y1, b->y1);
  gfloat x2 = MIN (a->x2, b->x2);
  gfloat y2 = MIN (a->y2, b->y2);

  gfloat inter_w = MAX (0.0f, x2 - x1);
  gfloat inter_h = MAX (0.0f, y2 - y1);
  gfloat inter_area = inter_w * inter_h;

  gfloat area_a = (a->x2 - a->x1) * (a->y2 - a->y1);
  gfloat area_b = (b->x2 - b->x1) * (b->y2 - b->y1);
  gfloat union_area = area_a + area_b - inter_area;

  if (union_area <= 0.0f)
    return 0.0f;

  return inter_area / union_area;
}

/**
 * compare_boxes_by_confidence:
 * @a: pointer to first FaceBox
 * @b: pointer to second FaceBox
 *
 * Comparison function for qsort. Sorts boxes in descending order
 * by confidence score (highest confidence first).
 *
 * Returns: -1 if a > b, 1 if a < b, 0 if equal
 */
static int
compare_boxes_by_confidence (const void *a, const void *b)
{
  const FaceBox *box_a = (const FaceBox *) a;
  const FaceBox *box_b = (const FaceBox *) b;

  if (box_a->confidence > box_b->confidence) return -1;
  if (box_a->confidence < box_b->confidence) return 1;
  return 0;
}

/**
 * nms_filter:
 * @boxes: array of FaceBox detections (modified in-place)
 * @count: number of detections in the array
 * @iou_threshold: IoU threshold for suppression
 *
 * Applies Non-Maximum Suppression to filter overlapping detections.
 * NMS works by:
 * 1. Sorting boxes by confidence (descending)
 * 2. Keeping the highest-confidence box
 * 3. Suppressing boxes that overlap significantly with kept boxes
 * 4. Repeating until all boxes are processed
 *
 * The result is compacted into the beginning of the array.
 *
 * Returns: number of remaining detections after NMS
 */
static gint
nms_filter (FaceBox *boxes, gint count, gfloat iou_threshold)
{
  gint i, j;
  gboolean *suppressed;
  gint result_count = 0;

  if (count <= 1)
    return count;

  /* Sort by confidence (descending) */
  qsort (boxes, count, sizeof (FaceBox), compare_boxes_by_confidence);

  /* Allocate suppression flags on stack */
  suppressed = g_alloca (count * sizeof (gboolean));
  memset (suppressed, 0, count * sizeof (gboolean));

  for (i = 0; i < count; i++) {
    if (suppressed[i])
      continue;

    /* Keep this box */
    if (result_count != i) {
      boxes[result_count] = boxes[i];
    }

    /* Suppress overlapping boxes */
    for (j = i + 1; j < count; j++) {
      if (!suppressed[j] &&
          compute_iou (&boxes[result_count], &boxes[j]) > iou_threshold) {
        suppressed[j] = TRUE;
      }
    }

    result_count++;
  }

  return result_count;
}

/* Pixelation Effect */

/**
 * pixelate_region:
 * @data: pointer to raw pixel data
 * @stride: row stride in bytes
 * @x1: left boundary (pixels)
 * @y1: top boundary (pixels)
 * @x2: right boundary (pixels)
 * @y2: bottom boundary (pixels)
 * @mosaic_size: size of each mosaic block in pixels
 * @r_off: red channel offset
 * @g_off: green channel offset
 * @b_off: blue channel offset
 *
 * Applies mosaic (pixelation) effect to a rectangular region.
 * The algorithm:
 * 1. Divides the region into blocks of mosaic_size x mosaic_size pixels
 * 2. Calculates the average color of each block
 * 3. Fills the entire block with that average color
 *
 * This creates the characteristic "pixelated" appearance used
 * to obscure facial features for privacy protection.
 */
static void
pixelate_region (guint8 *data, gint stride,
                 gint x1, gint y1, gint x2, gint y2,
                 gint mosaic_size,
                 gint r_off, gint g_off, gint b_off)
{
  gint by, bx, y, x;

  /* Process each mosaic block */
  for (by = y1; by < y2; by += mosaic_size) {
    for (bx = x1; bx < x2; bx += mosaic_size) {

      guint32 r_sum = 0, g_sum = 0, b_sum = 0;
      gint count = 0;

      gint block_y_end = MIN (by + mosaic_size, y2);
      gint block_x_end = MIN (bx + mosaic_size, x2);

      /* Calculate average color */
      for (y = by; y < block_y_end; y++) {
        guint8 *row = data + y * stride;
        for (x = bx; x < block_x_end; x++) {
          guint8 *px = row + x * 4;
          r_sum += px[r_off];
          g_sum += px[g_off];
          b_sum += px[b_off];
          count++;
        }
      }

      if (G_UNLIKELY (count == 0))
        continue;

      guint8 r_avg = r_sum / count;
      guint8 g_avg = g_sum / count;
      guint8 b_avg = b_sum / count;

      /* Fill block with average color */
      for (y = by; y < block_y_end; y++) {
        guint8 *row = data + y * stride;
        for (x = bx; x < block_x_end; x++) {
          guint8 *px = row + x * 4;
          px[r_off] = r_avg;
          px[g_off] = g_avg;
          px[b_off] = b_avg;
        }
      }
    }
  }
}

/* Frame Preprocessing    */

/**
 * preprocess_frame:
 * @self: the #GstFaceBlur instance
 * @src: source pixel data
 * @stride: source row stride in bytes
 * @width: source frame width
 * @height: source frame height
 *
 * Preprocesses a video frame for ONNX model inference:
 * 1. Resizes the frame to MODEL_W x MODEL_H using nearest-neighbor sampling
 * 2. Converts from RGBA (0-255) to normalized float (-1.0 to 1.0)
 * 3. Rearranges from HWC (height, width, channels) to CHW (channels, height, width)
 *
 * The normalization formula is: normalized = (pixel - 127) / 128
 * This maps [0, 255] to approximately [-1.0, 1.0]
 */
static void
preprocess_frame (GstFaceBlur *self, const guint8 *src, gint stride,
                  gint width, gint height)
{
  float *dst = self->input_tensor;
  gint hw = MODEL_W * MODEL_H;
  gint y, x;
  gfloat scale_x = (gfloat) width / MODEL_W;
  gfloat scale_y = (gfloat) height / MODEL_H;

  gint r_off = self->r_offset;
  gint g_off = self->g_offset;
  gint b_off = self->b_offset;

  for (y = 0; y < MODEL_H; y++) {
    gint src_y = CLAMP ((gint) (y * scale_y), 0, height - 1);
    const guint8 *src_row = src + src_y * stride;

    for (x = 0; x < MODEL_W; x++) {
      gint src_x = CLAMP ((gint) (x * scale_x), 0, width - 1);
      const guint8 *px = src_row + src_x * 4;
      gint di = y * MODEL_W + x;

      /* Normalize to [-1, 1] and rearrange to CHW format */
      dst[0 * hw + di] = (px[r_off] - 127.0f) / 128.0f;
      dst[1 * hw + di] = (px[g_off] - 127.0f) / 128.0f;
      dst[2 * hw + di] = (px[b_off] - 127.0f) / 128.0f;
    }
  }
}

/*   ONNX Runtime Inference  */

/**
 * run_onnx_inference:
 * @self: the #GstFaceBlur instance
 * @out_scores: output pointer for scores tensor data
 * @out_boxes: output pointer for boxes tensor data
 * @out_score_value: output pointer for scores OrtValue (caller must release)
 * @out_box_value: output pointer for boxes OrtValue (caller must release)
 *
 * Runs ONNX model inference on the preprocessed input tensor.
 * The model outputs two tensors:
 * - scores: [N, 2] where N=NUM_DETECTIONS, column 0=background, column 1=face
 * - boxes: [N, 4] normalized bounding boxes (x1, y1, x2, y2)
 *
 * This function validates output tensor shapes to prevent buffer overflows
 * if an incompatible model is loaded.
 *
 * Returns: %TRUE on success, %FALSE on error
 */
static gboolean
run_onnx_inference (GstFaceBlur * self, float ** out_scores, float ** out_boxes,
    OrtValue ** out_score_value, OrtValue ** out_box_value)
{
  OrtStatus *st = NULL;
  const char *input_names[] = { "input" };
  const char *output_names[] = { "scores", "boxes" };
  OrtValue *outputs[2] = { NULL, NULL };
  OrtTensorTypeAndShapeInfo *scores_info = NULL;
  OrtTensorTypeAndShapeInfo *boxes_info = NULL;
  size_t scores_count = 0;
  size_t boxes_count = 0;

  *out_score_value = NULL;
  *out_box_value = NULL;

  g_mutex_lock (&self->ort_lock);

  if (G_UNLIKELY (!self->model_loaded)) {
    g_mutex_unlock (&self->ort_lock);
    return FALSE;
  }

  /* Execute inference */
  st = self->ort->Run (self->session, NULL,
      input_names, (const OrtValue * const *) &self->input_value, 1,
      output_names, 2, outputs);

  if (G_UNLIKELY (st != NULL)) {
    GST_ERROR_OBJECT (self, "ONNX Runtime inference error: %s",
        self->ort->GetErrorMessage (st));
    self->ort->ReleaseStatus (st);
    g_mutex_unlock (&self->ort_lock);
    return FALSE;
  }

  /* Validate scores tensor shape */
  st = self->ort->GetTensorTypeAndShape (outputs[0], &scores_info);
  if (G_UNLIKELY (st != NULL)) {
    GST_ERROR_OBJECT (self, "Failed to get scores shape: %s",
        self->ort->GetErrorMessage (st));
    self->ort->ReleaseStatus (st);
    goto inference_error;
  }

  st = self->ort->GetTensorShapeElementCount (scores_info, &scores_count);
  self->ort->ReleaseTensorTypeAndShapeInfo (scores_info);
  if (G_UNLIKELY (st != NULL)) {
    GST_ERROR_OBJECT (self, "Failed to get scores count: %s",
        self->ort->GetErrorMessage (st));
    self->ort->ReleaseStatus (st);
    goto inference_error;
  }

  /* Validate boxes tensor shape */
  st = self->ort->GetTensorTypeAndShape (outputs[1], &boxes_info);
  if (G_UNLIKELY (st != NULL)) {
    GST_ERROR_OBJECT (self, "Failed to get boxes shape: %s",
        self->ort->GetErrorMessage (st));
    self->ort->ReleaseStatus (st);
    goto inference_error;
  }

  st = self->ort->GetTensorShapeElementCount (boxes_info, &boxes_count);
  self->ort->ReleaseTensorTypeAndShapeInfo (boxes_info);
  if (G_UNLIKELY (st != NULL)) {
    GST_ERROR_OBJECT (self, "Failed to get boxes count: %s",
        self->ort->GetErrorMessage (st));
    self->ort->ReleaseStatus (st);
    goto inference_error;
  }

  /* Verify expected tensor dimensions to prevent buffer overflow */
  if (G_UNLIKELY (scores_count != (size_t)(NUM_DETECTIONS * 2) ||
                  boxes_count != (size_t)(NUM_DETECTIONS * 4))) {
    GST_ERROR_OBJECT (self, "Unexpected tensor sizes: scores=%zu (expected %d), "
        "boxes=%zu (expected %d)", scores_count, NUM_DETECTIONS * 2,
        boxes_count, NUM_DETECTIONS * 4);
    goto inference_error;
  }

  /* Get raw tensor data pointers */
  st = self->ort->GetTensorMutableData (outputs[0], (void **) out_scores);
  if (G_UNLIKELY (st != NULL)) {
    GST_ERROR_OBJECT (self, "Failed to get scores: %s",
        self->ort->GetErrorMessage (st));
    self->ort->ReleaseStatus (st);
    goto inference_error;
  }

  st = self->ort->GetTensorMutableData (outputs[1], (void **) out_boxes);
  if (G_UNLIKELY (st != NULL)) {
    GST_ERROR_OBJECT (self, "Failed to get boxes: %s",
        self->ort->GetErrorMessage (st));
    self->ort->ReleaseStatus (st);
    goto inference_error;
  }

  g_mutex_unlock (&self->ort_lock);

  *out_score_value = outputs[0];
  *out_box_value = outputs[1];
  return TRUE;

inference_error:
  if (outputs[0])
    self->ort->ReleaseValue (outputs[0]);
  if (outputs[1])
    self->ort->ReleaseValue (outputs[1]);
  g_mutex_unlock (&self->ort_lock);
  return FALSE;
}

/*   Main Processing Pipeline  */

/**
 * process_frame:
 * @self: the #GstFaceBlur instance
 * @pixels: raw pixel data (modified in-place)
 * @stride: row stride in bytes
 * @width: frame width in pixels
 * @height: frame height in pixels
 *
 * Main processing function that performs the complete face blur pipeline:
 * 1. Preprocess frame for model input
 * 2. Run ONNX inference to detect faces
 * 3. Filter detections above confidence threshold
 * 4. Apply NMS to remove overlapping detections
 * 5. Pixelate detected face regions
 */
static void
process_frame (GstFaceBlur *self, guint8 *pixels, gint stride,
               gint width, gint height)
{
  float *scores = NULL;
  float *boxes = NULL;
  OrtValue *score_value = NULL;
  OrtValue *box_value = NULL;
  gint i;
  gint detection_count = 0;

  /* Step 1: Preprocess frame */
  preprocess_frame (self, pixels, stride, width, height);

  /* Step 2: Run inference */
  if (!run_onnx_inference (self, &scores, &boxes, &score_value, &box_value)) {
    return;
  }

  /* Step 3: Collect detections above threshold */
  for (i = 0; i < NUM_DETECTIONS && detection_count < MAX_FACES; i++) {
    gfloat confidence = scores[i * 2 + 1];

    if (confidence < self->confidence_threshold)
      continue;

    FaceBox *box = &self->detections[detection_count];
    box->x1 = boxes[i * 4 + 0];
    box->y1 = boxes[i * 4 + 1];
    box->x2 = boxes[i * 4 + 2];
    box->y2 = boxes[i * 4 + 3];
    box->confidence = confidence;
    detection_count++;
  }

  /* Release ONNX output tensors */
  g_mutex_lock (&self->ort_lock);
  if (score_value)
    self->ort->ReleaseValue (score_value);
  if (box_value)
    self->ort->ReleaseValue (box_value);
  g_mutex_unlock (&self->ort_lock);

  if (detection_count == 0)
    return;

  /* Step 4: Apply NMS */
  detection_count = nms_filter (self->detections, detection_count,
                                NMS_IOU_THRESHOLD);

  GST_LOG_OBJECT (self, "Detected %d faces after NMS", detection_count);

  /* Step 5: Pixelate each detected face */
  for (i = 0; i < detection_count; i++) {
    FaceBox *face = &self->detections[i];

    /* Convert normalized coordinates to pixel coordinates */
    gint x1 = (gint) (face->x1 * width);
    gint y1 = (gint) (face->y1 * height);
    gint x2 = (gint) (face->x2 * width);
    gint y2 = (gint) (face->y2 * height);

    /* Clamp to frame boundaries */
    x1 = CLAMP (x1, 0, width - 1);
    y1 = CLAMP (y1, 0, height - 1);
    x2 = CLAMP (x2, x1 + 1, width);
    y2 = CLAMP (y2, y1 + 1, height);

    GST_LOG_OBJECT (self, "  Face %d: (%d,%d)-(%d,%d) conf=%.2f",
                    i, x1, y1, x2, y2, face->confidence);

    pixelate_region (pixels, stride, x1, y1, x2, y2,
                     self->mosaic_size,
                     self->r_offset, self->g_offset, self->b_offset);
  }
}

/*  GstBaseTransform Virtual Methods  */

/**
 * gst_face_blur_transform_ip:
 * @bt: the #GstBaseTransform
 * @buf: the #GstBuffer to transform
 *
 * In-place transform function. This is called for each buffer
 * that passes through the element. The buffer is modified in-place
 * to add the face blur effect.
 *
 * Returns: GST_FLOW_OK on success
 */
static GstFlowReturn
gst_face_blur_transform_ip (GstBaseTransform *bt, GstBuffer *buf)
{
  GstFaceBlur *self = GST_FACE_BLUR (bt);
  GstVideoFrame frame;

  if (G_UNLIKELY (!self->model_loaded)) {
    GST_DEBUG_OBJECT (self, "Model not loaded, passing through");
    return GST_FLOW_OK;
  }

  if (G_UNLIKELY (!self->video_info_set)) {
    GST_WARNING_OBJECT (self, "Video info not set");
    return GST_FLOW_OK;
  }

  if (!gst_video_frame_map (&frame, &self->video_info, buf, GST_MAP_READWRITE)) {
    GST_ERROR_OBJECT (self, "Failed to map video frame");
    return GST_FLOW_ERROR;
  }

  process_frame (self,
                 GST_VIDEO_FRAME_PLANE_DATA (&frame, 0),
                 GST_VIDEO_FRAME_PLANE_STRIDE (&frame, 0),
                 GST_VIDEO_FRAME_WIDTH (&frame),
                 GST_VIDEO_FRAME_HEIGHT (&frame));

  gst_video_frame_unmap (&frame);
  return GST_FLOW_OK;
}

/**
 * gst_face_blur_set_caps:
 * @bt: the #GstBaseTransform
 * @incaps: input caps
 * @outcaps: output caps (same as input for in-place transform)
 *
 * Called when caps are negotiated. Parses video info and configures
 * channel offsets for the negotiated format.
 *
 * Returns: %TRUE on success
 */
static gboolean
gst_face_blur_set_caps (GstBaseTransform *bt, GstCaps *incaps,
    G_GNUC_UNUSED GstCaps *outcaps)
{
  GstFaceBlur *self = GST_FACE_BLUR (bt);
  GstVideoFormat format;

  if (!gst_video_info_from_caps (&self->video_info, incaps)) {
    GST_ERROR_OBJECT (self, "Failed to parse video caps");
    return FALSE;
  }

  format = GST_VIDEO_INFO_FORMAT (&self->video_info);
  if (!gst_face_blur_configure_format (self, format)) {
    return FALSE;
  }

  GST_INFO_OBJECT (self, "Configured for %dx%d @ %d/%d fps, format %s",
                   GST_VIDEO_INFO_WIDTH (&self->video_info),
                   GST_VIDEO_INFO_HEIGHT (&self->video_info),
                   GST_VIDEO_INFO_FPS_N (&self->video_info),
                   GST_VIDEO_INFO_FPS_D (&self->video_info),
                   gst_video_format_to_string (format));

  self->video_info_set = TRUE;
  return TRUE;
}

/*   ONNX Runtime Resource Management   */

/**
 * gst_face_blur_cleanup_onnx:
 * @self: the #GstFaceBlur instance
 *
 * Releases all ONNX Runtime resources in the correct order.
 * Must be called with ort_lock held or during finalization.
 */
static void
gst_face_blur_cleanup_onnx (GstFaceBlur *self)
{
  if (self->input_value != NULL) {
    self->ort->ReleaseValue (self->input_value);
    self->input_value = NULL;
  }
  if (self->session != NULL) {
    self->ort->ReleaseSession (self->session);
    self->session = NULL;
  }
  if (self->meminfo != NULL) {
    self->ort->ReleaseMemoryInfo (self->meminfo);
    self->meminfo = NULL;
  }
  if (self->env != NULL) {
    self->ort->ReleaseEnv (self->env);
    self->env = NULL;
  }
  g_clear_pointer (&self->input_tensor, g_free);
}

/**
 * gst_face_blur_start:
 * @bt: the #GstBaseTransform
 *
 * Called when the element transitions to PAUSED state.
 * Initializes ONNX Runtime and loads the face detection model.
 *
 * Returns: %TRUE on success, %FALSE on failure
 */
static gboolean
gst_face_blur_start (GstBaseTransform *bt)
{
  GstFaceBlur *self = GST_FACE_BLUR (bt);
  OrtStatus *st = NULL;
  OrtSessionOptions *opts = NULL;
  gsize tensor_size;
  int64_t shape[] = { 1, 3, MODEL_H, MODEL_W };
  G_STATIC_ASSERT (G_N_ELEMENTS (shape) == 4);  /* NCHW format validation */

  g_mutex_lock (&self->ort_lock);

  /* Validate model path */
  if (self->model_location == NULL || self->model_location[0] == '\0') {
    GST_ERROR_OBJECT (self, "model-location property not set");
    g_mutex_unlock (&self->ort_lock);
    return FALSE;
  }

  /* Initialize ONNX Runtime API */
  self->ort = OrtGetApiBase ()->GetApi (ORT_API_VERSION);
  if (G_UNLIKELY (self->ort == NULL)) {
    GST_ERROR_OBJECT (self, "Failed to get ONNX Runtime API (version %d)",
                      ORT_API_VERSION);
    g_mutex_unlock (&self->ort_lock);
    return FALSE;
  }

  /* Create ONNX environment */
  st = self->ort->CreateEnv (ORT_LOGGING_LEVEL_WARNING, "gst-faceblur",
                              &self->env);
  if (G_UNLIKELY (st != NULL)) {
    GST_ERROR_OBJECT (self, "CreateEnv failed: %s",
                      self->ort->GetErrorMessage (st));
    goto error;
  }

  /* Create CPU memory allocator */
  st = self->ort->CreateCpuMemoryInfo (OrtArenaAllocator, OrtMemTypeDefault,
                                        &self->meminfo);
  if (G_UNLIKELY (st != NULL)) {
    GST_ERROR_OBJECT (self, "CreateCpuMemoryInfo failed: %s",
                      self->ort->GetErrorMessage (st));
    goto error;
  }

  /* Configure session options with graph optimizations */
  st = self->ort->CreateSessionOptions (&opts);
  if (G_UNLIKELY (st != NULL)) {
    GST_ERROR_OBJECT (self, "CreateSessionOptions failed: %s",
                      self->ort->GetErrorMessage (st));
    goto error;
  }

  st = self->ort->SetSessionGraphOptimizationLevel (opts, ORT_ENABLE_ALL);
  if (G_UNLIKELY (st != NULL)) {
    GST_WARNING_OBJECT (self, "SetSessionGraphOptimizationLevel failed: %s",
                        self->ort->GetErrorMessage (st));
    self->ort->ReleaseStatus (st);
    st = NULL;
  }

  /* Configure thread pool (0 = auto-detect based on CPU cores) */
  st = self->ort->SetIntraOpNumThreads (opts, 0);
  if (st != NULL) {
    GST_DEBUG_OBJECT (self, "SetIntraOpNumThreads: %s (non-fatal)",
                      self->ort->GetErrorMessage (st));
    self->ort->ReleaseStatus (st);
    st = NULL;
  }

  /* Load ONNX model file */
#ifdef G_OS_WIN32
  {
    /* Windows requires wide character path */
    wchar_t *wpath = g_utf8_to_utf16 (self->model_location, -1, NULL, NULL, NULL);
    if (wpath == NULL) {
      GST_ERROR_OBJECT (self, "Failed to convert model path to UTF-16");
      self->ort->ReleaseSessionOptions (opts);
      goto error;
    }
    st = self->ort->CreateSession (self->env, wpath, opts, &self->session);
    g_free (wpath);
  }
#else
  st = self->ort->CreateSession (self->env, self->model_location, opts,
                                  &self->session);
#endif

  self->ort->ReleaseSessionOptions (opts);
  opts = NULL;

  if (G_UNLIKELY (st != NULL)) {
    GST_ERROR_OBJECT (self, "CreateSession failed for '%s': %s",
                      self->model_location, self->ort->GetErrorMessage (st));
    goto error;
  }

  GST_INFO_OBJECT (self, "Loaded ONNX model: %s", self->model_location);

  /* Allocate input tensor buffer (CHW format: 3 channels x H x W) */
  tensor_size = 3 * MODEL_W * MODEL_H * sizeof (float);
  self->input_tensor = (float *) g_malloc0 (tensor_size);
  if (G_UNLIKELY (self->input_tensor == NULL)) {
    GST_ERROR_OBJECT (self, "Failed to allocate input tensor (%zu bytes)",
                      tensor_size);
    goto error;
  }

  /* Create OrtValue wrapper for input tensor */
  st = self->ort->CreateTensorWithDataAsOrtValue (
      self->meminfo,
      self->input_tensor,
      tensor_size,
      shape, G_N_ELEMENTS (shape),
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      &self->input_value);

  if (G_UNLIKELY (st != NULL)) {
    GST_ERROR_OBJECT (self, "CreateTensorWithDataAsOrtValue failed: %s",
                      self->ort->GetErrorMessage (st));
    goto error;
  }

  /* Allocate detection results buffer */
  self->detections = g_new0 (FaceBox, MAX_FACES);
  if (G_UNLIKELY (self->detections == NULL)) {
    GST_ERROR_OBJECT (self, "Failed to allocate detections array");
    goto error;
  }

  self->model_loaded = TRUE;
  g_mutex_unlock (&self->ort_lock);

  GST_DEBUG_OBJECT (self, "ONNX Runtime initialized successfully");
  return TRUE;

error:
  if (st != NULL)
    self->ort->ReleaseStatus (st);
  if (opts != NULL)
    self->ort->ReleaseSessionOptions (opts);

  gst_face_blur_cleanup_onnx (self);
  g_mutex_unlock (&self->ort_lock);
  return FALSE;
}

/**
 * gst_face_blur_stop:
 * @bt: the #GstBaseTransform
 *
 * Called when the element transitions to READY state.
 * Releases ONNX Runtime resources.
 *
 * Returns: %TRUE always
 */
static gboolean
gst_face_blur_stop (GstBaseTransform *bt)
{
  GstFaceBlur *self = GST_FACE_BLUR (bt);

  g_mutex_lock (&self->ort_lock);
  self->model_loaded = FALSE;
  gst_face_blur_cleanup_onnx (self);
  g_clear_pointer (&self->detections, g_free);
  g_mutex_unlock (&self->ort_lock);

  self->video_info_set = FALSE;

  GST_DEBUG_OBJECT (self, "ONNX Runtime resources released");
  return TRUE;
}

/*   GObject Virtual Methods    */

static void
gst_face_blur_set_property (GObject *obj,
                            guint prop_id,
                            const GValue *value,
                            GParamSpec *pspec)
{
  GstFaceBlur *self = GST_FACE_BLUR (obj);

  switch (prop_id) {
    case PROP_MODEL_LOCATION:
      g_mutex_lock (&self->ort_lock);
      g_free (self->model_location);
      self->model_location = g_value_dup_string (value);
      g_mutex_unlock (&self->ort_lock);
      g_object_notify_by_pspec (obj, properties[PROP_MODEL_LOCATION]);
      break;
    case PROP_MOSAIC_SIZE:
      g_mutex_lock (&self->ort_lock);
      self->mosaic_size = g_value_get_int (value);
      g_mutex_unlock (&self->ort_lock);
      g_object_notify_by_pspec (obj, properties[PROP_MOSAIC_SIZE]);
      break;
    case PROP_CONFIDENCE_THRESHOLD:
      g_mutex_lock (&self->ort_lock);
      self->confidence_threshold = g_value_get_float (value);
      g_mutex_unlock (&self->ort_lock);
      g_object_notify_by_pspec (obj, properties[PROP_CONFIDENCE_THRESHOLD]);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (obj, prop_id, pspec);
      break;
  }
}

static void
gst_face_blur_get_property (GObject *obj,
                            guint prop_id,
                            GValue *value,
                            GParamSpec *pspec)
{
  GstFaceBlur *self = GST_FACE_BLUR (obj);

  switch (prop_id) {
    case PROP_MODEL_LOCATION:
      g_mutex_lock (&self->ort_lock);
      g_value_set_string (value, self->model_location);
      g_mutex_unlock (&self->ort_lock);
      break;
    case PROP_MOSAIC_SIZE:
      g_mutex_lock (&self->ort_lock);
      g_value_set_int (value, self->mosaic_size);
      g_mutex_unlock (&self->ort_lock);
      break;
    case PROP_CONFIDENCE_THRESHOLD:
      g_mutex_lock (&self->ort_lock);
      g_value_set_float (value, self->confidence_threshold);
      g_mutex_unlock (&self->ort_lock);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (obj, prop_id, pspec);
      break;
  }
}

static void
gst_face_blur_finalize (GObject *obj)
{
  GstFaceBlur *self = GST_FACE_BLUR (obj);

  gst_face_blur_cleanup_onnx (self);
  g_clear_pointer (&self->detections, g_free);
  g_free (self->model_location);

  g_mutex_clear (&self->ort_lock);

  G_OBJECT_CLASS (gst_face_blur_parent_class)->finalize (obj);
}


/*  Class and Instance Init  */


static void
gst_face_blur_class_init (GstFaceBlurClass *klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstBaseTransformClass *bt_class = GST_BASE_TRANSFORM_CLASS (klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstCaps *caps;

  gobject_class->set_property = gst_face_blur_set_property;
  gobject_class->get_property = gst_face_blur_get_property;
  gobject_class->finalize = gst_face_blur_finalize;

  bt_class->start = GST_DEBUG_FUNCPTR (gst_face_blur_start);
  bt_class->stop = GST_DEBUG_FUNCPTR (gst_face_blur_stop);
  bt_class->set_caps = GST_DEBUG_FUNCPTR (gst_face_blur_set_caps);
  bt_class->transform_ip = GST_DEBUG_FUNCPTR (gst_face_blur_transform_ip);

  /* Process all frames (no passthrough) */
  bt_class->passthrough_on_same_caps = FALSE;

  /**
   * GstFaceBlur:model-location:
   *
   * Path to the ONNX face detection model file.
   * The element is designed for UltraFace-RFB-320 but may work with
   * compatible models that have the same input/output signature.
   *
   * This property must be set before the element transitions to PAUSED.
   */
  properties[PROP_MODEL_LOCATION] = g_param_spec_string (
      "model-location", "Model Location",
      "Path to the ONNX face detection model file",
      NULL,
      G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
      GST_PARAM_MUTABLE_READY);

  /**
   * GstFaceBlur:mosaic-size:
   *
   * Size of the pixelation blocks in pixels.
   * Larger values provide stronger anonymization but coarser appearance.
   * Smaller values preserve more detail but may not fully obscure identity.
   */
  properties[PROP_MOSAIC_SIZE] = g_param_spec_int (
      "mosaic-size", "Mosaic Size",
      "Size of pixelation blocks in pixels",
      4, 64, DEFAULT_MOSAIC_SIZE,
      G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
      GST_PARAM_MUTABLE_READY);

  /**
   * GstFaceBlur:confidence-threshold:
   *
   * Minimum confidence score for face detection.
   * Higher values reduce false positives but may miss some faces.
   * Lower values detect more faces but may include false positives.
   */
  properties[PROP_CONFIDENCE_THRESHOLD] = g_param_spec_float (
      "confidence-threshold", "Confidence Threshold",
      "Minimum confidence score for face detection (0.0 to 1.0)",
      0.0f, 1.0f, DEFAULT_CONFIDENCE_THRESHOLD,
      G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
      GST_PARAM_MUTABLE_READY);

  g_object_class_install_properties (gobject_class, G_N_ELEMENTS (properties), properties);

  /* Element metadata for gst-inspect */
  gst_element_class_set_static_metadata (element_class,
      "Face Blur",
      "Filter/Effect/Video/Privacy",
      "Privacy-grade face pixelation using ONNX Runtime inference. "
      "Uses Non-Maximum Suppression for accurate detection.",
      "Arokya Matthew Nathan <arokyamatthewnathan@gmail.com>");

  /* Pad templates - 4-byte RGBA-family formats */
  caps = gst_caps_from_string (
      "video/x-raw, "
      "format = (string) { RGBA, RGBx, BGRA, BGRx, ARGB, ABGR }, "
      "width = (int) [ 1, MAX ], "
      "height = (int) [ 1, MAX ], "
      "framerate = (fraction) [ 0/1, MAX ]");

  gst_element_class_add_pad_template (element_class,
      gst_pad_template_new ("sink", GST_PAD_SINK, GST_PAD_ALWAYS, caps));
  gst_element_class_add_pad_template (element_class,
      gst_pad_template_new ("src", GST_PAD_SRC, GST_PAD_ALWAYS, caps));
  gst_caps_unref (caps);

  GST_DEBUG_CATEGORY_INIT (gst_faceblur_debug, "faceblur", 0,
      "Face Blur privacy filter with NMS");
}

static void
gst_face_blur_init (GstFaceBlur *self)
{
  g_mutex_init (&self->ort_lock);

  /* Initialize properties with defaults */
  self->model_location = NULL;
  self->mosaic_size = DEFAULT_MOSAIC_SIZE;
  self->confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD;

  /* Initialize state */
  self->video_info_set = FALSE;
  self->r_offset = 0;
  self->g_offset = 1;
  self->b_offset = 2;

  /* ONNX Runtime resources start NULL */
  self->ort = NULL;
  self->env = NULL;
  self->session = NULL;
  self->meminfo = NULL;
  self->input_tensor = NULL;
  self->input_value = NULL;
  self->detections = NULL;
  self->model_loaded = FALSE;
}

/* Plugin Registration */


static gboolean
plugin_init (GstPlugin *plugin)
{
  return gst_element_register (plugin,
                               "faceblur",
                               GST_RANK_NONE,
                               GST_TYPE_FACE_BLUR);
}

GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    faceblur,
    "Face Blur Plugin with ONNX Runtime and NMS",
    plugin_init,
    "1.0.0",
    "LGPL",
    "GstFaceBlur",
    "https://github.com/ArokyaMatthew"
)
