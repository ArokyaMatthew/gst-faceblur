#ifndef __GST_FACE_BLUR_H__
#define __GST_FACE_BLUR_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>

G_BEGIN_DECLS

#define GST_TYPE_FACE_BLUR (gst_face_blur_get_type())
#define GST_FACE_BLUR(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_FACE_BLUR, GstFaceBlur))
#define GST_FACE_BLUR_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_FACE_BLUR, GstFaceBlurClass))
#define GST_IS_FACE_BLUR(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_FACE_BLUR))
#define GST_IS_FACE_BLUR_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_FACE_BLUR))
#define GST_FACE_BLUR_GET_CLASS(obj) \
    (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_FACE_BLUR, GstFaceBlurClass))

typedef struct _GstFaceBlur GstFaceBlur;
typedef struct _GstFaceBlurClass GstFaceBlurClass;

GType gst_face_blur_get_type (void);

G_END_DECLS

#endif /* __GST_FACE_BLUR_H__ */
