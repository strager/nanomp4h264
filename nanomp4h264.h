// nanomp4h264.h - Minimal MP4+H.264 video encoder
//
// A small library for encoding RGB frames to H.264 video in an MP4 container.
//
// Usage:
//   1. Allocate a nanomp4h264_t (on stack or heap)
//   2. Call nanomp4h264_open() with config and output path
//   3. Call nanomp4h264_write_frame() for each frame
//   4. Call nanomp4h264_close() to finalize the file
//   5. Call nanomp4h264_get_error() to check for errors
//
// Example:
//   nanomp4h264_t enc;
//   nanomp4h264_config_t config = {
//       .width = 1920,
//       .height = 1080,
//       .fps_num = 30,
//       .fps_den = 1,
//   };
//   nanomp4h264_open(&enc, &config, "output.mp4");
//
//   for each frame:
//       uint8_t *rgb = ...;  // width * height * 3 bytes
//       nanomp4h264_write_frame(&enc, rgb, NANOMP4H264_FORMAT_RGB888);
//
//   nanomp4h264_close(&enc);
//   if (nanomp4h264_get_error(&enc) != 0) { handle error }

#ifndef NANOMP4H264_H
#define NANOMP4H264_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Pixel format for input frames
typedef enum nanomp4h264_format {
    // 3 bytes per pixel: R, G, B in row-major order (top to bottom)
    NANOMP4H264_FORMAT_RGB888 = 0,
} nanomp4h264_format_t;

// Encoder configuration
typedef struct nanomp4h264_config {
    int width;    // Frame width in pixels
    int height;   // Frame height in pixels
    int fps_num;  // Framerate numerator (e.g., 30 for 30fps, 30000 for 29.97fps)
    int fps_den;  // Framerate denominator (e.g., 1 for 30fps, 1001 for 29.97fps)
} nanomp4h264_config_t;

// Encoder state (allocate this yourself, do not access fields directly)
typedef struct nanomp4h264 {
    int _error;
    int _width, _height, _fps_num, _fps_den;
    int _padded_width, _padded_height;
    int _mb_width, _mb_height;
    int _crop_right, _crop_bottom;
    void *_file;
    long _mdat_start_pos;
    uint32_t _frame_count;
    uint32_t _frame_nal_size;
    uint8_t *_yuv_buffer;
} nanomp4h264_t;

// Open encoder and create output file.
//
// Parameters:
//   enc        - Pointer to caller-allocated encoder struct
//   config     - Encoder configuration (width, height, framerate)
//   filepath   - Path to the output .mp4 file
void nanomp4h264_open(nanomp4h264_t *enc, const nanomp4h264_config_t *config, const char *filepath);

// Encode and write a single frame.
//
// Parameters:
//   enc        - Encoder from nanomp4h264_open()
//   data       - Pointer to pixel data (size depends on format)
//   format     - Pixel format of the input data
void nanomp4h264_write_frame(nanomp4h264_t *enc, const uint8_t *data, nanomp4h264_format_t format);

// Flush encoder and update MP4 metadata.
//
// Ensures the MP4 file is readable by any decoder for all frames written
// so far. Call this when you need the file to be playable before closing.
// This may be called multiple times.
//
// Parameters:
//   enc        - Encoder from nanomp4h264_open()
void nanomp4h264_flush(nanomp4h264_t *enc);

// Finalize the MP4 file and release internal resources.
//
// This must be called to produce a valid MP4 file. The encoder struct
// remains valid for calling nanomp4h264_get_error() after this function
// returns.
//
// Parameters:
//   enc        - Encoder from nanomp4h264_open()
void nanomp4h264_close(nanomp4h264_t *enc);

// Get error status.
//
// Errors are sticky.
//
// Check this after nanomp4h264_close() to determine if encoding succeeded.
//
// Parameters:
//   enc        - Encoder from nanomp4h264_open()
//
// Returns:
//   0 on success, non-zero error code on failure.
int nanomp4h264_get_error(const nanomp4h264_t *enc);

#ifdef __cplusplus
}
#endif

#endif /* NANOMP4H264_H */
