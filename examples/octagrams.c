// octagrams.c - Port of Shadertoy shader to C with nanomp4h264
//
// Original shader: https://www.shadertoy.com/view/tlVGDt
// Created by whisky_shusuky in 2020-01-28
// Licensed under CC BY-NC-SA 3.0 Unported License

#include "../nanomp4h264.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 100
#define HEIGHT 100
#define FPS 30
#define DURATION 5
#define TOTAL_FRAMES (FPS * DURATION)

// Vector types
typedef struct { float x, y; } vec2;
typedef struct { float x, y, z; } vec3;
typedef struct { float m00, m01, m10, m11; } mat2;

// vec2 operations
static inline vec2 vec2_make(float x, float y) { return (vec2){x, y}; }
static inline vec2 vec2_sub(vec2 a, vec2 b) { return (vec2){a.x - b.x, a.y - b.y}; }
static inline vec2 vec2_scale(vec2 v, float s) { return (vec2){v.x * s, v.y * s}; }

// vec3 operations
static inline vec3 vec3_make(float x, float y, float z) { return (vec3){x, y, z}; }
static inline vec3 vec3_add(vec3 a, vec3 b) { return (vec3){a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline vec3 vec3_sub(vec3 a, vec3 b) { return (vec3){a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline vec3 vec3_scale(vec3 v, float s) { return (vec3){v.x * s, v.y * s, v.z * s}; }
static inline vec3 vec3_abs(vec3 v) { return (vec3){fabsf(v.x), fabsf(v.y), fabsf(v.z)}; }
static inline vec3 vec3_max(vec3 v, float m) { return (vec3){fmaxf(v.x, m), fmaxf(v.y, m), fmaxf(v.z, m)}; }
static inline float vec3_length(vec3 v) { return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); }
static inline vec3 vec3_normalize(vec3 v) { float len = vec3_length(v); return (vec3){v.x / len, v.y / len, v.z / len}; }

// mat2 operations
static inline mat2 mat2_make(float m00, float m01, float m10, float m11) { return (mat2){m00, m01, m10, m11}; }
static inline vec2 mat2_mul_vec2(mat2 m, vec2 v) { return (vec2){m.m00 * v.x + m.m01 * v.y, m.m10 * v.x + m.m11 * v.y}; }

// GLSL mod (always positive result for positive divisor)
static inline float glsl_mod(float x, float y) {
    return x - y * floorf(x / y);
}

// Cached rotation matrix for rot(0.8f)
static mat2 rot_0_8;

// Rotation matrix
static mat2 rot(float a) {
    float c = cosf(a), s = sinf(a);
    return mat2_make(c, s, -s, c);
}

// Signed distance box
static float sdBox(vec3 p, vec3 b) {
    vec3 q = vec3_sub(vec3_abs(p), b);
    float maxq = fmaxf(q.x, fmaxf(q.y, q.z));
    return vec3_length(vec3_max(q, 0.0f)) + fminf(maxq, 0.0f);
}

// Box with transform
static float box(vec3 pos, float scale) {
    pos = vec3_scale(pos, scale);
    float base = sdBox(pos, vec3_make(0.4f, 0.4f, 0.1f)) / 1.5f;
    // pos.xy *= 5.0 and other transforms not used for result
    float result = -base;
    return result;
}

// Animated box composition
static float box_set(vec3 pos, float gTime) {
    float sin_gTime = sinf(gTime * 0.4f);
    float abs_sin_gTime = fabsf(sin_gTime);

    vec3 pos1 = pos;
    pos1.y += sin_gTime * 2.5f;
    vec2 xy1 = mat2_mul_vec2(rot_0_8, vec2_make(pos1.x, pos1.y));
    pos1.x = xy1.x; pos1.y = xy1.y;
    float box1 = box(pos1, 2.0f - abs_sin_gTime * 1.5f);

    vec3 pos2 = pos;
    pos2.y -= sin_gTime * 2.5f;
    vec2 xy2 = mat2_mul_vec2(rot_0_8, vec2_make(pos2.x, pos2.y));
    pos2.x = xy2.x; pos2.y = xy2.y;
    float box2 = box(pos2, 2.0f - abs_sin_gTime * 1.5f);

    vec3 pos3 = pos;
    pos3.x += sin_gTime * 2.5f;
    vec2 xy3 = mat2_mul_vec2(rot_0_8, vec2_make(pos3.x, pos3.y));
    pos3.x = xy3.x; pos3.y = xy3.y;
    float box3 = box(pos3, 2.0f - abs_sin_gTime * 1.5f);

    vec3 pos4 = pos;
    pos4.x -= sin_gTime * 2.5f;
    vec2 xy4 = mat2_mul_vec2(rot_0_8, vec2_make(pos4.x, pos4.y));
    pos4.x = xy4.x; pos4.y = xy4.y;
    float box4 = box(pos4, 2.0f - abs_sin_gTime * 1.5f);

    vec3 pos5 = pos;
    vec2 xy5 = mat2_mul_vec2(rot_0_8, vec2_make(pos5.x, pos5.y));
    pos5.x = xy5.x; pos5.y = xy5.y;
    float box5 = box(pos5, 0.5f) * 6.0f;

    vec3 pos6 = pos;
    float box6 = box(pos6, 0.5f) * 6.0f;

    float result = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(box1, box2), box3), box4), box5), box6);
    return result;
}

// Scene distance function
static float map(vec3 pos, float gTime) {
    return box_set(pos, gTime);
}

// Main shader function - returns color for a pixel
static vec3 mainImage(vec2 fragCoord, float iTime) {
    vec2 iResolution = vec2_make((float)WIDTH, (float)HEIGHT);

    // Normalized coordinates
    vec2 p = vec2_scale(
        vec2_sub(vec2_scale(fragCoord, 2.0f), iResolution),
        1.0f / fminf(iResolution.x, iResolution.y)
    );

    // Ray origin and direction
    vec3 ro = vec3_make(0.0f, -0.2f, iTime * 4.0f);
    vec3 ray = vec3_normalize(vec3_make(p.x, p.y, 1.5f));

    // Rotate ray
    vec2 rxy = mat2_mul_vec2(rot(sinf(iTime * 0.03f) * 5.0f), vec2_make(ray.x, ray.y));
    ray.x = rxy.x; ray.y = rxy.y;
    vec2 ryz = mat2_mul_vec2(rot(sinf(iTime * 0.05f) * 0.2f), vec2_make(ray.y, ray.z));
    ray.y = ryz.x; ray.z = ryz.y;

    float t = 0.1f;
    float ac = 0.0f;

    // Raymarching loop
    for (int i = 0; i < 99; i++) {
        vec3 pos = vec3_add(ro, vec3_scale(ray, t));

        // pos = mod(pos - 2, 4) - 2
        pos.x = glsl_mod(pos.x - 2.0f, 4.0f) - 2.0f;
        pos.y = glsl_mod(pos.y - 2.0f, 4.0f) - 2.0f;
        pos.z = glsl_mod(pos.z - 2.0f, 4.0f) - 2.0f;

        float gTime = iTime - (float)i * 0.01f;

        float d = map(pos, gTime);
        d = fmaxf(fabsf(d), 0.01f);
        ac += expf(-d * 23.0f);

        t += d * 0.55f;
    }

    // Color calculation
    vec3 col = vec3_make(ac * 0.02f, ac * 0.02f, ac * 0.02f);
    col.y += 0.2f * fabsf(sinf(iTime));
    col.z += 0.5f + sinf(iTime) * 0.2f;

    return col;
}

int main(void) {
    nanomp4h264_t enc;
    nanomp4h264_config_t config = {
        .width = WIDTH,
        .height = HEIGHT,
        .fps_num = FPS,
        .fps_den = 1,
    };

    nanomp4h264_open(&enc, &config, "octagrams.mp4");

    if (nanomp4h264_get_error(&enc) != 0) {
        fprintf(stderr, "Failed to open encoder: %d\n", nanomp4h264_get_error(&enc));
        return 1;
    }

    uint8_t *rgb = malloc(WIDTH * HEIGHT * 3);
    if (!rgb) {
        fprintf(stderr, "Failed to allocate RGB buffer\n");
        return 1;
    }

    printf("Rendering %d frames at %dx%d...\n", TOTAL_FRAMES, WIDTH, HEIGHT);

    rot_0_8 = rot(0.8f);

    for (int frame = 0; frame < TOTAL_FRAMES; frame++) {
        float iTime = (float)frame / (float)FPS;

        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                // fragCoord uses center of pixel
                vec2 fragCoord = vec2_make((float)x + 0.5f, (float)(HEIGHT - 1 - y) + 0.5f);
                vec3 col = mainImage(fragCoord, iTime);

                // Clamp and convert to 8-bit
                int idx = (y * WIDTH + x) * 3;
                rgb[idx + 0] = (uint8_t)(fminf(fmaxf(col.x, 0.0f), 1.0f) * 255.0f);
                rgb[idx + 1] = (uint8_t)(fminf(fmaxf(col.y, 0.0f), 1.0f) * 255.0f);
                rgb[idx + 2] = (uint8_t)(fminf(fmaxf(col.z, 0.0f), 1.0f) * 255.0f);
            }
        }

        nanomp4h264_write_frame(&enc, rgb, NANOMP4H264_FORMAT_RGB888);
        if ((frame+1) % 20 == 0) {
            nanomp4h264_flush(&enc);
        }

        if (nanomp4h264_get_error(&enc) != 0) {
            fprintf(stderr, "Error after frame %d: %d\n", frame, nanomp4h264_get_error(&enc));
            break;
        }

        if ((frame + 1) % 60 == 0) {
            printf("Frame %d/%d\n", frame + 1, TOTAL_FRAMES);
        }
    }

    free(rgb);
    nanomp4h264_close(&enc);

    int err = nanomp4h264_get_error(&enc);
    if (err != 0) {
        fprintf(stderr, "Encoding error: %d\n", err);
        return 1;
    }

    printf("Done! Output: octagrams.mp4\n");
    return 0;
}
