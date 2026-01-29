// Port of Shadertoy shader with audio to C with nanomp4h264
//
// https://www.shadertoy.com/view/WfcGWj
// CC0: Trailing the Twinkling Tunnelwisp by Pestis

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
#define SAMPLE_RATE 44100
#define AUDIO_CHANNELS 2

// Vector types
typedef struct { float x, y; } vec2;
typedef struct { float x, y, z; } vec3;
typedef struct { float x, y, z, w; } vec4;

// vec2 operations
static inline vec2 vec2_make(float x, float y) { return (vec2){x, y}; }
static inline vec2 vec2_add(vec2 a, vec2 b) { return (vec2){a.x + b.x, a.y + b.y}; }
static inline vec2 vec2_sub(vec2 a, vec2 b) { return (vec2){a.x - b.x, a.y - b.y}; }
static inline vec2 vec2_scale(vec2 v, float s) { return (vec2){v.x * s, v.y * s}; }

// vec3 operations
static inline vec3 vec3_make(float x, float y, float z) { return (vec3){x, y, z}; }
static inline vec3 vec3_scale(vec3 v, float s) { return (vec3){v.x * s, v.y * s, v.z * s}; }
static inline float vec3_length(vec3 v) { return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); }
static inline vec3 vec3_normalize(vec3 v) { float len = vec3_length(v); return (vec3){v.x / len, v.y / len, v.z / len}; }

// vec4 operations
static inline vec4 vec4_make(float x, float y, float z, float w) { return (vec4){x, y, z, w}; }
static inline vec4 vec4_scale(vec4 v, float s) { return (vec4){v.x * s, v.y * s, v.z * s, v.w * s}; }
static inline vec4 vec4_add(vec4 a, vec4 b) { return (vec4){a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }
static inline vec4 sin4(vec4 v) { return (vec4){sinf(v.x), sinf(v.y), sinf(v.z), sinf(v.w)}; }
static inline vec4 cos4(vec4 v) { return (vec4){cosf(v.x), cosf(v.y), cosf(v.z), cosf(v.w)}; }
static inline float dot4(vec4 a, vec4 b) { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }

// GLSL mod (always positive result for positive divisor)
static inline float glsl_mod(float x, float y) {
    return x - y * floorf(x / y);
}

// GLSL fract
static inline float glsl_fract(float x) {
    return x - floorf(x);
}

// Gyroid distance function
static float g(vec4 p, float s) {
    p = vec4_scale(p, s);
    vec4 sinp = sin4(p);
    vec4 p_zxwy = vec4_make(p.z, p.x, p.w, p.y);  // swizzle p.zxwy
    vec4 cosp = cos4(p_zxwy);
    return fabsf(dot4(sinp, cosp) - 1.0f) / s;
}

// Main shader function - returns color for a pixel
static vec3 mainImage(vec2 fragCoord, float iTime) {
    vec2 r = vec2_make((float)WIDTH, (float)HEIGHT);
    float T = iTime;
    vec4 U = vec4_make(2.0f, 1.0f, 0.0f, 3.0f);

    float i = 0.0f, d = 0.0f, z = 0.0f, s = 0.0f;
    vec4 o = vec4_make(0.0f, 0.0f, 0.0f, 0.0f);
    vec4 q = vec4_make(0.0f, 0.0f, 0.0f, 0.0f);
    vec4 p = vec4_make(0.0f, 0.0f, 0.0f, 0.0f);

    while (++i < 79.0f) {
        // Accumulate glow - brighter and sharper if not mirrored (above axis)
        float mult = (s > 0.0f) ? 1.0f : 0.1f;
        float denom = fmaxf((s > 0.0f) ? d : d*d*d, 5e-4f);
        vec4 contribution = vec4_scale(p, mult * p.w / denom);
        o = vec4_add(o, contribution);

        // Advance along ray by current distance estimate (+ epsilon)
        z += d + 5e-4f;

        // Compute ray direction scaled by distance
        vec2 temp = vec2_sub(fragCoord, vec2_scale(r, 0.5f));
        vec3 dir = vec3_normalize(vec3_make(temp.x, temp.y, r.y));
        dir = vec3_scale(dir, z);
        q = vec4_make(dir.x, dir.y, dir.z, 0.2f);

        // Traverse through the cave
        q.z += T / 30.0f;

        // Save sign before mirroring (creates water reflection effect)
        s = q.y + 0.1f;
        q.y = fabsf(s);

        p = q;
        p.y -= 0.11f;

        // Twist cave walls based on depth
        // Uses approximation: mat2(cos(a), sin(a), -sin(a), cos(a)) ≈ mat2(cos(a + vec4(0,11,33,0)))
        float pz2 = 2.0f * p.z;
        float m00 = cosf(0.0f - pz2);
        float m01 = cosf(11.0f - pz2);
        float m10 = cosf(33.0f - pz2);
        float m11 = cosf(0.0f - pz2);
        float new_px = m00 * p.x + m01 * p.y;
        float new_py = m10 * p.x + m11 * p.y;
        p.x = new_px;
        p.y = new_py;

        p.y -= 0.2f;

        // Combine gyroid fields at two scales for more detail
        d = fabsf(g(p, 8.0f) - g(p, 24.0f)) / 4.0f;

        // Base glow color varies with distance from center
        float qz5 = 5.0f * q.z;
        p = vec4_make(
            1.0f + cosf(0.7f * U.x + qz5),
            1.0f + cosf(0.7f * U.y + qz5),
            1.0f + cosf(0.7f * U.z + qz5),
            1.0f + cosf(0.7f * U.w + qz5)
        );
    }

    // Add pulsing glow for the "tunnelwisp"
    float pulse = 1.4f + sinf(T) * sinf(1.7f * T) * sinf(2.3f * T);
    float len_qxy = sqrtf(q.x * q.x + q.y * q.y);
    vec4 glow = vec4_scale(U, pulse * 1000.0f / len_qxy);
    o = vec4_add(o, glow);

    // Apply tanh for soft tone mapping
    o = vec4_scale(o, 1.0f / 100000.0f);
    return vec3_make(tanhf(o.x), tanhf(o.y), tanhf(o.z));
}

// Audio synthesis function (port of mainSound from audio shader)
static vec2 mainSound(int samp, float t) {
    (void)samp;  // sample index not used directly
    vec2 r = {0.0f, 0.0f};

    for (float i = 0.0f; ++i < 4.0f; ) {
        for (float j = 0.0f; ++j < 5.0f; ) {
            float a = t * j / 32.0f + i / 3.0f;
            float b = glsl_fract(a);

            // n is the base frequency vector, m will accumulate harmonics
            vec2 n = vec2_make(t, t + 3.0f);
            n = vec2_add(n, vec2_scale(vec2_make(1.0f, 1.0f), t / j));
            vec2 m = {0.0f, 0.0f};

            // Create rich harmonic content by summing sine waves with increasing frequencies
            for (float c = 3.0f; c < 4.1f; c *= 1.02f) {
                m.x += sinf(n.x * c) / c;
                m.y += sinf(n.y * c) / c;
                n.x += c;
                n.y += c;
            }

            // Only contribute sound during the first part of the composition
            if (a < 9.0f) {
                // Complex waveform combining multiple modulations
                float slow_mod = 4.0f * sinf(t / j / 47.0f);
                float exp_pitch = exp2f(glsl_mod(a - b, 3.0f) / 6.0f + 8.5f);
                float mod_wave = sinf(exp_pitch * t * j * i + i + j);
                float carrier_arg_x = m.x + slow_mod * mod_wave;
                float carrier_arg_y = m.y + slow_mod * mod_wave;
                float envelope = exp2f(-b * 12.0f - 1.0f / b + 6.0f - (i + j) / 3.0f);

                r.x += sinf(carrier_arg_x) * envelope;
                r.y += sinf(carrier_arg_y) * envelope;
            }
        }
    }

    return r;
}

int main(void) {
    nanomp4h264_t enc;
    nanomp4h264_config_t config = {
        .width = WIDTH,
        .height = HEIGHT,
        .fps_num = FPS,
        .fps_den = 1,
        .audio_sample_rate = SAMPLE_RATE,
        .audio_channels = AUDIO_CHANNELS,
    };

    nanomp4h264_open(&enc, &config, "tunnel.mp4");

    if (nanomp4h264_get_error(&enc) != 0) {
        fprintf(stderr, "Failed to open encoder: %d\n", nanomp4h264_get_error(&enc));
        return 1;
    }

    uint8_t *rgb = malloc(WIDTH * HEIGHT * 3);
    if (!rgb) {
        fprintf(stderr, "Failed to allocate RGB buffer\n");
        return 1;
    }

    // Samples per frame
    int samples_per_frame = SAMPLE_RATE / FPS;
    int16_t *audio_buf = malloc(samples_per_frame * AUDIO_CHANNELS * sizeof(int16_t));
    if (!audio_buf) {
        fprintf(stderr, "Failed to allocate audio buffer\n");
        free(rgb);
        return 1;
    }

    printf("Rendering %d frames at %dx%d with audio...\n", TOTAL_FRAMES, WIDTH, HEIGHT);

    vec2 audio_prev_in = mainSound(0, 0.0f);
    vec2 audio_prev_out = vec2_make(0.0f, 0.0f);

    for (int frame = 0; frame < TOTAL_FRAMES; frame++) {
        float iTime = (float)frame / (float)FPS;

        // Render video frame
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

        // Generate audio for this frame
        for (int s = 0; s < samples_per_frame; s++) {
            float t = iTime + (float)s / (float)SAMPLE_RATE;
            vec2 raw_sample = mainSound(s, t);

            // High pass filter
            const float R = 0.995f;
            vec2 sample = vec2_make(
                raw_sample.x - audio_prev_in.x + R * audio_prev_out.x,
                raw_sample.y - audio_prev_in.y + R * audio_prev_out.y);
            audio_prev_in = raw_sample;
            audio_prev_out = sample;

            // Clamp to [-1, 1] and convert to int16
            int16_t left = (int16_t)(fmaxf(-1.0f, fminf(1.0f, sample.x)) * 32767.0f);
            int16_t right = (int16_t)(fmaxf(-1.0f, fminf(1.0f, sample.y)) * 32767.0f);
            audio_buf[s * 2] = left;
            audio_buf[s * 2 + 1] = right;
        }
        nanomp4h264_write_audio(&enc, audio_buf, samples_per_frame * AUDIO_CHANNELS * sizeof(int16_t),
                                NANOMP4H264_AUDIO_FORMAT_PCM16);

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

    free(audio_buf);
    free(rgb);
    nanomp4h264_close(&enc);

    int err = nanomp4h264_get_error(&enc);
    if (err != 0) {
        fprintf(stderr, "Encoding error: %d\n", err);
        return 1;
    }

    printf("Done! Output: tunnel.mp4\n");
    return 0;
}
