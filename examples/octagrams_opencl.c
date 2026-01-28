// octagrams_opencl.c - OpenCL port of the octagrams shader
//
// Original shader: https://www.shadertoy.com/view/tlVGDt
// Created by whisky_shusuky in 2020-01-28
// Licensed under CC BY-NC-SA 3.0 Unported License

#include "../nanomp4h264.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define WIDTH 100
#define HEIGHT 100
#define FPS 30
#define DURATION 5
#define TOTAL_FRAMES (FPS * DURATION)

// OpenCL kernel source - port of the GLSL shader
static const char *kernel_source =
"typedef struct { float x, y; } vec2;\n"
"typedef struct { float x, y, z; } vec3;\n"
"typedef struct { float m00, m01, m10, m11; } mat2;\n"
"\n"
"vec2 mat2_mul_vec2(mat2 m, vec2 v) {\n"
"    vec2 r; r.x = m.m00 * v.x + m.m01 * v.y; r.y = m.m10 * v.x + m.m11 * v.y;\n"
"    return r;\n"
"}\n"
"\n"
"mat2 rot(float a) {\n"
"    float c = cos(a), s = sin(a);\n"
"    mat2 m; m.m00 = c; m.m01 = s; m.m10 = -s; m.m11 = c;\n"
"    return m;\n"
"}\n"
"\n"
"float glsl_mod(float x, float y) { return x - y * floor(x / y); }\n"
"\n"
"float sdBox(vec3 p, vec3 b) {\n"
"    vec3 q;\n"
"    q.x = fabs(p.x) - b.x;\n"
"    q.y = fabs(p.y) - b.y;\n"
"    q.z = fabs(p.z) - b.z;\n"
"    float maxq = fmax(q.x, fmax(q.y, q.z));\n"
"    vec3 qc; qc.x = fmax(q.x, 0.0f); qc.y = fmax(q.y, 0.0f); qc.z = fmax(q.z, 0.0f);\n"
"    return sqrt(qc.x*qc.x + qc.y*qc.y + qc.z*qc.z) + fmin(maxq, 0.0f);\n"
"}\n"
"\n"
"float box(vec3 pos, float scale) {\n"
"    pos.x *= scale; pos.y *= scale; pos.z *= scale;\n"
"    vec3 b; b.x = 0.4f; b.y = 0.4f; b.z = 0.1f;\n"
"    float base = sdBox(pos, b) / 1.5f;\n"
"    return -base;\n"
"}\n"
"\n"
"float box_set(vec3 pos, float gTime, mat2 rot_0_8) {\n"
"    float sin_gTime = sin(gTime * 0.4f);\n"
"    float abs_sin_gTime = fabs(sin_gTime);\n"
"    float scale = 2.0f - abs_sin_gTime * 1.5f;\n"
"\n"
"    vec3 p1 = pos; p1.y += sin_gTime * 2.5f;\n"
"    vec2 xy1 = mat2_mul_vec2(rot_0_8, (vec2){p1.x, p1.y});\n"
"    p1.x = xy1.x; p1.y = xy1.y;\n"
"    float b1 = box(p1, scale);\n"
"\n"
"    vec3 p2 = pos; p2.y -= sin_gTime * 2.5f;\n"
"    vec2 xy2 = mat2_mul_vec2(rot_0_8, (vec2){p2.x, p2.y});\n"
"    p2.x = xy2.x; p2.y = xy2.y;\n"
"    float b2 = box(p2, scale);\n"
"\n"
"    vec3 p3 = pos; p3.x += sin_gTime * 2.5f;\n"
"    vec2 xy3 = mat2_mul_vec2(rot_0_8, (vec2){p3.x, p3.y});\n"
"    p3.x = xy3.x; p3.y = xy3.y;\n"
"    float b3 = box(p3, scale);\n"
"\n"
"    vec3 p4 = pos; p4.x -= sin_gTime * 2.5f;\n"
"    vec2 xy4 = mat2_mul_vec2(rot_0_8, (vec2){p4.x, p4.y});\n"
"    p4.x = xy4.x; p4.y = xy4.y;\n"
"    float b4 = box(p4, scale);\n"
"\n"
"    vec3 p5 = pos;\n"
"    vec2 xy5 = mat2_mul_vec2(rot_0_8, (vec2){p5.x, p5.y});\n"
"    p5.x = xy5.x; p5.y = xy5.y;\n"
"    float b5 = box(p5, 0.5f) * 6.0f;\n"
"\n"
"    float b6 = box(pos, 0.5f) * 6.0f;\n"
"\n"
"    return fmax(fmax(fmax(fmax(fmax(b1, b2), b3), b4), b5), b6);\n"
"}\n"
"\n"
"__kernel void render(__global uchar *output, int width, int height, float iTime) {\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    if (x >= width || y >= height) return;\n"
"\n"
"    // Pre-compute rotation matrix\n"
"    mat2 rot_0_8 = rot(0.8f);\n"
"\n"
"    // Normalized pixel coordinates\n"
"    float fx = (float)x + 0.5f;\n"
"    float fy = (float)(height - 1 - y) + 0.5f;\n"
"    float minRes = fmin((float)width, (float)height);\n"
"    float px = (2.0f * fx - (float)width) / minRes;\n"
"    float py = (2.0f * fy - (float)height) / minRes;\n"
"\n"
"    // Ray origin and direction\n"
"    vec3 ro = {0.0f, -0.2f, iTime * 4.0f};\n"
"    float rayLen = sqrt(px*px + py*py + 1.5f*1.5f);\n"
"    vec3 ray = {px/rayLen, py/rayLen, 1.5f/rayLen};\n"
"\n"
"    // Rotate ray\n"
"    mat2 rotXY = rot(sin(iTime * 0.03f) * 5.0f);\n"
"    vec2 rxy = mat2_mul_vec2(rotXY, (vec2){ray.x, ray.y});\n"
"    ray.x = rxy.x; ray.y = rxy.y;\n"
"\n"
"    mat2 rotYZ = rot(sin(iTime * 0.05f) * 0.2f);\n"
"    vec2 ryz = mat2_mul_vec2(rotYZ, (vec2){ray.y, ray.z});\n"
"    ray.y = ryz.x; ray.z = ryz.y;\n"
"\n"
"    float t = 0.1f;\n"
"    float ac = 0.0f;\n"
"\n"
"    // Raymarching loop\n"
"    for (int i = 0; i < 99; i++) {\n"
"        vec3 pos;\n"
"        pos.x = ro.x + ray.x * t;\n"
"        pos.y = ro.y + ray.y * t;\n"
"        pos.z = ro.z + ray.z * t;\n"
"\n"
"        // Tiling\n"
"        pos.x = glsl_mod(pos.x - 2.0f, 4.0f) - 2.0f;\n"
"        pos.y = glsl_mod(pos.y - 2.0f, 4.0f) - 2.0f;\n"
"        pos.z = glsl_mod(pos.z - 2.0f, 4.0f) - 2.0f;\n"
"\n"
"        float gTime = iTime - (float)i * 0.01f;\n"
"        float d = box_set(pos, gTime, rot_0_8);\n"
"        d = fmax(fabs(d), 0.01f);\n"
"        ac += exp(-d * 23.0f);\n"
"        t += d * 0.55f;\n"
"    }\n"
"\n"
"    // Color calculation\n"
"    float cr = ac * 0.02f;\n"
"    float cg = ac * 0.02f + 0.2f * fabs(sin(iTime));\n"
"    float cb = ac * 0.02f + 0.5f + sin(iTime) * 0.2f;\n"
"\n"
"    // Clamp and convert to 8-bit\n"
"    int idx = (y * width + x) * 3;\n"
"    output[idx + 0] = (uchar)(fmin(fmax(cr, 0.0f), 1.0f) * 255.0f);\n"
"    output[idx + 1] = (uchar)(fmin(fmax(cg, 0.0f), 1.0f) * 255.0f);\n"
"    output[idx + 2] = (uchar)(fmin(fmax(cb, 0.0f), 1.0f) * 255.0f);\n"
"}\n";

static void check_error(cl_int err, const char *op) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL error during %s: %d\n", op, err);
        exit(1);
    }
}

int main(void) {
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem output_buf;

    // Get platform and device
    err = clGetPlatformIDs(1, &platform, NULL);
    check_error(err, "clGetPlatformIDs");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("No GPU found, trying CPU...\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    }
    check_error(err, "clGetDeviceIDs");

    // Print device info
    char device_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("OpenCL device: %s\n", device_name);

    // Create context and command queue
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check_error(err, "clCreateContext");

#ifdef CL_VERSION_2_0
    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
#else
    queue = clCreateCommandQueue(context, device, 0, &err);
#endif
    check_error(err, "clCreateCommandQueue");

    // Create and build program
    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    check_error(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "Build error:\n%s\n", log);
        exit(1);
    }

    // Create kernel
    kernel = clCreateKernel(program, "render", &err);
    check_error(err, "clCreateKernel");

    // Create output buffer
    size_t buffer_size = WIDTH * HEIGHT * 3;
    output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buffer_size, NULL, &err);
    check_error(err, "clCreateBuffer");

    // Set up video encoder
    nanomp4h264_t enc;
    nanomp4h264_config_t config = {
        .width = WIDTH,
        .height = HEIGHT,
        .fps_num = FPS,
        .fps_den = 1,
    };

    nanomp4h264_open(&enc, &config, "octagrams_opencl.mp4");
    if (nanomp4h264_get_error(&enc) != 0) {
        fprintf(stderr, "Failed to open encoder: %d\n", nanomp4h264_get_error(&enc));
        return 1;
    }

    uint8_t *rgb = malloc(buffer_size);
    if (!rgb) {
        fprintf(stderr, "Failed to allocate RGB buffer\n");
        return 1;
    }

    printf("Rendering %d frames at %dx%d using OpenCL...\n", TOTAL_FRAMES, WIDTH, HEIGHT);

    // Set constant kernel arguments
    int width = WIDTH, height = HEIGHT;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &output_buf);
    clSetKernelArg(kernel, 1, sizeof(int), &width);
    clSetKernelArg(kernel, 2, sizeof(int), &height);

    size_t global_size[2] = {WIDTH, HEIGHT};

    for (int frame = 0; frame < TOTAL_FRAMES; frame++) {
        float iTime = (float)frame / (float)FPS;

        // Set time argument
        clSetKernelArg(kernel, 3, sizeof(float), &iTime);

        // Execute kernel
        err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, NULL, 0, NULL, NULL);
        check_error(err, "clEnqueueNDRangeKernel");

        // Read back result
        err = clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, buffer_size, rgb, 0, NULL, NULL);
        check_error(err, "clEnqueueReadBuffer");

        // Write frame to video
        nanomp4h264_write_frame(&enc, rgb, NANOMP4H264_FORMAT_RGB888);

        if (nanomp4h264_get_error(&enc) != 0) {
            fprintf(stderr, "Error after frame %d: %d\n", frame, nanomp4h264_get_error(&enc));
            break;
        }

        if ((frame + 1) % 60 == 0) {
            printf("Frame %d/%d\n", frame + 1, TOTAL_FRAMES);
        }
    }

    // Cleanup
    free(rgb);
    clReleaseMemObject(output_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    nanomp4h264_close(&enc);

    int enc_err = nanomp4h264_get_error(&enc);
    if (enc_err != 0) {
        fprintf(stderr, "Encoding error: %d\n", enc_err);
        return 1;
    }

    printf("Done! Output: octagrams_opencl.mp4\n");
    return 0;
}
