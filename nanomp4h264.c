// nanomp4h264 - Minimal MP4+H.264 encoder using I_PCM mode
// Single file, no dependencies beyond libc

#include "nanomp4h264.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Macros to expand values to big-endian bytes in array initializers
#define BE32(x) ((x) >> 24) & 0xFF, ((x) >> 16) & 0xFF, ((x) >> 8) & 0xFF, (x) & 0xFF
#define BE16(x) ((x) >> 8) & 0xFF, (x) & 0xFF

#define LE16(x) (x) & 0xFF, ((x) >> 8) & 0xFF

// Write a big-endian 32-bit value to a byte buffer
static void store_be32(uint8_t *p, uint32_t val) {
    p[0] = (val >> 24) & 0xFF;
    p[1] = (val >> 16) & 0xFF;
    p[2] = (val >> 8) & 0xFF;
    p[3] = val & 0xFF;
}

// Read a big-endian 32-bit value from a byte buffer
static uint32_t load_be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

// Append an stsc entry if samples_per_chunk changed from the last entry.
// Returns 0 on success, non-zero on allocation failure.
static int nanomp4h264_stsc_append(nanomp4h264_stsc_state_t *stsc, uint32_t chunk_index, uint32_t samples_per_chunk) {
    // Check if we need a new entry
    int need_new_entry = 0;
    if (stsc->entry_count == 0) {
        need_new_entry = 1;
    } else {
        uint32_t last_samples = load_be32(stsc->data + (stsc->entry_count - 1) * 12 + 4);
        if (samples_per_chunk != last_samples) {
            need_new_entry = 1;
        }
    }

    if (need_new_entry) {
        // Grow array if needed
        if (stsc->entry_count >= stsc->capacity) {
            uint32_t new_capacity = stsc->capacity == 0 ? 4 : stsc->capacity * 2;
            uint8_t *new_data = realloc(stsc->data, new_capacity * 12);
            if (!new_data) return 1;
            stsc->data = new_data;
            stsc->capacity = new_capacity;
        }

        // Append entry
        uint8_t *entry = stsc->data + stsc->entry_count * 12;
        store_be32(entry + 0, chunk_index + 1);  // 1-indexed
        store_be32(entry + 4, samples_per_chunk);
        store_be32(entry + 8, 1);  // sample_description_index
        stsc->entry_count++;
    }
    return 0;
}

// Append a chunk offset. Returns 0 on success, non-zero on allocation failure.
static int nanomp4h264_stco_append(nanomp4h264_stco_state_t *stco, uint32_t offset) {
    if (stco->count >= stco->capacity) {
        uint32_t new_capacity = stco->capacity == 0 ? 16 : stco->capacity * 2;
        uint8_t *new_data = realloc(stco->data, new_capacity * 4);
        if (!new_data) return 1;
        stco->data = new_data;
        stco->capacity = new_capacity;
    }
    store_be32(stco->data + stco->count * 4, offset);
    stco->count++;
    return 0;
}

// Bitstream writer for Exp-Golomb encoding
typedef struct {
    uint8_t *start;          // Output buffer start
    uint8_t *buf;            // Current write position
    uint64_t to_write;       // Bit accumulator (MSB-aligned)
    int      bits_to_write;  // Valid bits in to_write (0-63)
} bitstream_t;

static void bs_init(bitstream_t *bs, uint8_t *buf, int cap) {
    (void)cap;
    bs->start = buf;
    bs->buf = buf;
    bs->to_write = 0;
    bs->bits_to_write = 0;
}

static void bs_write_bits(bitstream_t *bs, uint32_t val, int n) {
    bs->to_write |= (uint64_t)val << (64 - bs->bits_to_write - n);
    bs->bits_to_write += n;
    while (bs->bits_to_write >= 8) {
        *bs->buf++ = (uint8_t)(bs->to_write >> 56);
        bs->to_write <<= 8;
        bs->bits_to_write -= 8;
    }
}

// Like stdc_first_leading_one.
static int first_leading_one_u32(uint32_t val) {
    int bits = 0;
    while (val) { bits++; val >>= 1; }
    return bits;
}

static void bs_write_ue(bitstream_t *bs, uint32_t val) {
    val++;
    int val_bits = first_leading_one_u32(val);
    int padding_bits = val_bits - 1;
    bs_write_bits(bs, val, val_bits + padding_bits);
}

static void bs_byte_align(bitstream_t *bs) {
    if (bs->bits_to_write > 0) {
        *bs->buf++ = (uint8_t)(bs->to_write >> 56);
        bs->to_write = 0;
        bs->bits_to_write = 0;
    }
}

// Call bs_byte_align(bs) before calling this function.
static int bs_pos(bitstream_t *bs) {
    assert(bs->bits_to_write == 0);
    return (int)(bs->buf - bs->start);
}

// RGB to YUV420 for a single 16x16 macroblock
static void rgb_to_yuv420_mb(const uint8_t *rgb, int rgb_w, int rgb_h,
                             int mb_x, int mb_y,
                             uint8_t *y_out,   // 256 bytes (16x16)
                             uint8_t *cb_out,  // 64 bytes (8x8)
                             uint8_t *cr_out)  // 64 bytes (8x8)
{
    int base_x = mb_x * 16;
    int base_y = mb_y * 16;

    // Convert Y plane (16x16)
    for (int row = 0; row < 16; row++) {
        for (int col = 0; col < 16; col++) {
            int src_y = base_y + row;
            int src_x = base_x + col;
            // Clamp to image bounds for edge padding
            if (src_y >= rgb_h) src_y = rgb_h - 1;
            if (src_x >= rgb_w) src_x = rgb_w - 1;
            const uint8_t *p = rgb + (src_y * rgb_w + src_x) * 3;
            int r = p[0], g = p[1], b = p[2];
            int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
            y_out[row * 16 + col] = y < 0 ? 0 : (y > 255 ? 255 : y);
        }
    }

    // Convert Cb/Cr planes (8x8, 2x2 subsampling)
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            int src_y = base_y + row * 2;
            int src_x = base_x + col * 2;
            // Clamp to image bounds for edge padding
            if (src_y >= rgb_h) src_y = rgb_h - 1;
            if (src_x >= rgb_w) src_x = rgb_w - 1;
            const uint8_t *p = rgb + (src_y * rgb_w + src_x) * 3;
            int r = p[0], g = p[1], b = p[2];
            int u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
            int v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
            cb_out[row * 8 + col] = u < 0 ? 0 : (u > 255 ? 255 : u);
            cr_out[row * 8 + col] = v < 0 ? 0 : (v > 255 ? 255 : v);
        }
    }
}

#define CONCAT(x, y) CONCAT_IMPL(x, y)
#define CONCAT_IMPL(x, y) x##y

#define WRITE_CONST(...) WRITE_CONST_IMPL(CONCAT(data_, __COUNTER__), __VA_ARGS__)
#define WRITE_CONST_IMPL(var_name, ...) \
    static const uint8_t var_name[] = __VA_ARGS__; \
    fwrite(var_name, 1, sizeof(var_name), f)

#define WRITE_DYNAMIC(...) WRITE_DYNAMIC_IMPL(CONCAT(data_, __COUNTER__), __VA_ARGS__)
#define WRITE_DYNAMIC_IMPL(var_name, ...) \
    uint8_t var_name[] = __VA_ARGS__; \
    fwrite(var_name, 1, sizeof(var_name), f)

void nanomp4h264_open(nanomp4h264_t *enc, const nanomp4h264_config_t *config,
                      const char *filepath) {
    memset(enc, 0, sizeof(*enc));

    enc->_width = config->width;
    enc->_height = config->height;
    enc->_fps_num = config->fps_num;
    enc->_fps_den = config->fps_den;
    enc->_audio_sample_rate = config->audio_sample_rate;
    enc->_audio_channels = config->audio_channels;

    // Compute padded dimensions (multiple of 16)
    enc->_padded_width = (enc->_width + 15) & ~15;
    enc->_padded_height = (enc->_height + 15) & ~15;
    enc->_mb_width = enc->_padded_width / 16;
    enc->_mb_height = enc->_padded_height / 16;
    enc->_crop_right = enc->_padded_width - enc->_width;
    enc->_crop_bottom = enc->_padded_height - enc->_height;
    enc->_frame_nal_size = (384 + 2) * enc->_mb_width * enc->_mb_height + 3;

    enc->_file = fopen(filepath, "wb");
    if (!enc->_file) {
        enc->_error = 1;
        return;
    }

    FILE *f = enc->_file;

    WRITE_CONST({
        // ftyp
        BE32(28),           // size
        'f', 't', 'y', 'p', // type
        'i', 's', 'o', 'm', // major_brand
        BE32(0x200),        // minor_version
        'i', 's', 'o', 'm', // compatible_brands[0]
        'a', 'v', 'c', '1', // compatible_brands[1]
        'm', 'p', '4', '1', // compatible_brands[2]

        // mdat
        // mdat_start_pos will be here.
        BE32(0),            // size
        'm', 'd', 'a', 't', // type
    });
    enc->_mdat_start_pos = ftell(f) - 8;
}

void nanomp4h264_write_frame(nanomp4h264_t *enc, const uint8_t *data,
                             nanomp4h264_format_t format) {
    if (enc->_error) return;

    (void)format;  // Only RGB888 supported currently

    FILE *f = enc->_file;
    int mb_count = enc->_mb_width * enc->_mb_height;
    uint32_t bytes_written = 0;

    // Record chunk offset before writing frame data
    if (nanomp4h264_stco_append(&enc->_video_stco, (uint32_t)ftell(f)) != 0) {
        enc->_error = 1;
        return;
    }

    // Update video stsc data (RLE-encoded, 1 sample per chunk)
    if (nanomp4h264_stsc_append(&enc->_video_stsc, enc->_frame_count, 1) != 0) {
        enc->_error = 1;
        return;
    }

    WRITE_DYNAMIC({ BE32(enc->_frame_nal_size) });

    // NAL header (IDR slice, nal_ref_idc=3, nal_unit_type=5) + slice header + first MB header
    // Slice header bits:
    //   1       ue(0)  first_mb_in_slice
    //   011     ue(2)  slice_type = I
    //   1       ue(0)  pic_parameter_set_id
    //   0000    u(4)   frame_num (log2_max_frame_num=4)
    //   1       ue(0)  idr_pic_id
    //   0       u(1)   no_output_of_prior_pics_flag
    //   0       u(1)   long_term_reference_flag
    //   1       se(0)  slice_qp_delta
    // First MB header bits:
    //   000011010  ue(25)  mb_type = I_PCM
    //   00         pcm_alignment_zero_bit (pad to byte)
    static const uint8_t slice_and_first_mb[] = {
        0x65,  // NAL header
        0xB8,  // 1_011_1_000: ue(0), ue(2), ue(0), frame_num[3:1]
        0x48,  // 0_1_0_0_1_000: frame_num[0], ue(0), 0, 0, se(0), ue(25)[8:6]
        0x68,  // 0_11010_00: ue(25)[5:0], alignment
    };
    fwrite(slice_and_first_mb, 1, sizeof(slice_and_first_mb), f);
    bytes_written += sizeof(slice_and_first_mb);

    // Subsequent MB headers: ue(25) + alignment = 0x0D 0x00
    // 00001101 0_0000000: ue(25), alignment
    static const uint8_t mb_hdr[] = {0x0D, 0x00};

    uint8_t mb_yuv[384];  // 256 Y + 64 Cb + 64 Cr

    for (int mb = 0; mb < mb_count; mb++) {
        if (mb != 0) {
            fwrite(mb_hdr, 1, sizeof(mb_hdr), f);
            bytes_written += sizeof(mb_hdr);
        }

        // Convert this macroblock from RGB to YUV420 on-the-fly
        int mb_x = mb % enc->_mb_width;
        int mb_y = mb / enc->_mb_width;
        uint8_t *y_out = mb_yuv;
        uint8_t *cb_out = mb_yuv + 16*16;
        uint8_t *cr_out = cb_out + 8*8;

        rgb_to_yuv420_mb(data, enc->_width, enc->_height, mb_x, mb_y,
                         y_out, cb_out, cr_out);

        // Write raw samples: 256 Y, 64 Cb, 64 Cr
        fwrite(mb_yuv, 1, sizeof(mb_yuv), f);
        bytes_written += sizeof(mb_yuv);
    }

    // RBSP trailing bits (stop bit + alignment)
    uint8_t trailing = 0x80;
    fwrite(&trailing, 1, 1, f);
    bytes_written += 1;

    assert(bytes_written == enc->_frame_nal_size);

    enc->_frame_count++;
}

static int host_is_little_endian(void) {
    uint16_t test = 0x0102;
    uint8_t b;
    memcpy(&b, &test, 1);
    return b == 0x02;
}

void nanomp4h264_write_audio(nanomp4h264_t *enc, const void *data, size_t data_size,
                             nanomp4h264_audio_format_t format) {
    if (enc->_error) return;

    (void)format;  // Only PCM16 supported currently

    // Validate audio is configured
    if (enc->_audio_sample_rate <= 0 || enc->_audio_channels <= 0) {
        enc->_error = 1;
        return;
    }

    FILE *f = enc->_file;

    // Calculate sample count: data_size / (2 bytes * channels)
    uint32_t sample_count = (uint32_t)(data_size / (2 * enc->_audio_channels));

    // Record chunk offset
    if (nanomp4h264_stco_append(&enc->_audio_stco, (uint32_t)ftell(f)) != 0) {
        enc->_error = 1;
        return;
    }

    // Update audio stsc data (RLE-encoded)
    if (nanomp4h264_stsc_append(&enc->_audio_stsc, enc->_audio_chunk_count, sample_count) != 0) {
        enc->_error = 1;
        return;
    }

    enc->_audio_chunk_count++;
    enc->_audio_total_samples += sample_count;

    if (host_is_little_endian()) {
        fwrite(data, 1, data_size, f);
    } else {
        // Convert each 16-bit sample to little endian.
        const uint16_t *src = (const uint16_t *)data;
        for (size_t i = 0; i < data_size; i += 2) {
            WRITE_DYNAMIC({ LE16(src[i]) });
        }
    }
}

void nanomp4h264_flush(nanomp4h264_t *enc) {
    if (enc->_error || enc->_frame_count == 0) return;

    FILE *f = enc->_file;
    long end_pos = ftell(f);

    // Build SPS
    uint8_t sps[256];
    int sps_len;
    {
        bitstream_t bs;
        bs_init(&bs, sps, sizeof(sps));

        // NAL header
        bs_write_bits(&bs, 0x67, 8);

        // SPS
        bs_write_bits(&bs, 66, 8);   // profile_idc = Baseline
        bs_write_bits(&bs, 0x80, 8); // constraint_set0_flag=1 (Baseline compatible)
        bs_write_bits(&bs, 40, 8);   // level_idc = 4.0

        bs_write_ue(&bs, 0);  // sps_id
        bs_write_ue(&bs, 0);  // log2_max_frame_num_minus4
        bs_write_ue(&bs, 2);  // pic_order_cnt_type
        bs_write_ue(&bs, 0);  // max_num_ref_frames
        bs_write_bits(&bs, 0, 1);  // gaps_in_frame_num_allowed
        bs_write_ue(&bs, enc->_mb_width - 1);   // pic_width_in_mbs_minus1
        bs_write_ue(&bs, enc->_mb_height - 1);  // pic_height_in_map_units_minus1
        bs_write_bits(&bs, 1, 1);  // frame_mbs_only_flag
        bs_write_bits(&bs, 1, 1);  // direct_8x8_inference_flag

        // Frame cropping
        int need_crop = enc->_crop_right > 0 || enc->_crop_bottom > 0;
        bs_write_bits(&bs, need_crop ? 1 : 0, 1);
        if (need_crop) {
            bs_write_ue(&bs, 0);                    // left
            bs_write_ue(&bs, enc->_crop_right / 2);  // right (chroma units)
            bs_write_ue(&bs, 0);                    // top
            bs_write_ue(&bs, enc->_crop_bottom / 2); // bottom
        }

        bs_write_bits(&bs, 0, 1);  // vui_parameters_present

        // RBSP trailing
        bs_write_bits(&bs, 1, 1);
        bs_byte_align(&bs);

        sps_len = bs_pos(&bs);
    }

    // PPS NAL unit (constant for this encoder)
    // Bit layout:
    //   0x68      NAL header (nal_ref_idc=3, nal_unit_type=8)
    //   1         ue(0)  pps_id
    //   1         ue(0)  sps_id
    //   0         u(1)   entropy_coding_mode (CAVLC)
    //   0         u(1)   bottom_field_pic_order
    //   1         ue(0)  num_slice_groups_minus1
    //   1         ue(0)  num_ref_idx_l0_default_active_minus1
    //   1         ue(0)  num_ref_idx_l1_default_active_minus1
    //   0         u(1)   weighted_pred_flag
    //   00        u(2)   weighted_bipred_idc
    //   1         se(0)  pic_init_qp_minus26
    //   1         se(0)  pic_init_qs_minus26
    //   1         se(0)  chroma_qp_index_offset
    //   0         u(1)   deblocking_filter_control
    //   0         u(1)   constrained_intra_pred
    //   0         u(1)   redundant_pic_cnt_present
    //   1         u(1)   RBSP stop bit
    //   0000000         alignment
    static const uint8_t pps[] = {
        0x68,  // NAL header
        0xCE,  // 11_00_1110: ue(0), ue(0), 0, 0, ue(0), ue(0), ue(0), 0
        0x38,  // 00_111_000: 00, se(0), se(0), se(0), 0, 0, 0
        0x80,  // 1_0000000: stop bit, alignment
    };

    uint32_t duration = enc->_frame_count * enc->_fps_den;
    uint32_t timescale = enc->_fps_num;
    uint32_t sample_size = enc->_frame_nal_size + 4;

    // Determine if audio exists
    int has_audio = (enc->_audio_sample_rate > 0 && enc->_audio_channels > 0 && enc->_audio_chunk_count > 0);

    // Use pre-computed stsc entry counts
    uint32_t stsc_entry_count = enc->_video_stsc.entry_count;
    uint32_t audio_stsc_entry_count = enc->_audio_stsc.entry_count;

    // Calculate box sizes (bottom-up)
    uint32_t stco_size = 16 + 4 * enc->_video_stco.count;  // header(8) + version/flags(4) + entry_count(4) + offsets(4*N)
    enum { STSZ_SIZE = 20 };
    uint32_t stsc_size = 16 + 12 * stsc_entry_count;  // header(8) + version/flags(4) + entry_count(4) + entries(12*N)
    enum { STTS_SIZE = 24 };
    uint32_t avcC_size = 8 + 8 + sps_len + 3 + sizeof(pps);  // header + config(6) + sps_len_field(2) + sps + numPPS(1) + pps_len_field(2) + pps
    uint32_t avc1_size = 8 + 78 + avcC_size;
    uint32_t stsd_size = 8 + 8 + avc1_size;
    uint32_t stbl_size = 8 + stsd_size + STTS_SIZE + stsc_size + STSZ_SIZE + stco_size;
    enum { DREF_SIZE = 28 };
    enum { DINF_SIZE = 36 };
    enum { VMHD_SIZE = 20 };
    uint32_t minf_size = 8 + VMHD_SIZE + DINF_SIZE + stbl_size;
    enum { HDLR_SIZE = 45 };
    enum { MDHD_SIZE = 32 };
    uint32_t mdia_size = 8 + MDHD_SIZE + HDLR_SIZE + minf_size;
    enum { TKHD_SIZE = 92 };
    uint32_t trak_size = 8 + TKHD_SIZE + mdia_size;
    enum { MVHD_SIZE = 108 };

    enum { SOWT_SIZE = 36 };
    enum { SMHD_SIZE = 16 };
    enum { AUDIO_HDLR_SIZE = 46 };
    uint32_t audio_stsd_size = 8 + 8 + SOWT_SIZE;
    uint32_t audio_stsc_size = 16 + 12 * audio_stsc_entry_count;
    uint32_t audio_stco_size = 16 + 4 * enc->_audio_stco.count;
    uint32_t audio_stbl_size = 8 + audio_stsd_size + STTS_SIZE + audio_stsc_size + STSZ_SIZE + audio_stco_size;
    uint32_t audio_minf_size = 8 + SMHD_SIZE + DINF_SIZE + audio_stbl_size;
    uint32_t audio_mdia_size = 8 + MDHD_SIZE + AUDIO_HDLR_SIZE + audio_minf_size;
    uint32_t audio_trak_size = has_audio ? 8 + TKHD_SIZE + audio_mdia_size : 0;

    uint32_t moov_size = 8 + MVHD_SIZE + trak_size + audio_trak_size;
    uint32_t next_track_id = has_audio ? 3 : 2;

    WRITE_DYNAMIC({
        // moov
        BE32(moov_size),
    });
    WRITE_CONST({
        'm', 'o', 'o', 'v',

        // mvhd
        BE32(MVHD_SIZE),         // size
        'm', 'v', 'h', 'd',      // box type
        BE32(0),                 // version, flags
        BE32(0),                 // creation_time
        BE32(0),                 // modification_time
    });
    WRITE_DYNAMIC({
        BE32(timescale),
        BE32(duration),
    });
    WRITE_CONST({
        BE32(0x00010000),        // rate (16.16 fixed)
        BE16(0x0100),            // volume (8.8 fixed)
        BE16(0),                 // reserved
        BE32(0),                 // reserved
        BE32(0),                 // reserved
        // Identity matrix (36 bytes)
        BE32(0x00010000),        // a
        BE32(0),                 // b
        BE32(0),                 // u
        BE32(0),                 // c
        BE32(0x00010000),        // d
        BE32(0),                 // v
        BE32(0),                 // x
        BE32(0),                 // y
        BE32(0x40000000),        // w
        // pre_defined (24 bytes)
        BE32(0), BE32(0), BE32(0),
        BE32(0), BE32(0), BE32(0),
    });
    WRITE_DYNAMIC({
        BE32(next_track_id),     // next_track_id

        // trak
        BE32(trak_size),
    });
    WRITE_CONST({
        't', 'r', 'a', 'k',

        // tkhd
        BE32(TKHD_SIZE),         // size
        't', 'k', 'h', 'd',      // box type
        BE32(0x03),              // version, flags (enabled + in_movie)
        BE32(0),                 // creation_time
        BE32(0),                 // modification_time
        BE32(1),                 // track_id
        BE32(0),                 // reserved
    });
    WRITE_DYNAMIC({
        BE32(duration),
    });
    WRITE_CONST({
        BE32(0),                 // reserved
        BE32(0),                 // reserved
        BE16(0),                 // layer
        BE16(0),                 // alternate_group
        BE16(0),                 // volume (video track)
        BE16(0),                 // reserved
        // Identity matrix (36 bytes)
        BE32(0x00010000),        // a
        BE32(0),                 // b
        BE32(0),                 // u
        BE32(0),                 // c
        BE32(0x00010000),        // d
        BE32(0),                 // v
        BE32(0),                 // x
        BE32(0),                 // y
        BE32(0x40000000),        // w
    });
    WRITE_DYNAMIC({
        BE32(enc->_width << 16),   // width 16.16
        BE32(enc->_height << 16),  // height 16.16
    });

    WRITE_DYNAMIC({
        // mdia
        BE32(mdia_size),
    });

    WRITE_CONST({
        'm', 'd', 'i', 'a',

        // mdhd
        BE32(MDHD_SIZE),         // size
        'm', 'd', 'h', 'd',      // box type
        BE32(0),                 // version, flags
        BE32(0),                 // creation_time
        BE32(0),                 // modification_time
    });
    WRITE_DYNAMIC({
        BE32(timescale),
        BE32(duration),
    });
    WRITE_CONST({
        BE16(0x55C4),            // language ("und")
        BE16(0),                 // pre_defined

        // hdlr
        BE32(HDLR_SIZE),         // size
        'h', 'd', 'l', 'r',      // box type
        BE32(0),                 // version, flags
        BE32(0),                 // pre_defined
        'v', 'i', 'd', 'e',      // handler_type
        BE32(0),                 // reserved
        BE32(0),                 // reserved
        BE32(0),                 // reserved
        'V', 'i', 'd', 'e', 'o', 'H', 'a', 'n', 'd', 'l', 'e', 'r', 0x00,  // name
    });

    WRITE_DYNAMIC({
        // minf
        BE32(minf_size),
    });
    WRITE_CONST({
        'm', 'i', 'n', 'f',

        // vmhd
        BE32(VMHD_SIZE),         // size
        'v', 'm', 'h', 'd',      // box type
        BE32(1),                 // version, flags
        BE16(0),                 // graphics_mode
        BE16(0),                 // opcolor[0]
        BE16(0),                 // opcolor[1]
        BE16(0),                 // opcolor[2]

        // dinf + dref
        BE32(DINF_SIZE),         // dinf size
        'd', 'i', 'n', 'f',      // dinf box type
        BE32(DREF_SIZE),         // dref size
        'd', 'r', 'e', 'f',      // dref box type
        BE32(0),                 // version, flags
        BE32(1),                 // entry_count
        BE32(12),                // url size
        'u', 'r', 'l', ' ',      // url box type
        BE32(1),                 // flags (self-contained)
    });

    WRITE_DYNAMIC({
        // stbl
        BE32(stbl_size),
        's', 't', 'b', 'l',

        // stsd
        BE32(stsd_size),
    });
    WRITE_CONST({
        's', 't', 's', 'd',      // box type
        BE32(0),                 // version, flags
        BE32(1),                 // entry_count
    });

    WRITE_DYNAMIC({
        // avc1
        BE32(avc1_size),
    });
    WRITE_CONST({
        'a', 'v', 'c', '1',      // box type
        BE32(0),                 // reserved
        BE16(0),                 // reserved
        BE16(1),                 // data_reference_index
        BE16(0),                 // pre_defined
        BE16(0),                 // reserved
        BE32(0),                 // pre_defined
        BE32(0),                 // pre_defined
        BE32(0),                 // pre_defined
    });
    WRITE_DYNAMIC({
        BE16(enc->_width),
        BE16(enc->_height),
    });
    WRITE_CONST({
        BE32(0x00480000),        // horiz_resolution (16.16 fixed)
        BE32(0x00480000),        // vert_resolution (16.16 fixed)
        BE32(0),                 // reserved
        BE16(1),                 // frame_count
        // compressor_name (32 bytes, empty pascal string)
        BE32(0), BE32(0), BE32(0), BE32(0),
        BE32(0), BE32(0), BE32(0), BE32(0),
        BE16(24),                // depth
        BE16(0xFFFF),            // pre_defined
    });

    WRITE_DYNAMIC({
        // avcC
        BE32(avcC_size),
    });
    WRITE_CONST({
        'a', 'v', 'c', 'C',      // box type
        0x01,                    // configurationVersion
    });
    WRITE_DYNAMIC({
        sps[1],   // AVCProfileIndication
        sps[2],   // profile_compatibility
        sps[3],   // AVCLevelIndication
    });
    WRITE_CONST({
        0xFF,  // lengthSizeMinusOne (4-byte lengths)
        0xE1,  // numOfSequenceParameterSets
    });

    WRITE_DYNAMIC({
        BE16(sps_len),
    });
    fwrite(sps, 1, sps_len, f);

    WRITE_CONST({
        1,        // numOfPictureParameterSets
        BE16(sizeof(pps)),
    });
    fwrite(pps, 1, sizeof(pps), f);

    WRITE_CONST({
        // stts
        BE32(STTS_SIZE),         // size
        's', 't', 't', 's',      // box type
        BE32(0),                 // version, flags
        BE32(1),                 // entry_count
    });
    WRITE_DYNAMIC({
        BE32(enc->_frame_count),
        BE32(enc->_fps_den),
    });

    WRITE_DYNAMIC({
        // stsc
        BE32(stsc_size),         // size
        's', 't', 's', 'c',
        BE32(0),                 // version, flags
        BE32(stsc_entry_count),  // entry_count
    });
    fwrite(enc->_video_stsc.data, 12, enc->_video_stsc.entry_count, f);

    WRITE_CONST({
        // stsz
        BE32(STSZ_SIZE),         // size
        's', 't', 's', 'z',      // box type
        BE32(0),                 // version, flags
    });
    WRITE_DYNAMIC({
        BE32(sample_size),  // sample_size (constant)
        BE32(enc->_frame_count),
    });

    WRITE_DYNAMIC({
        // stco
        BE32(stco_size),         // size
        's', 't', 'c', 'o',
        BE32(0),                 // version, flags
        BE32(enc->_video_stco.count), // entry_count
    });
    fwrite(enc->_video_stco.data, 4, enc->_video_stco.count, f);

    // Write audio track if present
    if (has_audio) {
        uint32_t audio_timescale = enc->_audio_sample_rate;
        uint32_t audio_duration = (uint32_t)enc->_audio_total_samples;
        uint32_t audio_sample_size = 2 * enc->_audio_channels;
        // Duration in tkhd (movie timescale)
        uint32_t tkhd_audio_duration = (uint32_t)(((uint64_t)enc->_audio_total_samples * timescale) / audio_timescale);

        WRITE_DYNAMIC({
            // trak (audio)
            BE32(audio_trak_size),
        });
        WRITE_CONST({
            't', 'r', 'a', 'k',

            // tkhd
            BE32(TKHD_SIZE),         // size
            't', 'k', 'h', 'd',      // box type
            BE32(0x03),              // version, flags (enabled + in_movie)
            BE32(0),                 // creation_time
            BE32(0),                 // modification_time
            BE32(2),                 // track_id
            BE32(0),                 // reserved
        });
        WRITE_DYNAMIC({
            BE32(tkhd_audio_duration),
        });
        WRITE_CONST({
            BE32(0),                 // reserved
            BE32(0),                 // reserved
            BE16(0),                 // layer
            BE16(0),                 // alternate_group
            BE16(0x0100),            // volume (1.0 for audio)
            BE16(0),                 // reserved
            // Identity matrix (36 bytes)
            BE32(0x00010000),        // a
            BE32(0),                 // b
            BE32(0),                 // u
            BE32(0),                 // c
            BE32(0x00010000),        // d
            BE32(0),                 // v
            BE32(0),                 // x
            BE32(0),                 // y
            BE32(0x40000000),        // w
            BE32(0),                 // width (0 for audio)
            BE32(0),                 // height (0 for audio)
        });

        WRITE_DYNAMIC({
            // mdia
            BE32(audio_mdia_size),
        });
        WRITE_CONST({
            'm', 'd', 'i', 'a',

            // mdhd
            BE32(MDHD_SIZE),         // size
            'm', 'd', 'h', 'd',      // box type
            BE32(0),                 // version, flags
            BE32(0),                 // creation_time
            BE32(0),                 // modification_time
        });
        WRITE_DYNAMIC({
            BE32(audio_timescale),
            BE32(audio_duration),
        });
        WRITE_CONST({
            BE16(0x55C4),            // language ("und")
            BE16(0),                 // pre_defined

            // hdlr (sound)
            BE32(AUDIO_HDLR_SIZE),   // size
            'h', 'd', 'l', 'r',      // box type
            BE32(0),                 // version, flags
            BE32(0),                 // pre_defined
            's', 'o', 'u', 'n',      // handler_type
            BE32(0),                 // reserved
            BE32(0),                 // reserved
            BE32(0),                 // reserved
            'S', 'o', 'u', 'n', 'd', 'H', 'a', 'n', 'd', 'l', 'e', 'r', 0x00, 0x00,  // name (14 bytes with NUL padding)
        });

        WRITE_DYNAMIC({
            // minf
            BE32(audio_minf_size),
        });
        WRITE_CONST({
            'm', 'i', 'n', 'f',

            // smhd
            BE32(SMHD_SIZE),         // size
            's', 'm', 'h', 'd',      // box type
            BE32(0),                 // version, flags
            BE16(0),                 // balance
            BE16(0),                 // reserved

            // dinf + dref
            BE32(DINF_SIZE),         // dinf size
            'd', 'i', 'n', 'f',      // dinf box type
            BE32(DREF_SIZE),         // dref size
            'd', 'r', 'e', 'f',      // dref box type
            BE32(0),                 // version, flags
            BE32(1),                 // entry_count
            BE32(12),                // url size
            'u', 'r', 'l', ' ',      // url box type
            BE32(1),                 // flags (self-contained)
        });

        WRITE_DYNAMIC({
            // stbl
            BE32(audio_stbl_size),
        });
        WRITE_CONST({
            's', 't', 'b', 'l',
        });

        WRITE_DYNAMIC({
            // stsd
            BE32(audio_stsd_size),
        });
        WRITE_CONST({
            's', 't', 's', 'd',      // box type
            BE32(0),                 // version, flags
            BE32(1),                 // entry_count

            // sowt (PCM16LE sample entry)
            BE32(SOWT_SIZE),
            's', 'o', 'w', 't',      // type (little-endian PCM)
            BE32(0),                 // reserved
            BE16(0),                 // reserved
            BE16(1),                 // data_reference_index
            BE16(0),                 // version
            BE16(0),                 // revision
            BE32(0),                 // vendor
        });
        WRITE_DYNAMIC({
            BE16(enc->_audio_channels),
            BE16(16),                // sampleSize (bits)
            BE16(0),                 // compressionID
            BE16(0),                 // packetSize
            BE32(enc->_audio_sample_rate << 16),  // sampleRate (16.16 fixed)
        });

        WRITE_CONST({
            // stts
            BE32(STTS_SIZE),         // size
            's', 't', 't', 's',      // box type
            BE32(0),                 // version, flags
            BE32(1),                 // entry_count
        });
        WRITE_DYNAMIC({
            BE32((uint32_t)enc->_audio_total_samples),
            BE32(1),                 // delta=1 (each sample is 1 time unit)

            // stsc
            BE32(audio_stsc_size),   // size
            's', 't', 's', 'c',
            BE32(0),                 // version, flags
            BE32(audio_stsc_entry_count),  // entry_count
        });
        fwrite(enc->_audio_stsc.data, 12, enc->_audio_stsc.entry_count, f);

        WRITE_CONST({
            // stsz
            BE32(STSZ_SIZE),         // size
            's', 't', 's', 'z',      // box type
            BE32(0),                 // version, flags
        });
        WRITE_DYNAMIC({
            BE32(audio_sample_size), // sample_size (constant: 2 * channels)
            BE32((uint32_t)enc->_audio_total_samples),

            // stco
            BE32(audio_stco_size),   // size
            's', 't', 'c', 'o',
            BE32(0),                 // version, flags
            BE32(enc->_audio_stco.count), // entry_count
        });
        fwrite(enc->_audio_stco.data, 4, enc->_audio_stco.count, f);
    }

    // Fix mdat size
    long final_pos = ftell(f);
    uint32_t mdat_size = (uint32_t)(end_pos - enc->_mdat_start_pos);
    fseek(f, enc->_mdat_start_pos, SEEK_SET);
    WRITE_DYNAMIC({ BE32(mdat_size) });
    fseek(f, end_pos, SEEK_SET);
    fflush(f);
}

void nanomp4h264_close(nanomp4h264_t *enc) {
    if (!enc) return;
    if (enc->_frame_count > 0 && !enc->_error) {
        nanomp4h264_flush(enc);
    }
    if (enc->_file) fclose(enc->_file);
    enc->_file = NULL;
    free(enc->_video_stco.data);
    free(enc->_video_stsc.data);
    enc->_video_stco.data = NULL;
    enc->_video_stsc.data = NULL;
    free(enc->_audio_stco.data);
    free(enc->_audio_stsc.data);
    enc->_audio_stco.data = NULL;
    enc->_audio_stsc.data = NULL;
}

int nanomp4h264_get_error(const nanomp4h264_t *enc) {
    return enc ? enc->_error : 1;
}
