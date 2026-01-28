// nanomp4h264 - Minimal MP4+H.264 encoder using I_PCM mode
// Single file, no dependencies beyond libc

#include "nanomp4h264.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Bitstream writer for Exp-Golomb encoding
typedef struct {
    uint8_t *buf;
    int cap;
    int byte_pos;
    int bit_pos;  // Bits remaining in current byte (8 = fresh byte)
} bitstream_t;

static void bs_init(bitstream_t *bs, uint8_t *buf, int cap) {
    bs->buf = buf;
    bs->cap = cap;
    bs->byte_pos = 0;
    bs->bit_pos = 8;
    if (cap > 0) buf[0] = 0;
}

static void bs_write_bits(bitstream_t *bs, uint32_t val, int n) {
    while (n > 0) {
        int bits_to_write = n < bs->bit_pos ? n : bs->bit_pos;
        int shift = bs->bit_pos - bits_to_write;
        uint32_t mask = (1 << bits_to_write) - 1;
        uint32_t bits = (val >> (n - bits_to_write)) & mask;
        bs->buf[bs->byte_pos] |= bits << shift;
        bs->bit_pos -= bits_to_write;
        n -= bits_to_write;
        if (bs->bit_pos == 0) {
            bs->byte_pos++;
            if (bs->byte_pos < bs->cap) bs->buf[bs->byte_pos] = 0;
            bs->bit_pos = 8;
        }
    }
}

static void bs_write_ue(bitstream_t *bs, uint32_t val) {
    val++;
    int bits = 0;
    uint32_t tmp = val;
    while (tmp) { bits++; tmp >>= 1; }
    for (int i = 0; i < bits - 1; i++) bs_write_bits(bs, 0, 1);
    bs_write_bits(bs, val, bits);
}

static void bs_write_se(bitstream_t *bs, int32_t val) {
    uint32_t uval = val <= 0 ? (uint32_t)(-val * 2) : (uint32_t)(val * 2 - 1);
    bs_write_ue(bs, uval);
}

static void bs_byte_align(bitstream_t *bs) {
    // Write zero bits until byte-aligned (for pcm_alignment_zero_bit)
    while (bs->bit_pos != 8) {
        bs_write_bits(bs, 0, 1);
    }
}

static int bs_pos(bitstream_t *bs) {
    return bs->bit_pos == 8 ? bs->byte_pos : bs->byte_pos + 1;
}

// Big-endian writers
static void write_be32(FILE *f, uint32_t v) {
    uint8_t b[4] = {v >> 24, v >> 16, v >> 8, v};
    fwrite(b, 1, 4, f);
}

static void write_be16(FILE *f, uint16_t v) {
    uint8_t b[2] = {v >> 8, v};
    fwrite(b, 1, 2, f);
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

void nanomp4h264_open(nanomp4h264_t *enc, const nanomp4h264_config_t *config,
                      const char *filepath) {
    memset(enc, 0, sizeof(*enc));

    enc->_width = config->width;
    enc->_height = config->height;
    enc->_fps_num = config->fps_num;
    enc->_fps_den = config->fps_den;

    // Compute padded dimensions (multiple of 16)
    enc->_padded_width = (enc->_width + 15) & ~15;
    enc->_padded_height = (enc->_height + 15) & ~15;
    enc->_mb_width = enc->_padded_width / 16;
    enc->_mb_height = enc->_padded_height / 16;
    enc->_crop_right = enc->_padded_width - enc->_width;
    enc->_crop_bottom = enc->_padded_height - enc->_height;

    enc->_file = fopen(filepath, "wb");
    if (!enc->_file) {
        enc->_error = 1;
        return;
    }

    FILE *f = enc->_file;

    // Write ftyp box (28 bytes)
    write_be32(f, 28);           // size
    fwrite("ftyp", 1, 4, f);     // type
    fwrite("isom", 1, 4, f);     // major_brand
    write_be32(f, 0x200);        // minor_version
    fwrite("isomavc1mp41", 1, 12, f);  // compatible_brands

    // Record mdat position and write placeholder
    enc->_mdat_start_pos = ftell(f);
    write_be32(f, 0);            // placeholder size
    fwrite("mdat", 1, 4, f);
}

void nanomp4h264_write_frame(nanomp4h264_t *enc, const uint8_t *data,
                             nanomp4h264_format_t format) {
    if (enc->_error) return;

    (void)format;  // Only RGB888 supported currently

    FILE *f = enc->_file;
    int mb_count = enc->_mb_width * enc->_mb_height;
    uint32_t nal_len = 386 * mb_count + 3;
    uint32_t bytes_written = 0;

    write_be32(f, nal_len);

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
        uint8_t *cb_out = mb_yuv + 256;
        uint8_t *cr_out = mb_yuv + 256 + 64;

        rgb_to_yuv420_mb(data, enc->_width, enc->_height, mb_x, mb_y,
                         y_out, cb_out, cr_out);

        // Write raw samples: 256 Y, 64 Cb, 64 Cr
        fwrite(y_out, 1, 256, f);
        fwrite(cb_out, 1, 64, f);
        fwrite(cr_out, 1, 64, f);
        bytes_written += 384;
    }

    // RBSP trailing bits (stop bit + alignment)
    uint8_t trailing = 0x80;
    fwrite(&trailing, 1, 1, f);
    bytes_written += 1;

    assert(bytes_written == nal_len);

    if (enc->_frame_count == 0) {
        enc->_frame_nal_size = nal_len;
    }
    enc->_frame_count++;
}

static void write_sps(nanomp4h264_t *enc, uint8_t *buf, int *len) {
    bitstream_t bs;
    bs_init(&bs, buf, 256);

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

    *len = bs_pos(&bs);
}

static void write_pps(uint8_t *buf, int *len) {
    bitstream_t bs;
    bs_init(&bs, buf, 256);

    // NAL header
    bs_write_bits(&bs, 0x68, 8);

    // PPS
    bs_write_ue(&bs, 0);  // pps_id
    bs_write_ue(&bs, 0);  // sps_id
    bs_write_bits(&bs, 0, 1);  // entropy_coding_mode (CAVLC)
    bs_write_bits(&bs, 0, 1);  // bottom_field_pic_order
    bs_write_ue(&bs, 0);  // num_slice_groups_minus1
    bs_write_ue(&bs, 0);  // num_ref_idx_l0_default_active_minus1
    bs_write_ue(&bs, 0);  // num_ref_idx_l1_default_active_minus1
    bs_write_bits(&bs, 0, 1);  // weighted_pred_flag
    bs_write_bits(&bs, 0, 2);  // weighted_bipred_idc
    bs_write_se(&bs, 0);  // pic_init_qp_minus26
    bs_write_se(&bs, 0);  // pic_init_qs_minus26
    bs_write_se(&bs, 0);  // chroma_qp_index_offset
    bs_write_bits(&bs, 0, 1);  // deblocking_filter_control
    bs_write_bits(&bs, 0, 1);  // constrained_intra_pred
    bs_write_bits(&bs, 0, 1);  // redundant_pic_cnt_present

    // RBSP trailing
    bs_write_bits(&bs, 1, 1);
    bs_byte_align(&bs);

    *len = bs_pos(&bs);
}

void nanomp4h264_flush(nanomp4h264_t *enc) {
    if (enc->_error || enc->_frame_count == 0) return;

    FILE *f = enc->_file;
    long end_pos = ftell(f);

    // Build SPS and PPS
    uint8_t sps[256], pps[256];
    int sps_len, pps_len;
    write_sps(enc, sps, &sps_len);
    write_pps(pps, &pps_len);

    uint32_t duration = enc->_frame_count * enc->_fps_den;
    uint32_t timescale = enc->_fps_num;
    uint32_t sample_size = enc->_frame_nal_size + 4;
    uint32_t chunk_offset = (uint32_t)(enc->_mdat_start_pos + 8);

    // Calculate box sizes (bottom-up)
    uint32_t stss_size = 16 + enc->_frame_count * 4;
    uint32_t stco_size = 16 + 4;
    uint32_t stsz_size = 20 + 0;  // Using constant sample_size
    uint32_t stsc_size = 16 + 12;
    uint32_t stts_size = 16 + 8;
    uint32_t avcC_size = 8 + 8 + sps_len + 3 + pps_len;  // header + config(6) + sps_len_field(2) + sps + numPPS(1) + pps_len_field(2) + pps
    uint32_t avc1_size = 8 + 78 + avcC_size;
    uint32_t stsd_size = 8 + 8 + avc1_size;
    uint32_t stbl_size = 8 + stsd_size + stts_size + stsc_size + stsz_size + stco_size + stss_size;
    uint32_t dref_size = 8 + 4 + 4 + 12;  // header + version/flags + entry_count + url entry
    uint32_t dinf_size = 8 + dref_size;
    uint32_t vmhd_size = 20;
    uint32_t minf_size = 8 + vmhd_size + dinf_size + stbl_size;
    uint32_t hdlr_size = 8 + 24 + 13;
    uint32_t mdhd_size = 32;
    uint32_t mdia_size = 8 + mdhd_size + hdlr_size + minf_size;
    uint32_t tkhd_size = 92;
    uint32_t trak_size = 8 + tkhd_size + mdia_size;
    uint32_t mvhd_size = 108;
    uint32_t moov_size = 8 + mvhd_size + trak_size;

    // Write moov
    write_be32(f, moov_size);
    fwrite("moov", 1, 4, f);

    // mvhd
    write_be32(f, mvhd_size);
    fwrite("mvhd", 1, 4, f);
    write_be32(f, 0);  // version/flags
    write_be32(f, 0);  // creation_time
    write_be32(f, 0);  // modification_time
    write_be32(f, timescale);
    write_be32(f, duration);
    write_be32(f, 0x00010000);  // rate = 1.0
    write_be16(f, 0x0100);      // volume = 1.0
    write_be16(f, 0);           // reserved
    write_be32(f, 0);           // reserved
    write_be32(f, 0);           // reserved
    // Identity matrix
    write_be32(f, 0x00010000); write_be32(f, 0); write_be32(f, 0);
    write_be32(f, 0); write_be32(f, 0x00010000); write_be32(f, 0);
    write_be32(f, 0); write_be32(f, 0); write_be32(f, 0x40000000);
    // Pre-defined
    for (int i = 0; i < 6; i++) write_be32(f, 0);
    write_be32(f, 2);  // next_track_id

    // trak
    write_be32(f, trak_size);
    fwrite("trak", 1, 4, f);

    // tkhd
    write_be32(f, tkhd_size);
    fwrite("tkhd", 1, 4, f);
    write_be32(f, 0x00000003);  // version=0, flags=enabled+in_movie
    write_be32(f, 0);  // creation_time
    write_be32(f, 0);  // modification_time
    write_be32(f, 1);  // track_id
    write_be32(f, 0);  // reserved
    write_be32(f, duration);
    write_be32(f, 0);  // reserved
    write_be32(f, 0);  // reserved
    write_be16(f, 0);  // layer
    write_be16(f, 0);  // alternate_group
    write_be16(f, 0);  // volume
    write_be16(f, 0);  // reserved
    // Identity matrix
    write_be32(f, 0x00010000); write_be32(f, 0); write_be32(f, 0);
    write_be32(f, 0); write_be32(f, 0x00010000); write_be32(f, 0);
    write_be32(f, 0); write_be32(f, 0); write_be32(f, 0x40000000);
    write_be32(f, enc->_width << 16);   // width 16.16
    write_be32(f, enc->_height << 16);  // height 16.16

    // mdia
    write_be32(f, mdia_size);
    fwrite("mdia", 1, 4, f);

    // mdhd
    write_be32(f, mdhd_size);
    fwrite("mdhd", 1, 4, f);
    write_be32(f, 0);  // version/flags
    write_be32(f, 0);  // creation_time
    write_be32(f, 0);  // modification_time
    write_be32(f, timescale);
    write_be32(f, duration);
    write_be16(f, 0x55C4);  // language = "und"
    write_be16(f, 0);       // quality

    // hdlr
    write_be32(f, hdlr_size);
    fwrite("hdlr", 1, 4, f);
    write_be32(f, 0);  // version/flags
    write_be32(f, 0);  // pre_defined
    fwrite("vide", 1, 4, f);
    write_be32(f, 0);  // reserved
    write_be32(f, 0);  // reserved
    write_be32(f, 0);  // reserved
    fwrite("VideoHandler", 1, 13, f);

    // minf
    write_be32(f, minf_size);
    fwrite("minf", 1, 4, f);

    // vmhd
    write_be32(f, vmhd_size);
    fwrite("vmhd", 1, 4, f);
    write_be32(f, 1);  // version=0, flags=1
    write_be16(f, 0);  // graphics_mode
    write_be16(f, 0);  // opcolor[0]
    write_be16(f, 0);  // opcolor[1]
    write_be16(f, 0);  // opcolor[2]

    // dinf
    write_be32(f, dinf_size);
    fwrite("dinf", 1, 4, f);

    // dref
    write_be32(f, dref_size);
    fwrite("dref", 1, 4, f);
    write_be32(f, 0);  // version/flags
    write_be32(f, 1);  // entry_count
    write_be32(f, 12); // url size
    fwrite("url ", 1, 4, f);
    write_be32(f, 1);  // flags = self-contained

    // stbl
    write_be32(f, stbl_size);
    fwrite("stbl", 1, 4, f);

    // stsd
    write_be32(f, stsd_size);
    fwrite("stsd", 1, 4, f);
    write_be32(f, 0);  // version/flags
    write_be32(f, 1);  // entry_count

    // avc1
    write_be32(f, avc1_size);
    fwrite("avc1", 1, 4, f);
    write_be32(f, 0);  // reserved
    write_be16(f, 0);  // reserved
    write_be16(f, 1);  // data_reference_index
    write_be16(f, 0);  // pre_defined
    write_be16(f, 0);  // reserved
    write_be32(f, 0);  // pre_defined
    write_be32(f, 0);  // pre_defined
    write_be32(f, 0);  // pre_defined
    write_be16(f, enc->_width);
    write_be16(f, enc->_height);
    write_be32(f, 0x00480000);  // horiz_resolution = 72 dpi
    write_be32(f, 0x00480000);  // vert_resolution = 72 dpi
    write_be32(f, 0);           // reserved
    write_be16(f, 1);           // frame_count
    for (int i = 0; i < 32; i++) fputc(0, f);  // compressor_name
    write_be16(f, 0x0018);      // depth = 24
    write_be16(f, 0xFFFF);      // pre_defined = -1

    // avcC
    write_be32(f, avcC_size);
    fwrite("avcC", 1, 4, f);
    fputc(1, f);        // configurationVersion
    fputc(sps[1], f);   // AVCProfileIndication
    fputc(sps[2], f);   // profile_compatibility
    fputc(sps[3], f);   // AVCLevelIndication
    fputc(0xFF, f);     // lengthSizeMinusOne = 3 (4-byte lengths)
    fputc(0xE1, f);     // numOfSequenceParameterSets = 1
    write_be16(f, sps_len);
    fwrite(sps, 1, sps_len, f);
    fputc(1, f);        // numOfPictureParameterSets
    write_be16(f, pps_len);
    fwrite(pps, 1, pps_len, f);

    // stts
    write_be32(f, stts_size);
    fwrite("stts", 1, 4, f);
    write_be32(f, 0);  // version/flags
    write_be32(f, 1);  // entry_count
    write_be32(f, enc->_frame_count);
    write_be32(f, enc->_fps_den);

    // stsc
    write_be32(f, stsc_size);
    fwrite("stsc", 1, 4, f);
    write_be32(f, 0);  // version/flags
    write_be32(f, 1);  // entry_count
    write_be32(f, 1);  // first_chunk
    write_be32(f, enc->_frame_count);  // samples_per_chunk
    write_be32(f, 1);  // sample_description_index

    // stsz
    write_be32(f, stsz_size);
    fwrite("stsz", 1, 4, f);
    write_be32(f, 0);  // version/flags
    write_be32(f, sample_size);  // sample_size (constant)
    write_be32(f, enc->_frame_count);

    // stco
    write_be32(f, stco_size);
    fwrite("stco", 1, 4, f);
    write_be32(f, 0);  // version/flags
    write_be32(f, 1);  // entry_count
    write_be32(f, chunk_offset);

    // stss (sync samples - all frames are keyframes)
    write_be32(f, stss_size);
    fwrite("stss", 1, 4, f);
    write_be32(f, 0);  // version/flags
    write_be32(f, enc->_frame_count);
    for (uint32_t i = 1; i <= enc->_frame_count; i++) {
        write_be32(f, i);
    }

    // Fix mdat size
    long final_pos = ftell(f);
    uint32_t mdat_size = (uint32_t)(end_pos - enc->_mdat_start_pos);
    fseek(f, enc->_mdat_start_pos, SEEK_SET);
    write_be32(f, mdat_size);
    fseek(f, final_pos, SEEK_SET);
    fflush(f);
}

void nanomp4h264_close(nanomp4h264_t *enc) {
    if (!enc) return;
    if (enc->_frame_count > 0 && !enc->_error) {
        nanomp4h264_flush(enc);
    }
    if (enc->_file) fclose(enc->_file);
    enc->_file = NULL;
}

int nanomp4h264_get_error(const nanomp4h264_t *enc) {
    return enc ? enc->_error : 1;
}
