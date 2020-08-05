/*****************************************************************************
 * FFT3DFilter.h
 *****************************************************************************
 * FFT3DFilter plugin for VapourSynth - 3D Frequency Domain filter
 *
 * Copyright (C) 2004-2006 A.G.Balakhnin aka Fizick <bag@hotmail.ru> http://avisynth.org.ru
 * Copyright (C) 2015      Yusuke Nakamura, <muken.the.vfrmaniac@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as published by
 * the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *****************************************************************************/

#include <string>
#include <memory>
#include <vector>
#include <cassert>
#include <fftw3.h>

#include <VapourSynth4.h>

void GetAnalysisWindow(int wintype, int ow, int oh, float *wanxl, float *wanxr, float *wanyl, float *wanyr);
void GetSynthesisWindow(int wintype, int ow, int oh, float *wsynxl, float *wsynxr, float *wsynyl, float *wsynyr);
void GetSharpenWindow(int bw, int bh, int outwidth, int outpitchelems, float svr, float scutoff, float *wsharpen);
void GetDeHaloWindow(int bw, int bh, int outwidth, int outpitchelems, float hr, float svr, float *wdehalo);
void GetPatternWindow(int bw, int bh, int outwidth, int outpitchelems, float pcutoff, float *pwin);

/** declarations of filtering functions: **/
/* C */
void ApplyWiener2D_C(fftwf_complex *out, int outwidth, int outpitchelems, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, float sharpen, float sigmaSquaredSharpenMin, float sigmaSquaredSharpenMax, const float *wsharpen, float dehalo, const float *wdehalo, float ht2n);
void ApplyPattern2D_C(fftwf_complex *outcur, int outwidth, int outpitchelems, int bh, int howmanyblocks, float pfactor, const float *pattern2d0, float beta);
void ApplyWiener3D2_C(fftwf_complex *outcur, const fftwf_complex *outprev, int outwidth, int outpitchelems, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta);
void ApplyPattern3D2_C(fftwf_complex *outcur, const fftwf_complex *outprev, int outwidth, int outpitchelems, int bh, int howmanyblocks, const float *pattern3d, float beta);
void ApplyWiener3D3_C(fftwf_complex *out, const fftwf_complex *outprev, const fftwf_complex *outnext, int outwidth, int outpitchelems, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta);
void ApplyPattern3D3_C(fftwf_complex *out, const fftwf_complex *outprev, const fftwf_complex *outnext, int outwidth, int outpitchelems, int bh, int howmanyblocks, const float *pattern3d, float beta);
void ApplyWiener3D4_C(fftwf_complex *out, const fftwf_complex *outprev2, const fftwf_complex *outprev, const fftwf_complex *outnext, int outwidth, int outpitchelems, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta);
void ApplyPattern3D4_C(fftwf_complex *out, const fftwf_complex *outprev2, const fftwf_complex *outprev, const fftwf_complex *outnext, int outwidth, int outpitchelems, int bh, int howmanyblocks, const float *pattern3d, float beta);
void ApplyWiener3D5_C(fftwf_complex *out, const fftwf_complex *outprev2, const fftwf_complex *outprev, const fftwf_complex *outnext, const fftwf_complex *outnext2, int outwidth, int outpitchelems, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta);
void ApplyPattern3D5_C(fftwf_complex *out, const fftwf_complex *outprev2, const fftwf_complex *outprev, const fftwf_complex *outnext, const fftwf_complex *outnext2, int outwidth, int outpitchelems, int bh, int howmanyblocks, const float *pattern3d, float beta);
void ApplyKalmanPattern_C(const fftwf_complex *outcur, fftwf_complex *outLast, fftwf_complex *covar, fftwf_complex *covarProcess, int outwidth, int outpitchelems, int bh, int howmanyblocks, const float *covarNoiseNormed, float kratio2);
void ApplyKalman_C(const fftwf_complex *outcur, fftwf_complex *outLast, fftwf_complex *covar, fftwf_complex *covarProcess, int outwidth, int outpitchelems, int bh, int howmanyblocks, float covarNoiseNormed, float kratio2);
void Sharpen_C(fftwf_complex *outcur, int outwidth, int outpitchelems, int bh, int howmanyblocks, float sharpen, float sigmaSquaredSharpenMin, float sigmaSquaredSharpenMax, const float *wsharpen, float dehalo, const float *wdehalo, float ht2n);
/* degrid_C */
void ApplyWiener2D_degrid_C(fftwf_complex *out, int outwidth, int outpitchelems, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, float sharpen, float sigmaSquaredSharpenMin, float sigmaSquaredSharpenMax, const float *wsharpen, float degrid, const fftwf_complex *gridsample, float dehalo, const float *wdehalo, float ht2n);
void ApplyWiener3D2_degrid_C(fftwf_complex *outcur, const fftwf_complex *outprev, int outwidth, int outpitchelems, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, float degrid, const fftwf_complex *gridsample);
void ApplyWiener3D3_degrid_C(fftwf_complex *outcur, const fftwf_complex *outprev, const fftwf_complex *outnext, int outwidth, int outpitchelems, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, float degrid, const fftwf_complex *gridsample);
void ApplyWiener3D4_degrid_C(fftwf_complex *outcur, const fftwf_complex *outprev2, const fftwf_complex *outprev, const fftwf_complex *outnext, int outwidth, int outpitchelems, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, float degrid, const fftwf_complex *gridsample);
void ApplyWiener3D5_degrid_C(fftwf_complex *outcur, const fftwf_complex *outprev2, const fftwf_complex *outprev, const fftwf_complex *outnext, const fftwf_complex *outnext2, int outwidth, int outpitchelems, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, float degrid, const fftwf_complex *gridsample);
void Sharpen_degrid_C(fftwf_complex *outcur, int outwidth, int outpitchelems, int bh, int howmanyblocks, float sharpen, float sigmaSquaredSharpenMin, float sigmaSquaredSharpenMax, const float *wsharpen, float degrid, const fftwf_complex *gridsample, float dehalo, const float *wdehalo, float ht2n);
void ApplyPattern2D_degrid_C(fftwf_complex *outcur, int outwidth, int outpitchelems, int bh, int howmanyblocks, float pfactor, const float *pattern2d0, float beta, float degrid, const fftwf_complex *gridsample);
void ApplyPattern3D2_degrid_C(fftwf_complex *outcur, const fftwf_complex *outprev, int outwidth, int outpitchelems, int bh, int howmanyblocks, const float *pattern3d, float beta, float degrid, const fftwf_complex *gridsample);
void ApplyPattern3D3_degrid_C(fftwf_complex *out, const fftwf_complex *outprev, const fftwf_complex *outnext, int outwidth, int outpitchelems, int bh, int howmanyblocks, const float *pattern3d, float beta, float degrid, const fftwf_complex *gridsample);
void ApplyPattern3D4_degrid_C(fftwf_complex *out, const fftwf_complex *outprev2, const fftwf_complex *outprev, const fftwf_complex *outnext, int outwidth, int outpitchelems, int bh, int howmanyblocks, const float *pattern3d, float beta, float degrid, const fftwf_complex *gridsample);
void ApplyPattern3D5_degrid_C(fftwf_complex *out, const fftwf_complex *outprev2, const fftwf_complex *outprev, const fftwf_complex *outnext, const fftwf_complex *outnext2, int outwidth, int outpitchelems, int bh, int howmanyblocks, const float *pattern3d, float beta, float degrid, const fftwf_complex *gridsample);

template<typename T>
static void fft3d_memset(T *dst, T val, size_t count) {
    for (size_t i = 0; i < count; i++)
        dst[i] = val;
}

class FFT3DFilterTransform;

class FFT3DFilter
{
private:
    /* parameters */
    float sigma;    /* noise level (std deviation) for high frequncies */
    float beta;     /* relative noise margin for Wiener filter */
    int   plane;    /* color plane */
    int   bw;       /* block width */
    int   bh;       /* block height */
    int   bt;       /* block size  along time (mumber of frames), =0 for Kalman, >0 for Wiener */
    int   ow;       /* overlap width - v.0.9 */
    int   oh;       /* overlap height - v.0.9 */
    float kratio;   /* threshold to sigma ratio for Kalman filter */
    float sharpen;  /* sharpen factor (0 to 1 and above) */
    float scutoff;  /* sharpen cufoff frequency (relative to max) - v1.7 */
    float svr;      /* sharpen vertical ratio (0 to 1 and above) - v.1.0 */
    float smin;     /* minimum limit for sharpen (prevent noise amplifying) - v.1.1 */
    float smax;     /* maximum limit for sharpen (prevent oversharping) - v.1.1 */
    int   pframe;   /* noise pattern frame number */
    int   px;       /* noise pattern window x-position */
    int   py;       /* noise pattern window y-position */
    float pfactor;  /* noise pattern denoise strength */
    float sigma2;   /* noise level for middle frequencies */
    float sigma3;   /* noise level for low frequencies */
    float sigma4;   /* noise level for lowest (zero) frequencies */
    float degrid;   /* decrease grid */
    float dehalo;   /* remove halo strength - v.1.9 */
    float hr;       /* halo radius - v1.9 */
    float ht;       /* halo threshold - v1.9 */

    /* additional parameterss */

    const VSFrameRef *gridsample;
    int nox, noy;
    int outwidth;
    int outpitch;
    int outpitchelems; /* v.1.7 */

    int outsize;
    int howmanyblocks;

    std::unique_ptr<float[], decltype(&fftw_free)> wsharpen;
    std::unique_ptr<float[], decltype(&fftw_free)> wdehalo;

    // Shared buffers only used for bt=0 (kalman) mode
    std::unique_ptr<fftwf_complex[], decltype(&fftw_free)> outLast;
    std::unique_ptr<fftwf_complex[], decltype(&fftw_free)> covar;
    std::unique_ptr<fftwf_complex[], decltype(&fftw_free)> covarProcess;
    //
    float sigmaSquaredNoiseNormed;
    float sigmaSquaredNoiseNormed2D;
    float sigmaSquaredSharpenMinNormed;
    float sigmaSquaredSharpenMaxNormed;
    float ht2n; /* halo threshold squared normed */
    float norm; /* normalization factor */

    std::unique_ptr<float[], decltype(&fftw_free)> pattern2d;
    std::unique_ptr<float[], decltype(&fftw_free)> pattern3d;

    const VSVideoInfo *vi;
    VSNodeRef *node;

    template < int btcur >
    void Wiener3D( int n, VSNodeRef *node, VSFrameRef *dst, VSFrameContext *frame_ctx, const VSAPI *vsapi );

public:
    VSFrameRef *ApplyFilter(int n, VSFrameContext *frame_ctx, VSCore *core, const VSAPI *vsapi);

    /* Constructor */
    FFT3DFilter
    (
        FFT3DFilterTransform *transform, const VSVideoInfo *vi,
        float _sigma, float _beta, int _plane, int _bw, int _bh, int _bt, int _ow, int _oh,
        float _kratio, float _sharpen, float _scutoff, float _svr, float _smin, float _smax,
        int _pframe, int _px, int _py, bool _pshow, float _pcutoff, float _pfactor,
        float _sigma2, float _sigma3, float _sigma4, float _degrid,
        float _dehalo, float _hr, float _ht, int _ncpu,
        VSNodeRef *node, VSCore *core, const VSAPI *vsapi
    );

    static const VSFrameRef *VS_CC GetFrame(int n, int activation_reason, void *instance_data, void **frame_data, VSFrameContext *frame_ctx, VSCore *core, const VSAPI *vsapi);
    static void VS_CC Free(void *instance_data, VSCore *core, const VSAPI *vsapi);
};

class FFT3DFilterTransform {
private:
    /* parameters */
    int plane;
    int bw;       /* block width */
    int bh;       /* block height */
    int ow;       /* overlap width - v.0.9 */
    int oh;       /* overlap height - v.0.9 */
    int px;
    int py;
    float pcutoff;
    float degrid;
    bool interlaced;
    int   ncpu;     /* number of threads - v2.0 */

    // set by constructor
    VSNodeRef *node;

    std::unique_ptr<uint8_t[]> coverbuf; /*  block buffer covering the frame without remainders (with sufficient width and heigth) */
    int coverwidth;
    int coverheight;
    int coverpitch;

    int mirw; /* mirror width for padding */
    int mirh; /* mirror height for padding */

    VSVideoInfo dstvi;

    int planeBase;

    int nox, noy;
    int outwidth;
    int outpitchelems; /* v.1.7 */

    std::unique_ptr<float[]> wanxl; /* analysis */
    std::unique_ptr<float[]> wanxr;
    std::unique_ptr<float[]> wanyl;
    std::unique_ptr<float[]> wanyr;

    std::unique_ptr<float[], decltype(&fftw_free)> in;
    std::unique_ptr<fftwf_plan_s, decltype(&fftwf_destroy_plan)> plan;

    template<typename T>
    void InitOverlapPlane(float *__restrict inp0, const T *__restrict srcp0, int src_pitch, int planeBase);
public:
    VSVideoInfo outvi; // fixme, hie behind a fucntion or something

    FFT3DFilterTransform(bool pshow, VSNodeRef *node, int plane, int wintype, int bw, int bh, int ow, int oh, int px, int py, float pcutoff, float degrid, bool interlaced, bool measure, int ncpu, VSCore *core, const VSAPI *vsapi);
    const VSFrameRef *GetGridSample(VSCore *core, const VSAPI *vsapi);
    VSFrameRef *GetFrame(const VSFrameRef *src, VSCore *core, const VSAPI *vsapi);
    void GetNoisePattern(int n, int &px, int &py, float *pattern2d, float &psigma, const fftwf_complex *gridsample, VSCore *core, const VSAPI *vsapi);
    VSFrameRef *GetPShowInfo(const VSFrameRef *src, VSCore *core, const VSAPI *vsapi);

    static const VSFrameRef *VS_CC GetFrame(int n, int activation_reason, void *instance_data, void **frame_data, VSFrameContext *frame_ctx, VSCore *core, const VSAPI *vsapi);
    static const VSFrameRef *VS_CC GetPShowFrame(int n, int activation_reason, void *instance_data, void **frame_data, VSFrameContext *frame_ctx, VSCore *core, const VSAPI *vsapi);
    static void VS_CC Free(void *instance_data, VSCore *core, const VSAPI *vsapi);
};

class FFT3DFilterInvTransform {
private:
    /* parameters */
    int bw;       /* block width */
    int bh;       /* block height */
    int ow;       /* overlap width - v.0.9 */
    int oh;       /* overlap height - v.0.9 */
    bool interlaced;
    int   ncpu;     /* number of threads - v2.0 */

    // set by constructor
    VSNodeRef *node;

    std::unique_ptr<uint8_t[]> coverbuf; /*  block buffer covering the frame without remainders (with sufficient width and heigth) */
    int coverwidth;
    int coverheight;
    int coverpitch;

    int mirw; /* mirror width for padding */
    int mirh; /* mirror height for padding */

    int planeBase; // fixme, added as a constant before transform and remove it again after inverse transform, pointless?

    int nox, noy;
    int outwidth;
    int outpitchelems; /* v.1.7 */

    float norm; /* normalization factor */

    std::unique_ptr<float[]> wsynxl;
    std::unique_ptr<float[]> wsynxr;
    std::unique_ptr<float[]> wsynyl;
    std::unique_ptr<float[]> wsynyr;

    std::unique_ptr<float[], decltype(&fftw_free)> in;
    std::unique_ptr<fftwf_plan_s, decltype(&fftwf_destroy_plan)> planinv;

    template<typename T>
    void DecodeOverlapPlane(const float *__restrict inp0, float norm, T *__restrict dstp0, int dst_pitch, int planeBase, int maxval);

public:
    VSVideoInfo dstvi; // fixme, hide

    FFT3DFilterInvTransform(VSNodeRef *node, const VSVideoInfo *vi, int plane, int wintype, int bw, int bh, int ow, int oh, bool interlaced, bool measure, int ncpu, VSCore *core, const VSAPI *vsapi);

    static const VSFrameRef *VS_CC GetFrame(int n, int activation_reason, void *instance_data, void **frame_data, VSFrameContext *frame_ctx, VSCore *core, const VSAPI *vsapi);
    static void VS_CC Free(void *instance_data, VSCore *core, const VSAPI *vsapi);
};

class FFT3DFilterPShow {
private:
    /* parameters */
    int plane;
    int bw;       /* block width */
    int bh;       /* block height */
    int ow;       /* overlap width - v.0.9 */
    int oh;       /* overlap height - v.0.9 */

    // set by constructor
    VSNodeRef *node;

    const VSVideoInfo *vi;

public:
    FFT3DFilterPShow(VSNodeRef *node, int plane, int bw, int bh, int ow, int oh, bool interlaced, VSCore *core, const VSAPI *vsapi);
    VSFrameRef *GetFrame(const VSFrameRef *src, VSCore *core, const VSAPI *vsapi);

    static const VSFrameRef *VS_CC GetFrame(int n, int activation_reason, void *instance_data, void **frame_data, VSFrameContext *frame_ctx, VSCore *core, const VSAPI *vsapi);
    static void VS_CC Free(void *instance_data, VSCore *core, const VSAPI *vsapi);
};