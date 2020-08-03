/*****************************************************************************
 * FFT3DFilter.cpp
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
 *****************************************************************************
 *
 * Plugin uses external FFTW library version 3 (http://www.fftw.org)
 * You must put libfftw3f-3.dll in the same directory as the plugin dll
 *
 * The algorithm is based on the 3D IIR/3D Frequency Domain Filter from:
 * MOTION PICTURE RESTORATION. by Anil Christopher Kokaram. Ph.D. Thesis. May 1993.
 * http://www.mee.tcd.ie/~ack/papers/a4ackphd.ps.gz
 *
 *****************************************************************************/

#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include "FFT3DFilter.h"

//-------------------------------------------------------------------------------------------
static void ApplyWiener2D( fftwf_complex *out, int outwidth, int outpitchelems, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed,
                           float beta, float sharpen, float sigmaSquaredSharpenMin, float sigmaSquaredSharpenMax, const float *wsharpen, float dehalo, const float *wdehalo, float ht2n )
{
    ApplyWiener2D_C( out, outwidth, outpitchelems, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, sharpen, sigmaSquaredSharpenMin, sigmaSquaredSharpenMax, wsharpen, dehalo, wdehalo, ht2n );
}
//-------------------------------------------------------------------------------------------
static void ApplyPattern2D( fftwf_complex *outcur, int outwidth, int outpitchelems, int bh, int howmanyblocks, float pfactor, const float *pattern2d0, float beta )
{
    ApplyPattern2D_C( outcur, outwidth, outpitchelems, bh, howmanyblocks, pfactor, pattern2d0, beta );
}
//-------------------------------------------------------------------------------------------
template < int btcur >
static void ApplyWiener3D_degrid(fftwf_complex *out, const fftwf_complex *outprev2, const fftwf_complex *outprev, const fftwf_complex *outnext, const fftwf_complex *outnext2, int outwidth, int outpitchelems, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, float degrid, const fftwf_complex *gridsample )
{
    if( btcur == 5 ) ApplyWiener3D5_degrid_C( out, outprev2, outprev, outnext, outnext2, outwidth, outpitchelems, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, degrid, gridsample );
    if( btcur == 4 ) ApplyWiener3D4_degrid_C( out, outprev2, outprev, outnext,           outwidth, outpitchelems, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, degrid, gridsample );
    if( btcur == 3 ) ApplyWiener3D3_degrid_C( out,           outprev, outnext,           outwidth, outpitchelems, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, degrid, gridsample );
    if( btcur == 2 ) ApplyWiener3D2_degrid_C( out,           outprev,                    outwidth, outpitchelems, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, degrid, gridsample );
}
//-------------------------------------------------------------------------------------------
template < int btcur >
static void ApplyPattern3D_degrid(fftwf_complex *out, const fftwf_complex *outprev2, const fftwf_complex *outprev, const fftwf_complex *outnext, const fftwf_complex *outnext2, int outwidth, int outpitchelems, int bh, int howmanyblocks, float *pattern3d, float beta, float degrid, const fftwf_complex *gridsample )
{
    if( btcur == 5 ) ApplyPattern3D5_degrid_C( out, outprev2, outprev, outnext, outnext2, outwidth, outpitchelems, bh, howmanyblocks, pattern3d, beta, degrid, gridsample );
    if( btcur == 4 ) ApplyPattern3D4_degrid_C( out, outprev2, outprev, outnext,           outwidth, outpitchelems, bh, howmanyblocks, pattern3d, beta, degrid, gridsample );
    if( btcur == 3 ) ApplyPattern3D3_degrid_C( out,           outprev, outnext,           outwidth, outpitchelems, bh, howmanyblocks, pattern3d, beta, degrid, gridsample );
    if( btcur == 2 ) ApplyPattern3D2_degrid_C( out,           outprev,                    outwidth, outpitchelems, bh, howmanyblocks, pattern3d, beta, degrid, gridsample );
}
//-------------------------------------------------------------------------------------------
template < int btcur >
static void ApplyWiener3D(fftwf_complex *out, const fftwf_complex *outprev2, const fftwf_complex *outprev, const fftwf_complex *outnext, const fftwf_complex *outnext2, int outwidth, int outpitchelems, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta )
{
    if( btcur == 5 ) ApplyWiener3D5_C( out, outprev2, outprev, outnext, outnext2, outwidth, outpitchelems, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta );
    if( btcur == 4 ) ApplyWiener3D4_C( out, outprev2, outprev, outnext,           outwidth, outpitchelems, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta );
    if( btcur == 3 ) ApplyWiener3D3_C( out,           outprev, outnext,           outwidth, outpitchelems, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta );
    if( btcur == 2 ) ApplyWiener3D2_C( out,           outprev,                    outwidth, outpitchelems, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta );
}
//-------------------------------------------------------------------------------------------
template < int btcur >
static void ApplyPattern3D(fftwf_complex *out, const fftwf_complex *outprev2, const fftwf_complex *outprev, const fftwf_complex *outnext, const fftwf_complex *outnext2, int outwidth, int outpitchelems, int bh, int howmanyblocks, const float *pattern3d, float beta )
{
    if( btcur == 5 ) ApplyPattern3D5_C( out, outprev2, outprev, outnext, outnext2, outwidth, outpitchelems, bh, howmanyblocks, pattern3d, beta );
    if( btcur == 4 ) ApplyPattern3D4_C( out, outprev2, outprev, outnext,           outwidth, outpitchelems, bh, howmanyblocks, pattern3d, beta );
    if( btcur == 3 ) ApplyPattern3D3_C( out,           outprev, outnext,           outwidth, outpitchelems, bh, howmanyblocks, pattern3d, beta );
    if( btcur == 2 ) ApplyPattern3D2_C( out,           outprev,                    outwidth, outpitchelems, bh, howmanyblocks, pattern3d, beta );
}
//-------------------------------------------------------------------------------------------
static void ApplyKalmanPattern( const fftwf_complex *outcur, fftwf_complex *outLast, fftwf_complex *covar, fftwf_complex *covarProcess, int outwidth, int outpitchelems, int bh, int howmanyblocks, const float *covarNoiseNormed, float kratio2 )
{
    ApplyKalmanPattern_C( outcur, outLast, covar, covarProcess, outwidth, outpitchelems, bh, howmanyblocks,  covarNoiseNormed, kratio2 );
}
//-------------------------------------------------------------------------------------------
static void ApplyKalman( const fftwf_complex *outcur, fftwf_complex *outLast, fftwf_complex *covar, fftwf_complex *covarProcess, int outwidth, int outpitchelems, int bh, int howmanyblocks,  float covarNoiseNormed, float kratio2 )
{
    ApplyKalman_C( outcur, outLast, covar, covarProcess, outwidth, outpitchelems, bh, howmanyblocks,  covarNoiseNormed, kratio2 );
}
//-------------------------------------------------------------------------------------------
static void Sharpen( fftwf_complex *outcur, int outwidth, int outpitchelems, int bh, int howmanyblocks, float sharpen, float sigmaSquaredSharpenMin, float sigmaSquaredSharpenMax, const float *wsharpen, float dehalo, const float *wdehalo, float ht2n )
{
    Sharpen_C( outcur, outwidth, outpitchelems, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMin, sigmaSquaredSharpenMax, wsharpen, dehalo, wdehalo, ht2n );
}
//-------------------------------------------------------------------------------------------
static void Sharpen_degrid( fftwf_complex *outcur, int outwidth, int outpitchelems, int bh, int howmanyblocks, float sharpen, float sigmaSquaredSharpenMin, float sigmaSquaredSharpenMax, const float *wsharpen, float degrid, const fftwf_complex *gridsample, float dehalo, const float *wdehalo, float ht2n )
{
    Sharpen_degrid_C( outcur, outwidth, outpitchelems, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMin, sigmaSquaredSharpenMax, wsharpen, degrid, gridsample, dehalo, wdehalo, ht2n );
}
//-------------------------------------------------------------------------------------------

//-------------------------------------------------------------------
static void fill_complex( fftwf_complex *plane, int outsize, float realvalue, float imgvalue)
{
    /* it is not fast, but called only in constructor */
    for( int w = 0; w < outsize; w++ )
    {
        plane[w][0] = realvalue;
        plane[w][1] = imgvalue;
    }
}
//-------------------------------------------------------------------
static void SigmasToPattern( float sigma, float sigma2, float sigma3, float sigma4, int bh, int outwidth, int outpitchelems, float norm, float *pattern2d )
{
    /* it is not fast, but called only in constructor */
    float sigmacur;
    const float ft2 = sqrt( 0.5f ) / 2; /* frequency for sigma2 */
    const float ft3 = sqrt( 0.5f ) / 4; /* frequency for sigma3 */
    for( int h = 0; h < bh; h++ )
    {
        for( int w = 0; w < outwidth; w++ )
        {
            float fy = (bh - 2.0f * abs( h - bh / 2)) / bh; /* normalized to 1 */
            float fx = (w * 1.0f) / outwidth;               /* normalized to 1 */
            float f = sqrt( (fx * fx + fy * fy) * 0.5f );   /* normalized to 1 */
            if( f < ft3 )
            {   /* low frequencies */
                sigmacur = sigma4 + (sigma3 - sigma4) * f / ft3;
            }
            else if( f < ft2 )
            {   /* middle frequencies */
                sigmacur = sigma3 + (sigma2 - sigma3) * (f - ft3) / (ft2 - ft3);
            }
            else
            {   /* high frequencies */
                sigmacur = sigma + (sigma2 - sigma) * (1 - f) / (1 - ft2);
            }
            pattern2d[w] = sigmacur * sigmacur / norm;
        }
        pattern2d += outpitchelems;
    }
}

static void Pattern2Dto3D(const float *pattern2d, int bh, int outwidth, int outpitchelems, float mult, float *pattern3d) {
    /* slow, but executed once only per clip */
    int size = bh * outpitchelems;
    for (int i = 0; i < size; i++) { /* get 3D pattern */
        pattern3d[i] = pattern2d[i] * mult;
    }
}

void VS_CC FFT3DFilter::Init(VSMap *in, VSMap *out, void **instance_data, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    FFT3DFilter *data = reinterpret_cast<FFT3DFilter *>(*instance_data);
    vsapi->setVideoInfo(vsapi->getVideoInfo(data->node), 1, node);
}

const VSFrameRef *VS_CC FFT3DFilter::GetFrame(int n, int activation_reason, void **instance_data, void **frame_data, VSFrameContext *frame_ctx, VSCore *core, const VSAPI *vsapi) {
    FFT3DFilter *data = reinterpret_cast<FFT3DFilter *>(*instance_data);
    if (activation_reason == arInitial) {
        int btcur = data->bt; /* bt used for current frame */
        if ((data->bt / 2 > n) || (data->bt - 1) / 2 > (data->vi->numFrames - 1 - n))
            btcur = 1; /* do 2D filter for first and last frames */

        if (btcur <= 1 ) {
            vsapi->requestFrameFilter(n, data->node, frame_ctx);
        } else {
            int fromframe = n - data->bt / 2;
            for (int i = 0; i < data->bt; i++) {
                vsapi->requestFrameFilter(fromframe + i, data->node, frame_ctx);
            }
        }
    } else if (activation_reason == arAllFramesReady) {
        return data->ApplyFilter(n, frame_ctx, core, vsapi);
    }

    return nullptr;
}

const VSFrameRef *VS_CC FFT3DFilter::GetPShowFrame(int n, int activation_reason, void **instance_data, void **frame_data, VSFrameContext *frame_ctx, VSCore *core, const VSAPI *vsapi) {
    FFT3DFilter *data = reinterpret_cast<FFT3DFilter *>(*instance_data);
    if (activation_reason == arInitial) {
        vsapi->requestFrameFilter(n, data->node, frame_ctx);
    } else if (activation_reason == arAllFramesReady) {
        return data->ApplyPShow(n, frame_ctx, core, vsapi);
    }

    return nullptr;
}

void VS_CC FFT3DFilter::Free(void *instance_data, VSCore *core, const VSAPI *vsapi) {
    FFT3DFilter *data = reinterpret_cast<FFT3DFilter *>(instance_data);
    vsapi->freeNode(data->node);
    vsapi->freeNode(data->pshownode);
    vsapi->freeFrame(data->gridsample);
    delete data;
}

//-------------------------------------------------------------------
FFT3DFilter::FFT3DFilter
(
    FFT3DFilterTransform *transform, const VSVideoInfo *_vi,
    float _sigma, float _beta, int _plane, int _bw, int _bh, int _bt, int _ow, int _oh,
    float _kratio, float _sharpen, float _scutoff, float _svr, float _smin, float _smax,
    bool _measure, bool _interlaced, int _wintype,
    int _pframe, int _px, int _py, bool _pshow, float _pcutoff, float _pfactor,
    float _sigma2, float _sigma3, float _sigma4, float _degrid,
    float _dehalo, float _hr, float _ht, int _ncpu,
    VSNodeRef *_node, VSNodeRef *_pshownode, VSCore *core, const VSAPI *vsapi
) : sigma(_sigma), beta(_beta), plane(_plane), bw(_bw), bh(_bh), bt(_bt), ow(_ow), oh(_oh),
kratio(_kratio), sharpen(_sharpen), scutoff(_scutoff), svr(_svr), smin(_smin), smax(_smax),
measure(_measure), interlaced(_interlaced), wintype(_wintype),
pframe(_pframe), px(_px), py(_py), pshow(_pshow), pcutoff(_pcutoff), pfactor(_pfactor),
sigma2(_sigma2), sigma3(_sigma3), sigma4(_sigma4), degrid(_degrid),
dehalo(_dehalo), hr(_hr), ht(_ht), ncpu(_ncpu), in(nullptr, nullptr),
wsharpen(nullptr, nullptr), wdehalo(nullptr, nullptr),
outLast(nullptr, nullptr), covar(nullptr, nullptr),
covarProcess(nullptr, nullptr), pattern2d(nullptr, nullptr),
pattern3d(nullptr, nullptr), vi(_vi), node(_node), pshownode(_pshownode) {

    if (ow < 0) ow = bw / 3; /* changed from bw/4 to bw/3 in v.1.2 */
    if (oh < 0) oh = bh / 3; /* changed from bh/4 to bh/3 in v.1.2 */

    maxval = (1 << vi->format->bitsPerSample) - 1;

    if (wintype == 9000)
        planeBase = (vi->format->sampleType == stInteger) ? (1 << (vi->format->bitsPerSample - 1)) : 0;
    else
        planeBase = (plane && vi->format->sampleType == stInteger) ? (1 << (vi->format->bitsPerSample - 1)) : 0;

    nox = ((vi->width >> (plane ? vi->format->subSamplingW : 0)) - ow + (bw - ow - 1)) / (bw - ow);
    noy = ((vi->height >> (plane ? vi->format->subSamplingH : 0)) - oh + (bh - oh - 1)) / (bh - oh);

    /* padding by 1 block per side */
    nox += 2;
    noy += 2;
    mirw = bw - ow; /* set mirror size as block interval */
    mirh = bh - oh;

    int insize = bw * bh * nox * noy;
    in = std::unique_ptr<float[], decltype(&fftw_free)>(fftwf_alloc_real(insize), fftwf_free);
    outwidth = bw / 2 + 1;                  /* width (pitch) of complex fft block */
    outpitchelems = ((outwidth + 1) / 2) * 2;    /* must be even for SSE - v1.7 */
    outpitch = outpitchelems * vi->format->bytesPerSample;

    outsize = outpitchelems * bh * nox * noy;   /* replace outwidth to outpitchelems here and below in v1.7 */

    if (bt == 0) /* Kalman */
    { 
        outLast = std::unique_ptr<fftwf_complex[], decltype(&fftw_free)>(fftwf_alloc_complex(outsize), fftwf_free);
        covar = std::unique_ptr<fftwf_complex[], decltype(&fftw_free)>(fftwf_alloc_complex(outsize), fftwf_free);
        covarProcess = std::unique_ptr<fftwf_complex[], decltype(&fftw_free)>(fftwf_alloc_complex(outsize), fftwf_free);
    }

    howmanyblocks = nox * noy;

    fftwf_plan_with_nthreads( ncpu );
    fftwf_plan_with_nthreads( 1 );

    wsharpen = std::unique_ptr<float[], decltype(&fftw_free)>(fftwf_alloc_real(bh * outpitchelems), fftwf_free);
    wdehalo  = std::unique_ptr<float[], decltype(&fftw_free)>(fftwf_alloc_real(bh * outpitchelems), fftwf_free);

    GetSharpenWindow(bw, bh, outwidth, outpitchelems, svr, scutoff, wsharpen.get());
    GetDeHaloWindow(bw, bh, outwidth, outpitchelems, hr, svr, wdehalo.get());

    norm = 1.0f / (bw * bh); /* do not forget set FFT normalization factor */

    sigmaSquaredNoiseNormed2D = sigma * sigma / norm;
    sigmaNoiseNormed2D = sigma / sqrtf( norm );
    sigmaMotionNormed  = sigma * kratio / sqrtf( norm );
    sigmaSquaredSharpenMinNormed = smin * smin / norm;
    sigmaSquaredSharpenMaxNormed = smax * smax / norm;
    ht2n = ht * ht / norm; /* halo threshold squared and normed - v1.9 */

    /* init Kalman */
    if( bt == 0 ) /* Kalman */
    {
        fill_complex( outLast.get(),      outsize, 0, 0 );
        fill_complex( covar.get(),        outsize, sigmaSquaredNoiseNormed2D, sigmaSquaredNoiseNormed2D );
        fill_complex( covarProcess.get(), outsize, sigmaSquaredNoiseNormed2D, sigmaSquaredNoiseNormed2D );
    }

    pattern2d = std::unique_ptr<float[], decltype(&fftw_free)>(fftwf_alloc_real(bh * outpitchelems), fftwf_free); /* noise pattern window array */
    pattern3d = std::unique_ptr<float[], decltype(&fftw_free)>(fftwf_alloc_real(bh * outpitchelems), fftwf_free); /* noise pattern window array */

    bool isPatternSet = false;
    if( (sigma2 != sigma || sigma3 != sigma || sigma4 != sigma) && pfactor == 0 )
    {   /* we have different sigmas, so create pattern from sigmas */
        SigmasToPattern( sigma, sigma2, sigma3, sigma4, bh, outwidth, outpitchelems, norm, pattern2d.get());
        isPatternSet = true;
        pfactor = 1;
    }

    gridsample = transform->GetGridSample(core, vsapi);
    if (pfactor != 0 && isPatternSet == false && pshow == false) /* get noise pattern */ {
        const VSFrameRef *psrc = vsapi->getFrame(pframe, node, nullptr, 0);
        assert(psrc);
        // FIXME, is psigma used? if not remove from class
        transform->GetNoisePattern(psrc, px, py, pattern2d.get(), psigma, reinterpret_cast<const fftwf_complex *>(vsapi->getReadPtr(gridsample, 0)), core, vsapi);
    }

    if (bt > 1)
        Pattern2Dto3D(pattern2d.get(), bh, outwidth, outpitchelems, (float)bt, pattern3d.get());
}
//-----------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
static void PutPatternOnly( fftwf_complex *outcur, int outwidth, int outpitchelems, int bh, int nox, int noy, int px, int py )
{
    int h,w;
    int block;
    int pblock = py * nox + px;
    int blocks = nox * noy;

    for( block = 0; block < pblock; block++ )
    {
        for( h = 0; h < bh; h++ )
        {
            for( w = 0; w < outwidth; w++ )
            {
                outcur[w][0] = 0;
                outcur[w][1] = 0;
            }
            outcur += outpitchelems;
        }
    }

    outcur += bh * outpitchelems;

    for( block = pblock + 1; block < blocks; block++ )
    {
        for( h = 0; h < bh; h++ )
        {
            for( w = 0; w < outwidth; w++ )
            {
                outcur[w][0] = 0;
                outcur[w][1] = 0;
            }
            outcur += outpitchelems;
        }
    }
}
//-------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------
template < int btcur >
void FFT3DFilter::Wiener3D
(
    int               n,
    VSNodeRef *node,
    VSFrameRef *dst,
    VSFrameContext   *frame_ctx,
    const VSAPI      *vsapi
)
{
    int fromframe = n - btcur / 2;
    int outcenter = btcur / 2;
    const fftwf_complex *frames[btcur] = {};
    const VSFrameRef *frefs[btcur] = {};

    for (int i = 0; i < btcur; i++) {
        frefs[i] = vsapi->getFrameFilter(fromframe + i, node, frame_ctx);
        frames[i] = reinterpret_cast<const fftwf_complex *>(vsapi->getReadPtr(frefs[i], 0));
    }

    // unwrap the frames to outp again, because this step was in the original code and rewriting things nicer is effort
    // also clamp the index unlike the original so no reads happen beyond the array bounds...
    const fftwf_complex *outp[5] = { frames[std::max(0, outcenter - 2)], frames[std::max(0, outcenter - 1)], nullptr, frames[std::min(outcenter + 1, btcur - 1)], frames[std::min(outcenter + 2, btcur - 1)] };

    if( degrid != 0 )
    {
        if( pfactor != 0 )
            ApplyPattern3D_degrid< btcur >(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outp[0], outp[1], outp[3], outp[4], outwidth, outpitchelems, bh, howmanyblocks, pattern3d.get(), beta, degrid, reinterpret_cast<const fftwf_complex *>(vsapi->getReadPtr(gridsample, 0)));
        else
            ApplyWiener3D_degrid< btcur >(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outp[0], outp[1], outp[3], outp[4], outwidth, outpitchelems, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, degrid, reinterpret_cast<const fftwf_complex *>(vsapi->getReadPtr(gridsample, 0)));
        Sharpen_degrid(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outwidth, outpitchelems, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen.get(), degrid, reinterpret_cast<const fftwf_complex *>(vsapi->getReadPtr(gridsample, 0)), dehalo, wdehalo.get(), ht2n );
    }
    else
    {
        if( pfactor != 0 )
            ApplyPattern3D< btcur >(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outp[0], outp[1], outp[3], outp[4], outwidth, outpitchelems, bh, howmanyblocks, pattern3d.get(), beta );
        else
            ApplyWiener3D< btcur >(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outp[0], outp[1], outp[3], outp[4], outwidth, outpitchelems, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta );
        Sharpen(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outwidth, outpitchelems, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen.get(), dehalo, wdehalo.get(), ht2n );
    }

    for (int i = 0; i < btcur; i++)
        vsapi->freeFrame(frefs[i]);
}

VSFrameRef *FFT3DFilter::ApplyPShow(int n, VSFrameContext *frame_ctx, VSCore *core, const VSAPI *vsapi) {
    const VSFrameRef *src = vsapi->getFrameFilter(n, node, frame_ctx);
    const VSFrameRef *pshowsrc = vsapi->getFrameFilter(n, pshownode, frame_ctx);
    // fixme, pass on pxf and pxy and psigma, this is probably wrong and should be from the previous filter
    int pxf = px;
    int pyf = py;
    vsapi->freeFrame(pshowsrc);

    VSFrameRef *dst = vsapi->copyFrame(src, core);
    vsapi->freeFrame(src);
    PutPatternOnly(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outwidth, outpitchelems, bh, nox, noy, pxf, pyf);

    return dst;
}

VSFrameRef *FFT3DFilter::ApplyFilter
(
    int               n,
    VSFrameContext   *frame_ctx,
    VSCore           *core,
    const VSAPI      *vsapi
)
{
#if 0
    // FIXME, probably needs to be split into a separate filter or done on startup
    if( pfactor != 0 && pshow )
    {  
        // 1. Forward transform (FFT3DFilterTransform), same as src clip
        // 2. Findpattern, setpattern and such
        // X. --no inverse transform, only pattern block (pxf, pyf), psigma and pattern2d carries over?-- if px and py is set then they're not calculated
        // 3. Forward transform (FFT3DFilterTransform), but with analysis window set to all 1 (wan*) and adjusted planeBase
        // 4. PutPatternOnly, adjusted planeBase has an effect?
        // 5. Inverse transform FFT3DFilterInvTransform, but with synthesis windows set to all 1 (wsyn*) and adjusted planeBase

        // dependencies =>
        // lump 1 and 2 together as a simple FFT3DFilterTransform helper function?
        // very linear, no shortcuts possible apart the clean source also having to be provided to step 3
        // changed values in 3-5 are all constant
        // setting wintype == 2 yields correct wan* constants, but not syn, simply add a wintype=9000 mode that secretly sets all values including planebase correctly?
        // planebase communicated how?


        // Group 1-X? How to carry over all esoteric values?
        
        /* show noise pattern window */
        /* put source bytes to float array of overlapped blocks */
        FramePlaneToCoverbuf( plane, src, reinterpret_cast<T *>(coverbuf.get()), coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, vsapi );
        InitOverlapPlane( in.get(), reinterpret_cast<T *>(coverbuf.get()), coverpitch, planeBase );
        /* make FFT 2D */
        fftwf_execute_dft_r2c( plan.get(), in.get(), outrez.get());

        // transformed src here

        int pxf, pyf;
        if( px == 0 && py == 0 ) /* try find pattern block with minimal noise sigma */
            FindPatternBlock( outrez.get(), outwidth, outpitchelems, bh, nox, noy, pxf, pyf, pwin.get(), degrid, gridsample.get() );
        else
        {
            pxf = px; /* fixed bug in v1.6 */
            pyf = py;
        }

        // FIXME, is pattern2d ever used? NOPE? But pass it on anyway since the destination space needs to be allocated
        SetPattern( outrez.get(), outwidth, outpitchelems, bh, nox, noy, pxf, pyf, pwin.get(), pattern2d.get(), psigma, degrid, gridsample.get());

        // actually affects the next transform and inverse transform call
        // how to pipeline this shitshow?
        /* change analysis and synthesis window to constant to show */
        for( int i = 0; i < ow; i++ )
        {
            wanxl[i] = 1;    wanxr[i] = 1;    wsynxl[i] = 1;    wsynxr[i] = 1;
        }
        for( int i = 0; i < oh; i++ )
        {
            wanyl[i] = 1;    wanyr[i] = 1;    wsynyl[i] = 1;    wsynyr[i] = 1;
        }

        //FIXME, why is planebase assigned here? originally always assigned 128
        // probably to get positie and negative part of the pattern
        planeBase = (vi.format->sampleType == stInteger) ? (1 << (vi.format->bitsPerSample - 1)) : 0;

        /* put source bytes to float array of overlapped blocks */
        /* cur frame */
        FramePlaneToCoverbuf( plane, src, reinterpret_cast<T *>(coverbuf.get()), coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, vsapi );
        InitOverlapPlane( in.get(), reinterpret_cast<T *>(coverbuf.get()), coverpitch, planeBase );
        /* make FFT 2D */
        fftwf_execute_dft_r2c( plan.get(), in.get(), outrez.get());

        PutPatternOnly( outrez.get(), outwidth, outpitchelems, bh, nox, noy, pxf, pyf );
        /* do inverse 2D FFT, get filtered 'in' array */
        fftwf_execute_dft_c2r( planinv.get(), outrez.get(), in.get());

        /* make destination frame plane from current overlaped blocks */
        DecodeOverlapPlane( in.get(), norm, reinterpret_cast<T *>(coverbuf.get()), coverpitch, planeBase, maxval );
        CoverbufToFramePlane( plane, reinterpret_cast<T *>(coverbuf.get()), coverwidth, coverheight, coverpitch, dst, mirw, mirh, interlaced, vsapi );
        vsapi->propSetData(vsapi->getFramePropsRW(dst), "FFT3DFilterPShowSigma", std::to_string(psigma).c_str(), -1, paAppend);
        return;
    }
#endif

    const VSFrameRef *src = vsapi->getFrameFilter(n, node, frame_ctx);
    VSFrameRef *dst = vsapi->copyFrame(src, core);
    vsapi->freeFrame(src);

    int btcur = bt; /* bt used for current frame */

    if( (bt / 2 > n) || (bt - 1) / 2 > (vi->numFrames - 1 - n) )
    {
        btcur = 1; /* do 2D filter for first and last frames */
    }


    if( btcur > 0 ) /* Wiener */
    {
        sigmaSquaredNoiseNormed = btcur * sigma * sigma / norm; /* normalized variation=sigma^2 */

        /* get power spectral density (abs quadrat) for every block and apply filter */

        if( btcur == 1 ) /* 2D */
        {
            if( degrid != 0 )
            {
                if( pfactor != 0 )
                {
                    ApplyPattern2D_degrid_C(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outwidth, outpitchelems, bh, howmanyblocks, pfactor, pattern2d.get(), beta, degrid, reinterpret_cast<const fftwf_complex *>(vsapi->getReadPtr(gridsample, 0)));
                    Sharpen_degrid(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outwidth, outpitchelems, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen.get(), degrid, reinterpret_cast<const fftwf_complex *>(vsapi->getReadPtr(gridsample, 0)), dehalo, wdehalo.get(), ht2n );
                }
                else
                    ApplyWiener2D_degrid_C(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outwidth, outpitchelems, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen.get(), degrid, reinterpret_cast<const fftwf_complex *>(vsapi->getReadPtr(gridsample, 0)), dehalo, wdehalo.get(), ht2n );
            }
            else
            {
                if( pfactor != 0 )
                {
                    ApplyPattern2D(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outwidth, outpitchelems, bh, howmanyblocks, pfactor, pattern2d.get(), beta );
                    Sharpen(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outwidth, outpitchelems, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen.get(), dehalo, wdehalo.get(), ht2n );
                }
                else
                    ApplyWiener2D(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outwidth, outpitchelems, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen.get(), dehalo, wdehalo.get(), ht2n );
            }
        }
        else if( btcur == 2 ) /* 3D2 */
        {
            Wiener3D< 2 >( n, node, dst, frame_ctx, vsapi );
        }
        else if( btcur == 3 ) /* 3D3 */
        {
            Wiener3D< 3 >( n, node, dst, frame_ctx, vsapi );
        }
        else if( btcur == 4 ) /* 3D4 */
        {
            Wiener3D< 4 >( n, node, dst, frame_ctx, vsapi );
        }
        else if( btcur == 5 ) /* 3D5 */
        {
            Wiener3D< 5 >( n, node, dst, frame_ctx, vsapi );
        }
    }
    else if( bt == 0 ) /* Kalman filter */
    {
        /* get power spectral density (abs quadrat) for every block and apply filter */
        if( n == 0 )
            return dst;

        if( pfactor != 0 )
            ApplyKalmanPattern(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outLast.get(), covar.get(), covarProcess.get(), outwidth, outpitchelems, bh, howmanyblocks, pattern2d.get(), kratio * kratio );
        else
            ApplyKalman(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outLast.get(), covar.get(), covarProcess.get(), outwidth, outpitchelems, bh, howmanyblocks, sigmaSquaredNoiseNormed2D, kratio * kratio );

        /* copy outLast to outrez */
        memcpy(outLast.get(),
            reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)),
                outsize * sizeof(fftwf_complex));
        if( degrid != 0 )
            Sharpen_degrid(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outwidth, outpitchelems, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen.get(), degrid, reinterpret_cast<const fftwf_complex *>(vsapi->getReadPtr(gridsample, 0)), dehalo, wdehalo.get(), ht2n );
        else
            Sharpen(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outwidth, outpitchelems, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen.get(), dehalo, wdehalo.get(), ht2n );
    }
    else if( bt == -1 ) /* sharpen only */
    {
        if( degrid != 0 )
            Sharpen_degrid(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outwidth, outpitchelems, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen.get(), degrid, reinterpret_cast<const fftwf_complex *>(vsapi->getReadPtr(gridsample, 0)), dehalo, wdehalo.get(), ht2n );
        else
            Sharpen(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outwidth, outpitchelems, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen.get(), dehalo, wdehalo.get(), ht2n );
    }

    return dst;
}

