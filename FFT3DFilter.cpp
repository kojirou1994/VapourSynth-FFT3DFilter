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
static void ApplyWiener3D_degrid(fftwf_complex *out, const fftwf_complex *outprev2, const fftwf_complex *outprev, const fftwf_complex *outnext, const fftwf_complex *outnext2, int outwidth, int outpitchelems, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, float degrid, fftwf_complex *gridsample )
{
    if( btcur == 5 ) ApplyWiener3D5_degrid_C( out, outprev2, outprev, outnext, outnext2, outwidth, outpitchelems, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, degrid, gridsample );
    if( btcur == 4 ) ApplyWiener3D4_degrid_C( out, outprev2, outprev, outnext,           outwidth, outpitchelems, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, degrid, gridsample );
    if( btcur == 3 ) ApplyWiener3D3_degrid_C( out,           outprev, outnext,           outwidth, outpitchelems, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, degrid, gridsample );
    if( btcur == 2 ) ApplyWiener3D2_degrid_C( out,           outprev,                    outwidth, outpitchelems, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, degrid, gridsample );
}
//-------------------------------------------------------------------------------------------
template < int btcur >
static void ApplyPattern3D_degrid(fftwf_complex *out, const fftwf_complex *outprev2, const fftwf_complex *outprev, const fftwf_complex *outnext, const fftwf_complex *outnext2, int outwidth, int outpitchelems, int bh, int howmanyblocks, float *pattern3d, float beta, float degrid, fftwf_complex *gridsample )
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

//-------------------------------------------------------------------
FFT3DFilter::FFT3DFilter
(
    float _sigma, float _beta, int _plane, int _bw, int _bh, int _bt, int _ow, int _oh,
    float _kratio, float _sharpen, float _scutoff, float _svr, float _smin, float _smax,
    bool _measure, bool _interlaced, int _wintype,
    int _pframe, int _px, int _py, bool _pshow, float _pcutoff, float _pfactor,
    float _sigma2, float _sigma3, float _sigma4, float _degrid,
    float _dehalo, float _hr, float _ht, int _ncpu,
    VSVideoInfo _vi, VSNodeRef *_node, VSCore *core, const VSAPI *vsapi
) : sigma(_sigma), beta(_beta), plane(_plane), bw(_bw), bh(_bh), bt(_bt), ow(_ow), oh(_oh),
kratio(_kratio), sharpen(_sharpen), scutoff(_scutoff), svr(_svr), smin(_smin), smax(_smax),
measure(_measure), interlaced(_interlaced), wintype(_wintype),
pframe(_pframe), px(_px), py(_py), pshow(_pshow), pcutoff(_pcutoff), pfactor(_pfactor),
sigma2(_sigma2), sigma3(_sigma3), sigma4(_sigma4), degrid(_degrid),
dehalo(_dehalo), hr(_hr), ht(_ht), ncpu(_ncpu), in(nullptr, nullptr),
plan(nullptr, nullptr),
planinv(nullptr, nullptr),
wsharpen(nullptr, nullptr), wdehalo(nullptr, nullptr),
outLast(nullptr, nullptr), covar(nullptr, nullptr),
covarProcess(nullptr, nullptr), pattern2d(nullptr, nullptr),
pattern3d(nullptr, nullptr), vi(_vi), node(_node) {
    int istat = fftwf_init_threads();
    if (istat == 0)
        throw std::runtime_error{ "fftwf_init_threads() failed!" };

    fftwf_make_planner_thread_safe();

    int i, j;

    if (ow < 0) ow = bw / 3; /* changed from bw/4 to bw/3 in v.1.2 */
    if (oh < 0) oh = bh / 3; /* changed from bh/4 to bh/3 in v.1.2 */

    maxval = (1 << vi.format->bitsPerSample) - 1;

    planeBase = (plane && vi.format->sampleType == stInteger) ? (1 << (vi.format->bitsPerSample - 1)) : 0;

    nox = ((vi.width >> (plane ? vi.format->subSamplingW : 0)) - ow + (bw - ow - 1)) / (bw - ow);
    noy = ((vi.height >> (plane ? vi.format->subSamplingH : 0)) - oh + (bh - oh - 1)) / (bh - oh);

    /* padding by 1 block per side */
    nox += 2;
    noy += 2;
    mirw = bw - ow; /* set mirror size as block interval */
    mirh = bh - oh;

    coverwidth = nox * (bw - ow) + ow;
    coverheight = noy * (bh - oh) + oh;
    coverpitch = ((coverwidth + 7) / 8) * 8 * vi.format->bytesPerSample;
    coverbuf = std::unique_ptr<uint8_t[]>(new uint8_t[coverheight * coverpitch]);

    int insize = bw * bh * nox * noy;
    in = std::unique_ptr<float[], decltype(&fftw_free)>(fftwf_alloc_real(insize), fftwf_free);
    outwidth = bw / 2 + 1;                  /* width (pitch) of complex fft block */
    outpitchelems = ((outwidth + 1) / 2) * 2;    /* must be even for SSE - v1.7 */
    outpitch = outpitchelems * vi.format->bytesPerSample;

    outsize = outpitchelems * bh * nox * noy;   /* replace outwidth to outpitchelems here and below in v1.7 */

    if (bt == 0) /* Kalman */
    { 
        outLast = std::unique_ptr<fftwf_complex[], decltype(&fftw_free)>(fftwf_alloc_complex(outsize), fftwf_free);
        covar = std::unique_ptr<fftwf_complex[], decltype(&fftw_free)>(fftwf_alloc_complex(outsize), fftwf_free);
        covarProcess = std::unique_ptr<fftwf_complex[], decltype(&fftw_free)>(fftwf_alloc_complex(outsize), fftwf_free);
    }
    // FIXME, temp space that can be reallocated on demand, is it needed at all?
    std::unique_ptr<fftwf_complex[], decltype(&fftw_free)> outrez = std::unique_ptr<fftwf_complex[], decltype(&fftw_free)>(fftwf_alloc_complex(outsize), fftwf_free);

    int planFlags = measure ? FFTW_MEASURE: FFTW_ESTIMATE;
    int rank = 2; /* 2d */
    ndim[0] = bh; /* size of block along height */
    ndim[1] = bw; /* size of block along width */
    int istride = 1;
    int ostride = 1;
    int idist   = bw * bh;
    int odist   = outpitchelems * bh;/*  v1.7 (was outwidth) */
    inembed[0] = bh;
    inembed[1] = bw;
    onembed[0] = bh;
    onembed[1] = outpitchelems;      /*  v1.7 (was outwidth) */
    howmanyblocks = nox * noy;

    fftwf_plan_with_nthreads( ncpu );

    plan = std::unique_ptr<fftwf_plan_s, decltype(&fftwf_destroy_plan)>(fftwf_plan_many_dft_r2c(rank, ndim, howmanyblocks,
        in.get(), inembed, istride, idist, outrez.get(), onembed, ostride, odist, planFlags), fftwf_destroy_plan);
    if( !plan )
        throw std::runtime_error{ "fftwf_plan_many_dft_r2c" };

    planinv = std::unique_ptr<fftwf_plan_s, decltype(&fftwf_destroy_plan)>(fftwf_plan_many_dft_c2r( rank, ndim, howmanyblocks,
                                       outrez.get(), onembed, ostride, odist, in.get(), inembed, istride, idist, planFlags), fftwf_destroy_plan);
    if( !planinv )
        throw std::runtime_error{ "fftwf_plan_many_dft_c2r" };

    fftwf_plan_with_nthreads( 1 );

    wanxl = std::unique_ptr<float[]>(new float[ow]);
    wanxr = std::unique_ptr<float[]>(new float[ow]);
    wanyl = std::unique_ptr<float[]>(new float[oh]);
    wanyr = std::unique_ptr<float[]>(new float[oh]);

    wsynxl = std::unique_ptr<float[]>(new float[ow]);
    wsynxr = std::unique_ptr<float[]>(new float[ow]);
    wsynyl = std::unique_ptr<float[]>(new float[oh]);
    wsynyr = std::unique_ptr<float[]>(new float[oh]);

    wsharpen = std::unique_ptr<float[], decltype(&fftw_free)>(fftwf_alloc_real(bh * outpitchelems), fftwf_free);
    wdehalo  = std::unique_ptr<float[], decltype(&fftw_free)>(fftwf_alloc_real(bh * outpitchelems), fftwf_free);

    GetAnalysisWindow(wintype, ow, oh, wanxl.get(), wanxr.get(), wanyl.get(), wanyr.get());
    GetSynthesisWindow(wintype, ow, oh, wsynxl.get(), wsynxr.get(), wsynyl.get(), wsynyr.get());
    GetSharpenWindow(bw, bh, outwidth, outpitchelems, svr, scutoff, wsharpen.get());
    GetDeHaloWindow(bw, bh, outwidth, outpitchelems, hr, svr, wdehalo.get());

    btcurlast = -999; /* init as nonexistant */

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

    pwin = std::unique_ptr<float[]>(new float[bh * outpitchelems]); /* pattern window array */

    for( j = 0; j < bh; j++ )
    {
        float fh2;
        if( j < bh / 2 )
            fh2 = (j * 2.0f / bh) * (j * 2.0f / bh);
        else
            fh2 = ((bh - 1 - j) * 2.0f / bh) * ((bh - 1 - j) * 2.0f / bh);
        for( i = 0; i < outwidth; i++ )
        {
            float fw2 = (i * 2.0f / bw) * (j * 2.0f / bw);
            pwin[i + j * outpitchelems] = (fh2 + fw2) / (fh2 + fw2 + pcutoff * pcutoff);
        }
    }

    pattern2d = std::unique_ptr<float[], decltype(&fftw_free)>(fftwf_alloc_real(bh * outpitchelems), fftwf_free); /* noise pattern window array */
    pattern3d = std::unique_ptr<float[], decltype(&fftw_free)>(fftwf_alloc_real(bh * outpitchelems), fftwf_free); /* noise pattern window array */

    if( (sigma2 != sigma || sigma3 != sigma || sigma4 != sigma) && pfactor == 0 )
    {   /* we have different sigmas, so create pattern from sigmas */
        SigmasToPattern( sigma, sigma2, sigma3, sigma4, bh, outwidth, outpitchelems, norm, pattern2d.get());
        isPatternSet = true;
        pfactor = 1;
    }
    else
    {
        isPatternSet = false; /* pattern must be estimated */
    }

    // FIXME, this section is equivalent to calling the transform filter with a blank clip of max pixel value,
    // should probably have the upstream transform class passed or the transformed plane since they're the same but this is fine for now

    FFT3DFilterTransformPlane *gridSampleSrc = new FFT3DFilterTransformPlane(node, plane, wintype, bw, bh, ow, oh, interlaced, measure, core, vsapi);
    gridsample = gridSampleSrc->GetGridSample(core, vsapi);
    delete gridSampleSrc;
}
//-----------------------------------------------------------------------

//-------------------------------------------------------------------------------------------
static void FindPatternBlock( const fftwf_complex *outcur0, int outwidth, int outpitchelems, int bh, int nox, int noy, int &px, int &py, const float *pwin, float degrid, const fftwf_complex *gridsample )
{
    /* since v1.7 outwidth must be really an outpitchelems */
    int h;
    int w;
    const fftwf_complex *outcur;
    float psd;
    float sigmaSquaredcur;
    float sigmaSquared = 1e15f;

    for( int by = 2; by < noy - 2; by++ )
    {
        for( int bx = 2; bx < nox - 2; bx++ )
        {
            outcur = outcur0 + nox * by * bh * outpitchelems + bx * bh * outpitchelems;
            sigmaSquaredcur = 0;
            float gcur = degrid * outcur[0][0] / gridsample[0][0]; /* grid (windowing) correction factor */
            for( h = 0; h < bh; h++ )
            {
                for( w = 0; w < outwidth; w++ )
                {
                    float grid0 = gcur * gridsample[w][0];
                    float grid1 = gcur * gridsample[w][1];
                    float corrected0 = outcur[w][0] - grid0;
                    float corrected1 = outcur[w][1] - grid1;
                    psd = corrected0 * corrected0 + corrected1 * corrected1;
                    sigmaSquaredcur += psd * pwin[w]; /* windowing */
                }
                outcur     += outpitchelems;
                pwin       += outpitchelems;
                gridsample += outpitchelems;
            }
            pwin -= outpitchelems * bh; /* restore */
            if( sigmaSquaredcur < sigmaSquared )
            {
                px = bx;
                py = by;
                sigmaSquared = sigmaSquaredcur;
            }
        }
    }
}
//-------------------------------------------------------------------------------------------
static void SetPattern( const fftwf_complex *outcur, int outwidth, int outpitchelems, int bh, int nox, int noy, int px, int py, const float *pwin, float *pattern2d, float &psigma, float degrid, const fftwf_complex *gridsample )
{
    int h;
    int w;
    outcur += nox * py * bh * outpitchelems + px * bh * outpitchelems;
    float psd;
    float sigmaSquared = 0;
    float weight = 0;

    for( h = 0; h < bh; h++ )
    {
        for( w = 0; w < outwidth; w++ )
        {
            weight += pwin[w];
        }
        pwin += outpitchelems;
    }
    pwin -= outpitchelems * bh; /* restore */

    float gcur = degrid * outcur[0][0] / gridsample[0][0]; /* grid (windowing) correction factor */

    for( h = 0; h < bh; h++ )
    {
        for( w = 0; w < outwidth; w++ )
        {
            float grid0 = gcur * gridsample[w][0];
            float grid1 = gcur * gridsample[w][1];
            float corrected0 = outcur[w][0] - grid0;
            float corrected1 = outcur[w][1] - grid1;
            psd = corrected0 * corrected0 + corrected1 * corrected1;
            pattern2d[w] = psd * pwin[w]; /* windowing */
            sigmaSquared += pattern2d[w]; /* sum */
        }
        outcur     += outpitchelems;
        pattern2d  += outpitchelems;
        pwin       += outpitchelems;
        gridsample += outpitchelems;
    }
    psigma = sqrt( sigmaSquared / (weight * bh * outwidth) ); /* mean std deviation (sigma) */
}
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
static void Pattern2Dto3D( const float *pattern2d, int bh, int outwidth, int outpitchelems, float mult, float *pattern3d )
{
    /* slow, but executed once only per clip */
    int size = bh * outpitchelems;
    for( int i = 0; i < size; i++ )
    { /* get 3D pattern */
        pattern3d[i] = pattern2d[i] * mult;
    }
}
//-------------------------------------------------------------------------------------------
template < int btcur >
void FFT3DFilter::Wiener3D
(
    int               n,
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
        frefs[i] = vsapi->getFrameFilter(fromframe + i);
        frames[i] = reinterpret_cast<const fftwf_complex *>(vsapi->getReadPtr(frefs[i], 0));
    }

    // dst is used as scratch space, before outp[0] was used but now it's outp[2], inject src frame somewhere to compensate

    /*
    for (int i = 0; i <= toframe - fromframe; i++) {
        // FIXME, equivalent to i == 0?
        if (i + fromframe == n - btcur / 2) {
            // oldest frame will be used as scratch space so use a copy of it instead
            // originally a complicated scheme that assumed perfectly linear access was used to actually consume the cached frame but that's too complicated,
            // and possibly not that great when out of order access can happen due to multithreading
            memcpy(outrez.get(),
                frames[i],
                outsize * sizeof(fftwf_complex));
        }
    }*/

    // unwrap the frames to outp again, because this step was in the original code and rewriting things nicer is effort
    // also clamp the index unlike the original so no reads happen beyond the array bounds...
    const fftwf_complex *outp[5] = { frames[std::max(0, outcenter - 2)], frames[std::max(0, outcenter - 1)], nullptr, frames[std::min(outcenter + 1, btcur - 1)], frames[std::min(outcenter + 2, btcur - 1)] };
    //outp[2 - btcur / 2] = outrez.get();

    if( degrid != 0 )
    {
        if( pfactor != 0 )
            ApplyPattern3D_degrid< btcur >(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outp[0], outp[1], outp[3], outp[4], outwidth, outpitchelems, bh, howmanyblocks, pattern3d.get(), beta, degrid, gridsample.get());
        else
            ApplyWiener3D_degrid< btcur >(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outp[0], outp[1], outp[3], outp[4], outwidth, outpitchelems, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, degrid, gridsample.get());
        Sharpen_degrid(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outwidth, outpitchelems, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen.get(), degrid, gridsample.get(), dehalo, wdehalo.get(), ht2n );
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

template<typename T>
void FFT3DFilter::ApplyFilter
(
    int               n,
    VSFrameRef       *dst,
    VSFrameContext   *frame_ctx,
    VSCore           *core,
    const VSAPI      *vsapi
)
{
#if 1
    // FIXME, probably needs to be split into a separate filter or done on startup
    if( pfactor != 0 && isPatternSet == false && pshow == false ) /* get noise pattern */
    {
        const VSFrameRef *psrc = vsapi->getFrameFilter( pframe, node, frame_ctx ); /* get noise pattern frame */

        /* put source bytes to float array of overlapped blocks */
        FramePlaneToCoverbuf( plane, psrc, reinterpret_cast<T *>(coverbuf.get()), coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, vsapi );
        vsapi->freeFrame( psrc );
        InitOverlapPlane( in.get(), reinterpret_cast<T *>(coverbuf.get()), coverpitch, planeBase );
        /* make FFT 2D */
        fftwf_execute_dft_r2c( plan.get(), in.get(), outrez.get());
        if( px == 0 && py == 0 ) /* try find pattern block with minimal noise sigma */
            FindPatternBlock( outrez.get(), outwidth, outpitchelems, bh, nox, noy, px, py, pwin.get(), degrid, gridsample.get());
        SetPattern( outrez.get(), outwidth, outpitchelems, bh, nox, noy, px, py, pwin.get(), pattern2d.get(), psigma, degrid, gridsample.get());
        isPatternSet = true;
    }
    else if( pfactor != 0 && pshow == true )
    {   /* show noise pattern window */
        /* put source bytes to float array of overlapped blocks */
        FramePlaneToCoverbuf( plane, src, reinterpret_cast<T *>(coverbuf.get()), coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, vsapi );
        InitOverlapPlane( in.get(), reinterpret_cast<T *>(coverbuf.get()), coverpitch, planeBase );
        /* make FFT 2D */
        fftwf_execute_dft_r2c( plan.get(), in.get(), outrez.get());

        int pxf, pyf;
        if( px == 0 && py == 0 ) /* try find pattern block with minimal noise sigma */
            FindPatternBlock( outrez.get(), outwidth, outpitchelems, bh, nox, noy, pxf, pyf, pwin.get(), degrid, gridsample.get() );
        else
        {
            pxf = px; /* fixed bug in v1.6 */
            pyf = py;
        }
        SetPattern( outrez.get(), outwidth, outpitchelems, bh, nox, noy, pxf, pyf, pwin.get(), pattern2d.get(), psigma, degrid, gridsample.get());

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

    int btcur = bt; /* bt used for current frame */

    if( (bt / 2 > n) || (bt - 1) / 2 > (vi.numFrames - 1 - n) )
    {
        btcur = 1; /* do 2D filter for first and last frames */
    }

    const VSFrameRef *src = vsapi->getFrameFilter(n, node, frame_ctx);

    if( btcur > 0 ) /* Wiener */
    {
        sigmaSquaredNoiseNormed = btcur * sigma * sigma / norm; /* normalized variation=sigma^2 */

        if( btcur != btcurlast )
            Pattern2Dto3D( pattern2d.get(), bh, outwidth, outpitchelems, (float)btcur, pattern3d.get());

        /* get power spectral density (abs quadrat) for every block and apply filter */

        /* put source bytes to float array of overlapped blocks */

        if( btcur == 1 ) /* 2D */
        {
            // FIXME, copy src=>dst for this to work
            if( degrid != 0 )
            {
                if( pfactor != 0 )
                {
                    ApplyPattern2D_degrid_C(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outwidth, outpitchelems, bh, howmanyblocks, pfactor, pattern2d.get(), beta, degrid, gridsample.get());
                    Sharpen_degrid(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outwidth, outpitchelems, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen.get(), degrid, gridsample.get(), dehalo, wdehalo.get(), ht2n );
                }
                else
                    ApplyWiener2D_degrid_C(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outwidth, outpitchelems, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen.get(), degrid, gridsample.get(), dehalo, wdehalo.get(), ht2n );
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
            Wiener3D< T, 2 >( n, dst, frame_ctx, vsapi );
        }
        else if( btcur == 3 ) /* 3D3 */
        {
            Wiener3D< T, 3 >( n, dst, frame_ctx, vsapi );
        }
        else if( btcur == 4 ) /* 3D4 */
        {
            Wiener3D< T, 4 >( n, dst, frame_ctx, vsapi );
        }
        else if( btcur == 5 ) /* 3D5 */
        {
            Wiener3D< T, 5 >( n, dst, frame_ctx, vsapi );
        }
        /* make destination frame plane from current overlaped blocks */
        DecodeOverlapPlane( in.get(), norm, reinterpret_cast<T *>(coverbuf.get()), coverpitch, planeBase, maxval);
        CoverbufToFramePlane( plane, reinterpret_cast<T *>(coverbuf.get()), coverwidth, coverheight, coverpitch, dst, mirw, mirh, interlaced, vsapi );
    }
    else if( bt == 0 ) /* Kalman filter */
    {
        /* get power spectral density (abs quadrat) for every block and apply filter */

        if( n == 0 )
            return;

        /* put source bytes to float array of overlapped blocks */
        /* cur frame */
        //FramePlaneToCoverbuf( plane, src, reinterpret_cast<T *>(coverbuf.get()), coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, vsapi );
        //InitOverlapPlane( in.get(), reinterpret_cast<T *>(coverbuf.get()), coverpitch, planeBase );
        /* make FFT 2D */
        //fftwf_execute_dft_r2c( plan.get(), in.get(), outrez.get());

        if( pfactor != 0 )
            ApplyKalmanPattern(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outLast.get(), covar.get(), covarProcess.get(), outwidth, outpitchelems, bh, howmanyblocks, pattern2d.get(), kratio * kratio );
        else
            ApplyKalman(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outLast.get(), covar.get(), covarProcess.get(), outwidth, outpitchelems, bh, howmanyblocks, sigmaSquaredNoiseNormed2D, kratio * kratio );

        /* copy outLast to outrez */
        memcpy(outLast.get(),
            reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)),
                outsize * sizeof(fftwf_complex));
        if( degrid != 0 )
            Sharpen_degrid(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outwidth, outpitchelems, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen.get(), degrid, gridsample.get(), dehalo, wdehalo.get(), ht2n );
        else
            Sharpen(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outwidth, outpitchelems, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen.get(), dehalo, wdehalo.get(), ht2n );
        /* do inverse FFT 2D, get filtered 'in' array
         * note: input "out" array is destroyed by execute algo.
         * that is why we must have its copy in "outLast" array */
        //fftwf_execute_dft_c2r( planinv.get(), outrez.get(), in.get());
        /* make destination frame plane from current overlaped blocks */
        //DecodeOverlapPlane( in.get(), norm, reinterpret_cast<T *>(coverbuf.get()), coverpitch, planeBase, maxval);
        //CoverbufToFramePlane( plane, reinterpret_cast<T *>(coverbuf.get()), coverwidth, coverheight, coverpitch, dst, mirw, mirh, interlaced, vsapi );
    }
    else if( bt == -1 ) /* sharpen only */
    {
        if( degrid != 0 )
            Sharpen_degrid(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outwidth, outpitchelems, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen.get(), degrid, gridsample.get(), dehalo, wdehalo.get(), ht2n );
        else
            Sharpen(reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)), outwidth, outpitchelems, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen.get(), dehalo, wdehalo.get(), ht2n );
    }

    btcurlast = btcur;

    /* As we now are finished processing the image. */
}

//-------------------------------------------------------------------
FFT3DFilterMulti::FFT3DFilterMulti
(
    float _sigma, float _beta, bool _process[3], int _bw, int _bh, int _bt, int _ow, int _oh,
    float _kratio, float _sharpen, float _scutoff, float _svr, float _smin, float _smax,
    bool _measure, bool _interlaced, int _wintype,
    int _pframe, int _px, int _py, bool _pshow, float _pcutoff, float _pfactor,
    float _sigma2, float _sigma3, float _sigma4, float _degrid,
    float _dehalo, float _hr, float _ht, int _ncpu,
    const VSMap *in, const VSAPI *vsapi
) : Clips(),
    bt( _bt ), pframe( _pframe ), pshow( _pshow ), pfactor( _pfactor )
{
    node =  vsapi->propGetNode( in, "clip", 0, 0 );
    vi   = *vsapi->getVideoInfo( node );

    try {
        if ((vi.format->bitsPerSample > 16 && vi.format->sampleType == stInteger) || (vi.format->bitsPerSample != 32 && vi.format->sampleType == stFloat))
            throw std::runtime_error{ "only 8-16 bit integer and 32 bit float are supported" };

        for (int i = 0; i < vi.format->numPlanes; i++) {
            if (_process[i])
                Clips[i] = new FFT3DFilter(_sigma, _beta, i, _bw, _bh, _bt, _ow, _oh,
                    _kratio, _sharpen, _scutoff, _svr, _smin, _smax,
                    _measure, _interlaced, _wintype,
                    _pframe, _px, _py, _pshow, _pcutoff, _pfactor,
                    _sigma2, _sigma3, _sigma4, _degrid, _dehalo, _hr, _ht, _ncpu,
                    vi, node, core, vsapi);
        }

        for (int i = 2; i >= 0; i--) {
            if (Clips[i]) {
                isPatternSet = Clips[i]->getIsPatternSet();
                break;
            }
        }
    } catch (std::runtime_error &) {
        Free(vsapi);
        throw;
    }
}

void FFT3DFilterMulti::Free(const VSAPI *vsapi) {
    for (int i = 0; i < 3; i++)
        delete Clips[i];
    vsapi->freeNode(node);
    delete this;
}

void FFT3DFilterMulti::RequestFrame
(
    int             n,
    VSFrameContext *frame_ctx,
    VSCore         *core,
    const VSAPI    *vsapi
)
{
    if( pfactor != 0 && isPatternSet == false && pshow == false )
        vsapi->requestFrameFilter( pframe, node, frame_ctx );

    int btcur = bt; /* bt used for current frame */

    if( (bt / 2 > n) || (bt - 1) / 2 > (vi.numFrames - 1 - n) )
    {
        btcur = 1; /* do 2D filter for first and last frames */
    }

    if( btcur > 0 )
    {
        for( int i = n - btcur / 2; i <= n + (btcur - 1) / 2; ++i )
            vsapi->requestFrameFilter( i, node, frame_ctx );
    }
    else
        vsapi->requestFrameFilter( n, node, frame_ctx );
}

VSFrameRef *FFT3DFilterMulti::GetFrame
(
    int             n,
    VSFrameContext *frame_ctx,
    VSCore         *core,
    const VSAPI    *vsapi
)
{
    /* Request frame 'n' from the source clip. */
    const VSFrameRef *src = vsapi->getFrameFilter( n, node, frame_ctx );

    VSFrameRef *dst = nullptr;
    if( pfactor != 0 && pshow == true )
        dst = vsapi->copyFrame( src, core );
    else if( bt == 0 && n == 0 )
        /* Kalman filter does nothing for the first frame. */
        dst = vsapi->copyFrame(src, core);
    else
    {
        int planes[3] = { 0, 1, 2 };
        const VSFrameRef *srcf[3] = { Clips[0] ? nullptr : src, Clips[1] ? nullptr : src, Clips[2] ? nullptr : src };
        dst = vsapi->newVideoFrame2(vsapi->getFrameFormat(src), vsapi->getFrameWidth(src, 0), vsapi->getFrameHeight(src, 0), srcf, planes, src, core);
    }

    // fixme, copyframe?
    
    for (int i = 0; i < 3; i++) {
        if (Clips[i]) {
            if (vi.format->bytesPerSample == 1)
                Clips[i]->ApplyFilter<uint8_t>(n, dst, frame_ctx, core, vsapi);
            else if (vi.format->bytesPerSample == 2)
                Clips[i]->ApplyFilter<uint16_t>(n, dst, frame_ctx, core, vsapi);
            else if (vi.format->bytesPerSample == 4)
                Clips[i]->ApplyFilter<float>(n, dst, frame_ctx, core, vsapi);
        }
    }

    for (int i = 2; i >= 0; i--) {
        if (Clips[i]) {
            isPatternSet = Clips[i]->getIsPatternSet();
            break;
        }
    }

    vsapi->freeFrame( src );
    return dst;
}
