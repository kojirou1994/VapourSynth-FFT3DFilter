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
#include <fftw3.h>

#include "VapourSynth.h"

class FFT3DFilter
{
public:
    struct FFTCacheRec {
        fftwf_complex *fft = nullptr;
        int what = -1;
        void swap(FFTCacheRec &v) noexcept {
            std::swap(fft, v.fft);
            std::swap(what, v.what);
        }
        ~FFTCacheRec() {
            fftwf_free(fft);
        }
    };
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
    bool  measure;  /* fft optimal method */
    bool  interlaced;
    int   wintype;  /* window type */
    int   pframe;   /* noise pattern frame number */
    int   px;       /* noise pattern window x-position */
    int   py;       /* noise pattern window y-position */
    bool  pshow;    /* show noise pattern */
    float pcutoff;  /* pattern cutoff frequency (relative to max) */
    float pfactor;  /* noise pattern denoise strength */
    float sigma2;   /* noise level for middle frequencies */
    float sigma3;   /* noise level for low frequencies */
    float sigma4;   /* noise level for lowest (zero) frequencies */
    float degrid;   /* decrease grid */
    float dehalo;   /* remove halo strength - v.1.9 */
    float hr;       /* halo radius - v1.9 */
    float ht;       /* halo threshold - v1.9 */
    int   ncpu;     /* number of threads - v2.0 */

    /* additional parameterss */
    float *in;
    fftwf_complex *outcache[5], *outtemp;
    fftwf_complex *outrez, *gridsample; /* v1.8 */
    fftwf_plan plan, planinv, plan1;
    int nox, noy;
    int outwidth;
    int outpitch;
    int outpitchelems; /* v.1.7 */

    int outsize;
    int howmanyblocks;

    int ndim[2];
    int inembed[2];
    int onembed[2];

    std::unique_ptr<float[]> wanxl; /* analysis */
    std::unique_ptr<float[]> wanxr;
    std::unique_ptr<float[]> wanyl;
    std::unique_ptr<float[]> wanyr;

    std::unique_ptr<float[]> wsynxl; /* synthesis */
    std::unique_ptr<float[]> wsynxr;
    std::unique_ptr<float[]> wsynyl;
    std::unique_ptr<float[]> wsynyr;

    float *wsharpen;
    float *wdehalo;

    int nlast;  /* frame number at last step */
    int btcurlast;  /* v1.7 */

    fftwf_complex *outLast,*covar, *covarProcess;
    float sigmaSquaredNoiseNormed;
    float sigmaSquaredNoiseNormed2D;
    float sigmaNoiseNormed2D;
    float sigmaMotionNormed;
    float sigmaSquaredSharpenMinNormed;
    float sigmaSquaredSharpenMaxNormed;
    float ht2n; /* halo threshold squared normed */
    float norm; /* normalization factor */

    std::unique_ptr<uint8_t[]> coverbuf; /*  block buffer covering the frame without remainders (with sufficient width and heigth) */
    int coverwidth;
    int coverheight;
    int coverpitch;

    int mirw; /* mirror width for padding */
    int mirh; /* mirror height for padding */

    int planeBase; /* color base value (0 for luma, 128 for chroma) */
    int maxval;

    std::unique_ptr<float[]> mean;

    std::unique_ptr<float[]> pwin;
    float *pattern2d;
    float *pattern3d;
    bool  isPatternSet;
    float psigma;

    std::vector<FFTCacheRec> fftcache;

    template<typename T>
    void InitOverlapPlane(float * __restrict inp0, const T * __restrict srcp0, int src_pitch, int planeBase);
    template<typename T>
    void DecodeOverlapPlane(const float * __restrict inp0, float norm, T * __restrict dstp0, int dst_pitch, int planeBase, int maxval);
    template < typename T, int btcur >
    void Wiener3D( int n, const VSFrameRef *src, VSFrameContext *frame_ctx, const VSAPI *vsapi );

public:
    VSVideoInfo vi;
    VSNodeRef  *node;

    template<typename T>
    void ApplyFilter( int n, VSFrameRef *dst, const VSFrameRef *src, VSFrameContext *frame_ctx, VSCore *core, const VSAPI *vsapi );

    inline bool getIsPatternSet() { return isPatternSet; }

    /* Constructor */
    FFT3DFilter
    (
        float _sigma, float _beta, int _plane, int _bw, int _bh, int _bt, int _ow, int _oh,
        float _kratio, float _sharpen, float _scutoff, float _svr, float _smin, float _smax,
        bool _measure, bool _interlaced, int _wintype,
        int _pframe, int _px, int _py, bool _pshow, float _pcutoff, float _pfactor,
        float _sigma2, float _sigma3, float _sigma4, float _degrid,
        float _dehalo, float _hr, float _ht, int _ncpu,
        VSVideoInfo _vi, VSNodeRef *node
    );

    /* Destructor */
    ~FFT3DFilter();
};

class FFT3DFilterMulti
{
    FFT3DFilter *Clips[3];
    int   bt;       /* block size  along time (mumber of frames), =0 for Kalman, >0 for Wiener */
    int   pframe;   /* noise pattern frame number */
    bool  pshow;    /* show noise pattern */
    float pfactor;  /* noise pattern denoise strength */
    bool  isPatternSet;

public:
    VSVideoInfo vi;
    VSNodeRef  *node;

    void RequestFrame( int n, VSFrameContext *frame_ctx, VSCore *core, const VSAPI *vsapi );
    VSFrameRef *GetFrame( int n, VSFrameContext *frame_ctx, VSCore *core, const VSAPI *vsapi );

    /* Constructor */
    FFT3DFilterMulti
    (
        float _sigma, float _beta, bool _process[3], int _bw, int _bh, int _bt, int _ow, int _oh,
        float _kratio, float _sharpen, float _scutoff, float _svr, float _smin, float _smax,
        bool _measure, bool _interlaced, int _wintype,
        int _pframe, int _px, int _py, bool _pshow, float _pcutoff, float _pfactor,
        float _sigma2, float _sigma3, float _sigma4, float _degrid,
        float _dehalo, float _hr, float _ht, int _ncpu,
        const VSMap *in, const VSAPI *vsapi
    );

    /* Destructor */
    void Free(const VSAPI *vsapi);
};
