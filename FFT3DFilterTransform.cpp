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

void GetAnalysisWindow(int wintype, int ow, int oh, float *wanxl, float *wanxr, float *wanyl, float *wanyr) {
    constexpr float pi = 3.1415926535897932384626433832795f;
    if (wintype == 0) {  
        /* 
         * half-cosine, the same for analysis and synthesis
         * define analysis windows */
        for (int i = 0; i < ow; i++) {
            wanxl[i] = cosf(pi * (i - ow + 0.5f) / (ow * 2)); /* left analize window (half-cosine) */
            wanxr[i] = cosf(pi * (i + 0.5f) / (ow * 2)); /* right analize window (half-cosine) */
        }
        for (int i = 0; i < oh; i++) {
            wanyl[i] = cosf(pi * (i - oh + 0.5f) / (oh * 2));
            wanyr[i] = cosf(pi * (i + 0.5f) / (oh * 2));
        }
    } else if (wintype == 1) {
        /* define analysis windows as more flat (to decrease grid) */
        for (int i = 0; i < ow; i++) {
            wanxl[i] = sqrt(cosf(pi * (i - ow + 0.5f) / (ow * 2)));
            wanxr[i] = sqrt(cosf(pi * (i + 0.5f) / (oh * 2)));
        }
        for (int i = 0; i < oh; i++) {
            wanyl[i] = sqrt(cosf(pi * (i - oh + 0.5f) / (oh * 2)));
            wanyr[i] = sqrt(cosf(pi * (i + 0.5f) / (oh * 2)));
        }
    } else /* (wintype==2) */  {
        /* define analysis windows as flat (to prevent grid) */
        for (int i = 0; i < ow; i++) {
            wanxl[i] = 1;
            wanxr[i] = 1;
        }
        for (int i = 0; i < oh; i++) {
            wanyl[i] = 1;
            wanyr[i] = 1;
        }
    }
}

void GetSynthesisWindow(int wintype, int ow, int oh, float *wsynxl, float *wsynxr, float *wsynyl, float *wsynyr) {
    constexpr float pi = 3.1415926535897932384626433832795f;
    if (wintype == 0) { 
        for (int i = 0; i < ow; i++) {
            wsynxl[i] = cosf(pi * (i - ow + 0.5f) / (ow * 2)); /* left analize window (half-cosine) */
            wsynxr[i] = cosf(pi * (i + 0.5f) / (ow * 2)); /* right analize window (half-cosine) */
        }
        for (int i = 0; i < oh; i++) {
            wsynyl[i] = cosf(pi * (i - oh + 0.5f) / (oh * 2));
            wsynyr[i] = cosf(pi * (i + 0.5f) / (oh * 2));
        }
    } else if (wintype == 1) {
        /* define synthesis as supplenent to rised cosine (Hanning) */
        for (int i = 0; i < ow; i++) {
            float wanxl = sqrt(cosf(pi * (i - ow + 0.5f) / (ow * 2)));
            float wanxr = sqrt(cosf(pi * (i + 0.5f) / (oh * 2)));
            wsynxl[i] = wanxl * wanxl * wanxl; /* left window */
            wsynxr[i] = wanxr * wanxr * wanxr; /* right window */
        }
        for (int i = 0; i < oh; i++) {
            float wanyl = sqrt(cosf(pi * (i - oh + 0.5f) / (oh * 2)));
            float wanyr = sqrt(cosf(pi * (i + 0.5f) / (oh * 2)));
            wsynyl[i] = wanyl * wanyl * wanyl;
            wsynyr[i] = wanyr * wanyr * wanyr;
        }
    } else {
        /* define synthesis as rised cosine (Hanning) */
        for (int i = 0; i < ow; i++) {
            wsynxl[i] = cosf(pi * (i - ow + 0.5f) / (ow * 2));
            wsynxl[i] = wsynxl[i] * wsynxl[i];    /* left window (rised cosine) */
            wsynxr[i] = cosf(pi * (i + 0.5f) / (ow * 2));
            wsynxr[i] = wsynxr[i] * wsynxr[i];    /* right window (falled cosine) */
        }
        for (int i = 0; i < oh; i++) {
            wsynyl[i] = cosf(pi * (i - oh + 0.5f) / (oh * 2));
            wsynyl[i] = wsynyl[i] * wsynyl[i];
            wsynyr[i] = cosf(pi * (i + 0.5f) / (oh * 2));
            wsynyr[i] = wsynyr[i] * wsynyr[i];
        }
    }
}


//
template<typename T>
static void FramePlaneToCoverbuf( int plane, const VSFrameRef *src, T * __restrict coverbuf, int coverwidth, int coverheight, int coverpitch, int mirw, int mirh, bool interlaced, const VSAPI *vsapi )
{
    const T * __restrict srcp = reinterpret_cast<const T *>(vsapi->getReadPtr(src, plane));
    int            src_height = vsapi->getFrameHeight(src, plane);
    int            src_width = vsapi->getFrameWidth(src, plane);
    int            src_pitch = vsapi->getStride(src, plane) / sizeof(T);
    coverpitch /= sizeof(T);

    int h, w;
    int width2 = src_width + src_width + mirw + mirw - 2;
    T * __restrict coverbuf1 = coverbuf + coverpitch * mirh;

    if( !interlaced ) /* progressive */
    {
        for( h = mirh; h < src_height + mirh; h++ )
        {
            for( w = 0; w < mirw; w++ )
            {
                coverbuf1[w] = coverbuf1[mirw + mirw - w]; /* mirror left border */
            }
            memcpy(coverbuf1 + mirw, srcp, src_width * sizeof(T)); /* copy line */
            for( w = src_width + mirw; w < coverwidth; w++ )
            {
                coverbuf1[w] = coverbuf1[width2 - w]; /* mirror right border */
            }
            coverbuf1 += coverpitch;
            srcp      += src_pitch;
        }
    }
    else /* interlaced */
    {
        for( h = mirh; h < src_height / 2 + mirh; h++ ) /* first field */
        {
            for( w = 0; w < mirw; w++ )
            {
                coverbuf1[w] = coverbuf1[mirw + mirw - w]; /* mirror left border */
            }
            memcpy(coverbuf1 + mirw, srcp, src_width * sizeof(T)); /* copy line */
            for( w = src_width + mirw; w < coverwidth; w++ )
            {
                coverbuf1[w] = coverbuf1[width2 - w]; /* mirror right border */
            }
            coverbuf1 += coverpitch;
            srcp      += src_pitch * 2;
        }

        srcp -= src_pitch;
        for( h = src_height / 2 + mirh; h < src_height + mirh; h++ ) /* flip second field */
        {
            for( w = 0; w < mirw; w++ )
            {
                coverbuf1[w] = coverbuf1[mirw + mirw - w]; /* mirror left border */
            }
            memcpy(coverbuf1 + mirw, srcp, src_width * sizeof(T)); /* copy line */
            for( w = src_width + mirw; w < coverwidth; w++ )
            {
                coverbuf1[w] = coverbuf1[width2 - w]; /* mirror right border */
            }
            coverbuf1 += coverpitch;
            srcp      -= src_pitch * 2;
        }
    }

    T *pmirror = coverbuf1 - coverpitch * 2; /* pointer to vertical mirror */
    for( h = src_height + mirh; h < coverheight; h++ )
    {
        memcpy( coverbuf1, pmirror, coverwidth * sizeof(T)); /* mirror bottom line by line */
        coverbuf1 += coverpitch;
        pmirror   -= coverpitch;
    }
    coverbuf1 = coverbuf;
    pmirror   = coverbuf1 + coverpitch * mirh * 2; /* pointer to vertical mirror */
    for( h = 0; h < mirh; h++ )
    {
        memcpy( coverbuf1, pmirror, coverwidth * sizeof(T)); /* mirror bottom line by line */
        coverbuf1 += coverpitch;
        pmirror   -= coverpitch;
    }
}
//-----------------------------------------------------------------------
//
template<typename T>
static void CoverbufToFramePlane(const T * __restrict coverbuf, int coverwidth, int coverheight, int coverpitch, VSFrameRef *dst, int mirw, int mirh, bool interlaced, const VSAPI *vsapi )
{
    T *__restrict dstp = reinterpret_cast<T *>(vsapi->getWritePtr(dst, 0));
    int      dst_height = vsapi->getFrameHeight(dst, 0);
    int      dst_width = vsapi->getFrameWidth(dst, 0);
    int      dst_pitch = vsapi->getStride(dst, 0) / sizeof(T);
    coverpitch /= sizeof(T);

    const T * __restrict coverbuf1 = coverbuf + coverpitch * mirh + mirw;
    if( !interlaced ) /* progressive */
    {
        for( int h = 0; h < dst_height; h++ )
        {
            memcpy( dstp, coverbuf1, dst_width * sizeof(T) ); /* copy pure frame size only */
            dstp      += dst_pitch;
            coverbuf1 += coverpitch;
        }
    }
    else /* interlaced */
    {
        for( int h = 0; h < dst_height; h += 2 )
        {
            memcpy( dstp, coverbuf1, dst_width * sizeof(T)); /* copy pure frame size only */
            dstp      += dst_pitch * 2;
            coverbuf1 += coverpitch;
        }
        /* second field is flipped */
        dstp -= dst_pitch;
        for( int h = 0; h < dst_height; h += 2 )
        {
            memcpy( dstp, coverbuf1, dst_width * sizeof(T)); /* copy pure frame size only */
            dstp      -= dst_pitch * 2;
            coverbuf1 += coverpitch;
        }
    }
}
//-----------------------------------------------------------------------

FFT3DFilterTransformPlane::FFT3DFilterTransformPlane(VSNodeRef *node, int plane_, int wintype, int bw_, int bh_, int ow_, int oh_, bool interlaced_, bool measure, VSCore *core, const VSAPI *vsapi) : plane(plane_), bw(bw_), bh(bh_), ow(ow_), oh(oh_), interlaced(interlaced), in(nullptr, nullptr), plan(nullptr, nullptr) {
    if (ow < 0)
        ow = bw / 3;
    if (oh < 0)
        oh = bh / 3;

    const VSVideoInfo *srcvi = vsapi->getVideoInfo(node);

    planeBase = (plane && srcvi->format->sampleType == stInteger) ? (1 << (srcvi->format->bitsPerSample - 1)) : 0;

    nox = ((srcvi->width >> (plane ? srcvi->format->subSamplingW : 0)) - ow + (bw - ow - 1)) / (bw - ow);
    noy = ((srcvi->height >> (plane ? srcvi->format->subSamplingH : 0)) - oh + (bh - oh - 1)) / (bh - oh);

    wanxl = std::unique_ptr<float[]>(new float[ow]);
    wanxr = std::unique_ptr<float[]>(new float[ow]);
    wanyl = std::unique_ptr<float[]>(new float[oh]);
    wanyr = std::unique_ptr<float[]>(new float[oh]);

    GetAnalysisWindow(wintype, ow, oh, wanxl.get(), wanxr.get(), wanyl.get(), wanyr.get());

    /* padding by 1 block per side */
    nox += 2;
    noy += 2;
    mirw = bw - ow; /* set mirror size as block interval */
    mirh = bh - oh;

    coverwidth = nox * (bw - ow) + ow;
    coverheight = noy * (bh - oh) + oh;
    coverpitch = ((coverwidth + 7) / 8) * 8 * srcvi->format->bytesPerSample;
    coverbuf = std::unique_ptr<uint8_t[]>(new uint8_t[coverheight * coverpitch]);

    int insize = bw * bh * nox * noy;
    in = std::unique_ptr<float[], decltype(&fftw_free)>(fftwf_alloc_real(insize), fftwf_free);
    outwidth = bw / 2 + 1;                  /* width (pitch) of complex fft block */
    outpitchelems = ((outwidth + 1) / 2) * 2;
    outsize = outpitchelems * bh * nox * noy;   /* replace outwidth to outpitchelems here and below in v1.7 */

    int planFlags = (measure ? FFTW_MEASURE : FFTW_ESTIMATE) | FFTW_DESTROY_INPUT;
    int ndim[2] = { bh, bw }; 
    int idist = bw * bh;
    int odist = outpitchelems * bh;
    int inembed[2] = { bh, bw };
    int onembed[2] = { bh, outpitchelems };
    int howmanyblocks = nox * noy;

    dstvi = {};
    dstvi.format = vsapi->getFormatPreset(pfGrayS, core);
    dstvi.width = outsize;
    dstvi.height = 1;

    VSFrameRef *outrez = vsapi->newVideoFrame(dstvi.format, dstvi.width, dstvi.height, nullptr, core);

    plan = std::unique_ptr<fftwf_plan_s, decltype(&fftwf_destroy_plan)>(fftwf_plan_many_dft_r2c(2, ndim, howmanyblocks,
        in.get(), inembed, 1, idist, reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(outrez, 0)), onembed, 1, odist, planFlags), fftwf_destroy_plan);

    vsapi->freeFrame(outrez);
}

const VSFrameRef *VS_CC FFT3DFilterTransformPlane::GetFrame(int n, int activation_reason, void **instance_data, void **frame_data, VSFrameContext *frame_ctx, VSCore *core, const VSAPI *vsapi) {
    FFT3DFilterTransformPlane *data = reinterpret_cast<FFT3DFilterTransformPlane *>(*instance_data);
    if (activation_reason == arInitial) {
        vsapi->requestFrameFilter(n, data->node, frame_ctx);
    } else if (activation_reason == arAllFramesReady) {
        const VSFrameRef *src = vsapi->getFrameFilter(n, data->node, frame_ctx);
        const VSFormat *fi = vsapi->getFrameFormat(src);

        if (fi->bytesPerSample == 1) {
            FramePlaneToCoverbuf<uint8_t>(data->plane, src, reinterpret_cast<uint8_t *>(data->coverbuf.get()), data->coverwidth, data->coverheight, data->coverpitch, data->mirw, data->mirh, data->interlaced, vsapi);
            data->InitOverlapPlane<uint8_t>(data->in.get(), reinterpret_cast<uint8_t *>(data->coverbuf.get()), data->coverpitch, data->planeBase);
        }

        vsapi->freeFrame(src);

        VSFrameRef *dst = vsapi->newVideoFrame(data->dstvi.format, data->dstvi.width, data->dstvi.height, nullptr, core);
        fftwf_execute_dft_r2c(data->plan.get(), data->in.get(), reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)));
    }

    return nullptr;
}

void VS_CC FFT3DFilterTransformPlane::Free(void *instance_data, VSCore *core, const VSAPI *vsapi) {
    FFT3DFilterTransformPlane *data = reinterpret_cast<FFT3DFilterTransformPlane *>(instance_data);
    vsapi->freeNode(data->node);
    delete data;
}

//-----------------------------------------------------------------------
/* put source bytes to float array of overlapped blocks
 * use analysis windows */
template<typename T>
void FFT3DFilterTransformPlane::InitOverlapPlane( float * __restrict inp0, const T * __restrict srcp0, int src_pitch, int planeBase )
{
    int w, h;
    int ihx, ihy;
    const T * __restrict srcp = srcp0;
    float ftmp;
    int xoffset = bh * bw - (bw - ow); /* skip frames */
    int yoffset = bw * nox * bh - bw * (bh - oh); /* vertical offset of same block (overlap) */
    src_pitch /= sizeof(T);

    float * __restrict inp = inp0;

    ihy = 0; /* first top (big non-overlapped) part */
    {
        for( h = 0; h < oh; h++ )
        {
            inp = inp0 + h * bw;
            for( w = 0; w < ow; w++ )   /* left part  (non-overlapped) row of first block */
            {
                inp[w] = float(wanxl[w] * wanyl[h] * (srcp[w] - planeBase));   /* Copy each byte from source to float array */
            }
            for( w = ow; w < bw - ow; w++ )   /* left part  (non-overlapped) row of first block */
            {
                inp[w] = float(wanyl[h] * (srcp[w] - planeBase));   /* Copy each byte from source to float array */
            }
            inp  += bw - ow;
            srcp += bw - ow;
            for( ihx =1; ihx < nox; ihx += 1 ) /* middle horizontal blocks */
            {
                for( w = 0; w < ow; w++ )   /* first part (overlapped) row of block */
                {
                    ftmp = float(wanyl[h] * (srcp[w] - planeBase));   /* Copy each byte from source to float array */
                    inp[w          ] = ftmp * wanxr[w]; /* cur block */
                    inp[w + xoffset] = ftmp * wanxl[w]; /* overlapped Copy - next block */
                }
                inp  += ow;
                inp  += xoffset;
                srcp += ow;
                for( w = 0; w < bw - ow - ow; w++ )   /* center part  (non-overlapped) row of first block */
                {
                    inp[w] = float(wanyl[h] * (srcp[w] - planeBase));   /* Copy each byte from source to float array */
                }
                inp  += bw - ow - ow;
                srcp += bw - ow - ow;
            }
            for( w = 0; w < ow; w++ )   /* last part (non-overlapped) of line of last block */
            {
                inp[w] = float(wanxr[w] * wanyl[h] * (srcp[w] - planeBase));   /* Copy each byte from source to float array */
            }
            inp  += ow;
            srcp += ow;
            srcp += (src_pitch - coverwidth);  /* Add the pitch of one line (in bytes) to the source image. */
        }
        for( h = oh; h < bh - oh; h++ )
        {
            inp = inp0 + h * bw;
            for( w = 0; w < ow; w++ )   /* left part  (non-overlapped) row of first block */
            {
                inp[w] = float(wanxl[w] * (srcp[w] - planeBase));   /* Copy each byte from source to float array */
            }
            for( w = ow; w < bw - ow; w++ )   /* left part  (non-overlapped) row of first block */
            {
                inp[w] = float((srcp[w] - planeBase));   /* Copy each byte from source to float array */
            }
            inp  += bw - ow;
            srcp += bw - ow;
            for( ihx = 1; ihx < nox; ihx += 1 ) /* middle horizontal blocks */
            {
                for( w = 0; w < ow; w++ )   /* first part (overlapped) row of block */
                {
                    ftmp = float((srcp[w] - planeBase));  /* Copy each byte from source to float array */
                    inp[w          ] = ftmp * wanxr[w]; /* cur block */
                    inp[w + xoffset] = ftmp * wanxl[w]; /* overlapped Copy - next block */
                }
                inp  += ow;
                inp  += xoffset;
                srcp += ow;
                for( w = 0; w < bw - ow - ow; w++ )   /* center part  (non-overlapped) row of first block */
                {
                    inp[w] = float((srcp[w] - planeBase));   /* Copy each byte from source to float array */
                }
                inp  += bw - ow - ow;
                srcp += bw - ow - ow;
            }
            for( w = 0; w < ow; w++ )   /* last part (non-overlapped) line of last block */
            {
                inp[w] = float(wanxr[w] * (srcp[w] - planeBase));   /* Copy each byte from source to float array */
            }
            inp  += ow;
            srcp += ow;

            srcp += (src_pitch - coverwidth);  /* Add the pitch of one line (in bytes) to the source image. */
        }
    }

    for( ihy = 1; ihy < noy; ihy += 1 ) /* middle vertical */
    {
        for( h = 0; h < oh; h++ ) /* top overlapped part */
        {
            inp = inp0 + (ihy - 1) * (yoffset + (bh - oh) * bw) + (bh - oh) * bw + h * bw;
            for( w = 0; w < ow; w++ )   /* first half line of first block */
            {
                ftmp = float(wanxl[w] * (srcp[w] - planeBase));
                inp[w          ] = ftmp * wanyr[h];   /* Copy each byte from source to float array */
                inp[w + yoffset] = ftmp * wanyl[h];   /* y overlapped */
            }
            for( w = ow; w < bw - ow; w++ )   /* first half line of first block */
            {
                ftmp = float((srcp[w] - planeBase));
                inp[w          ] = ftmp * wanyr[h];   /* Copy each byte from source to float array */
                inp[w + yoffset] = ftmp * wanyl[h];   /* y overlapped */
            }
            inp  += bw - ow;
            srcp += bw - ow;
            for( ihx = 1; ihx < nox; ihx++ ) /* middle blocks */
            {
                for( w = 0; w < ow; w++ )   /* half overlapped line of block */
                {
                    ftmp = float((srcp[w] - planeBase));   /* Copy each byte from source to float array */
                    inp[w                    ] = ftmp * wanxr[w] * wanyr[h];
                    inp[w + xoffset          ] = ftmp * wanxl[w] * wanyr[h];   /* x overlapped */
                    inp[w           + yoffset] = ftmp * wanxr[w] * wanyl[h];
                    inp[w + xoffset + yoffset] = ftmp * wanxl[w] * wanyl[h];   /* x overlapped */
                }
                inp  += ow;
                inp  += xoffset;
                srcp += ow;
                for( w = 0; w < bw - ow - ow; w++ )   /* half non-overlapped line of block */
                {
                    ftmp = float((srcp[w] - planeBase));   /* Copy each byte from source to float array */
                    inp[w          ] = ftmp * wanyr[h];
                    inp[w + yoffset] = ftmp * wanyl[h];
                }
                inp  += bw - ow - ow;
                srcp += bw - ow - ow;
            }
            for( w = 0; w < ow; w++ )   /* last half line of last block */
            {
                ftmp = float(wanxr[w] * (srcp[w] - planeBase)); /* Copy each byte from source to float array */
                inp[w          ] = ftmp * wanyr[h];
                inp[w + yoffset] = ftmp * wanyl[h];
            }
            inp  += ow;
            srcp += ow;

            srcp += (src_pitch - coverwidth);  /* Add the pitch of one line (in bytes) to the source image. */
        }
        /* middle  vertical nonovelapped part */
        for( h = 0; h < bh - oh - oh; h++ )
        {
            inp = inp0 + (ihy - 1) * (yoffset + (bh - oh) * bw) + (bh) * bw + h * bw + yoffset;
            for( w = 0; w < ow; w++ )   /* first half line of first block */
            {
                ftmp = float(wanxl[w] * (srcp[w] - planeBase));
                inp[w] = ftmp;   /* Copy each byte from source to float array */
            }
            for( w = ow; w < bw - ow; w++ )   /* first half line of first block */
            {
                ftmp = float((srcp[w] - planeBase));
                inp[w] = ftmp;   /* Copy each byte from source to float array */
            }
            inp  += bw - ow;
            srcp += bw - ow;
            for( ihx = 1; ihx < nox; ihx++ ) /* middle blocks */
            {
                for( w = 0; w < ow; w++ )   /* half overlapped line of block */
                {
                    ftmp = float((srcp[w] - planeBase));   /* Copy each byte from source to float array */
                    inp[w          ] = ftmp * wanxr[w];
                    inp[w + xoffset] = ftmp * wanxl[w];   /* x overlapped */
                }
                inp  += ow;
                inp  += xoffset;
                srcp += ow;
                for( w = 0; w < bw - ow - ow; w++ )   /* half non-overlapped line of block */
                {
                    ftmp = float((srcp[w] - planeBase));   /* Copy each byte from source to float array */
                    inp[w] = ftmp;
                }
                inp  += bw - ow - ow;
                srcp += bw - ow - ow;
            }
            for( w = 0; w < ow; w++ )   /* last half line of last block */
            {
                ftmp = float(wanxr[w] * (srcp[w] - planeBase)); /* Copy each byte from source to float array */
                inp[w] = ftmp;
            }
            inp  += ow;
            srcp += ow;

            srcp += (src_pitch - coverwidth);  /* Add the pitch of one line (in bytes) to the source image. */
        }

    }

    ihy = noy ; /* last bottom  part */
    {
        for( h = 0; h < oh; h++ )
        {
            inp = inp0 + (ihy - 1) * (yoffset + (bh - oh) * bw) + (bh - oh) * bw + h * bw;
            for( w = 0; w < ow; w++ )   /* first half line of first block */
            {
                ftmp = float(wanxl[w] * wanyr[h] * (srcp[w] - planeBase));
                inp[w] = ftmp;   /* Copy each byte from source to float array */
            }
            for( w = ow; w < bw - ow; w++ )   /* first half line of first block */
            {
                ftmp = float(wanyr[h] * (srcp[w] - planeBase));
                inp[w] = ftmp;   /* Copy each byte from source to float array */
            }
            inp  += bw - ow;
            srcp += bw - ow;
            for( ihx = 1; ihx < nox; ihx++ ) /* middle blocks */
            {
                for( w = 0; w < ow; w++ )   /* half line of block */
                {
                    float ftmp = float(wanyr[h] * (srcp[w] - planeBase));   /* Copy each byte from source to float array */
                    inp[w          ] = ftmp * wanxr[w];
                    inp[w + xoffset] = ftmp * wanxl[w];   /* overlapped Copy */
                }
                inp  += ow;
                inp  += xoffset;
                srcp += ow;
                for( w = 0; w < bw - ow - ow; w++ )   /* center part  (non-overlapped) row of first block */
                {
                    inp[w] = float(wanyr[h] * (srcp[w] - planeBase));   /* Copy each byte from source to float array */
                }
                inp  += bw - ow - ow;
                srcp += bw - ow - ow;
            }
            for( w = 0; w < ow; w++ )   /* last half line of last block */
            {
                ftmp = float(wanxr[w] * wanyr[h] * (srcp[w] - planeBase));
                inp[w] = ftmp;   /* Copy each byte from source to float array */
            }
            inp  += ow;
            srcp += ow;

            srcp += (src_pitch - coverwidth);  /* Add the pitch of one line (in bytes) to the source image. */
        }

    }
}
//
//-----------------------------------------------------------------------------------------

FFT3DFilterInvTransform::FFT3DFilterInvTransform(VSNodeRef *node, const VSVideoInfo *dstvi, int plane, int wintype, int bw_, int bh_, int ow_, int oh_, bool interlaced_, bool measure, VSCore *core, const VSAPI *vsapi) : bw(bw_), bh(bh_), ow(ow_), oh(oh_), interlaced(interlaced), in(nullptr, nullptr), planinv(nullptr, nullptr) {
    if (ow < 0)
        ow = bw / 3;
    if (oh < 0)
        oh = bh / 3;

    const VSVideoInfo *srcvi = vsapi->getVideoInfo(node);

    // FIXME, dstvi is the extracted grayscale format of the source plane

    planeBase = (plane && dstvi->format->sampleType == stInteger) ? (1 << (dstvi->format->bitsPerSample - 1)) : 0;

    nox = ((dstvi->width >> (plane ? dstvi->format->subSamplingW : 0)) - ow + (bw - ow - 1)) / (bw - ow);
    noy = ((dstvi->height >> (plane ? dstvi->format->subSamplingH : 0)) - oh + (bh - oh - 1)) / (bh - oh);

    wsynxl = std::unique_ptr<float[]>(new float[ow]);
    wsynxr = std::unique_ptr<float[]>(new float[ow]);
    wsynyl = std::unique_ptr<float[]>(new float[oh]);
    wsynyr = std::unique_ptr<float[]>(new float[oh]);

    GetSynthesisWindow(wintype, ow, oh, wsynxl.get(), wsynxr.get(), wsynyl.get(), wsynyr.get());

    /* padding by 1 block per side */
    nox += 2;
    noy += 2;
    mirw = bw - ow; /* set mirror size as block interval */
    mirh = bh - oh;

    coverwidth = nox * (bw - ow) + ow;
    coverheight = noy * (bh - oh) + oh;
    coverpitch = ((coverwidth + 7) / 8) * 8 * dstvi->format->bytesPerSample;
    coverbuf = std::unique_ptr<uint8_t[]>(new uint8_t[coverheight * coverpitch]);

    int insize = bw * bh * nox * noy;
    in = std::unique_ptr<float[], decltype(&fftw_free)>(fftwf_alloc_real(insize), fftwf_free);
    outwidth = bw / 2 + 1;                  /* width (pitch) of complex fft block */
    outpitchelems = ((outwidth + 1) / 2) * 2;
    outsize = outpitchelems * bh * nox * noy;   /* replace outwidth to outpitchelems here and below in v1.7 */
    norm = 1.0f / (bw * bh); /* do not forget set FFT normalization factor */

    int planFlags = (measure ? FFTW_MEASURE : FFTW_ESTIMATE) | FFTW_PRESERVE_INPUT; // needed since a read only frame is the source
    int ndim[2] = { bh, bw };
    int idist = bw * bh;
    int odist = outpitchelems * bh;
    int inembed[2] = { bh, bw };
    int onembed[2] = { bh, outpitchelems };
    int howmanyblocks = nox * noy;

    VSFrameRef *src = vsapi->newVideoFrame(srcvi->format, srcvi->width, srcvi->height, nullptr, core);

    planinv = std::unique_ptr<fftwf_plan_s, decltype(&fftwf_destroy_plan)>(fftwf_plan_many_dft_c2r(2, ndim, howmanyblocks,
        reinterpret_cast<fftwf_complex *>(const_cast<uint8_t *>(vsapi->getReadPtr(src, 0))), onembed, 1, odist, in.get(), inembed, 1, idist, planFlags), fftwf_destroy_plan);
}

const VSFrameRef *VS_CC FFT3DFilterInvTransform::GetFrame(int n, int activation_reason, void **instance_data, void **frame_data, VSFrameContext *frame_ctx, VSCore *core, const VSAPI *vsapi) {
    FFT3DFilterInvTransform *data = reinterpret_cast<FFT3DFilterInvTransform *>(*instance_data);
    if (activation_reason == arInitial) {
        vsapi->requestFrameFilter(n, data->node, frame_ctx);
    } else if (activation_reason == arAllFramesReady) {
        const VSFrameRef *src = vsapi->getFrameFilter(n, data->node, frame_ctx);
        const VSFormat *fi = vsapi->getFrameFormat(src);

        // FFTW_PRESERVE_INPUT, is used so despite the cast the source pointer isn't used
        fftwf_execute_dft_c2r(data->planinv.get(), reinterpret_cast<fftwf_complex *>(const_cast<uint8_t *>(vsapi->getReadPtr(src, 0))), data->in.get());
        vsapi->freeFrame(src);

        VSFrameRef *dst = vsapi->newVideoFrame(data->dstvi.format, data->dstvi.width, data->dstvi.height, nullptr, core);

        if (fi->bytesPerSample == 1) {
            data->DecodeOverlapPlane(data->in.get(), data->norm, reinterpret_cast<uint8_t *>(data->coverbuf.get()), data->coverpitch, data->planeBase, data->maxval);
            CoverbufToFramePlane(reinterpret_cast<uint8_t *>(data->coverbuf.get()), data->coverwidth, data->coverheight, data->coverpitch, dst, data->mirw, data->mirh, data->interlaced, vsapi);
        }

        return dst;
    }

    return nullptr;
}

void VS_CC FFT3DFilterInvTransform::Free(void *instance_data, VSCore *core, const VSAPI *vsapi) {
    FFT3DFilterInvTransform *data = reinterpret_cast<FFT3DFilterInvTransform *>(instance_data);
    vsapi->freeNode(data->node);
    delete data;
}

/* make destination frame plane from overlaped blocks
 * use synthesis windows wsynxl, wsynxr, wsynyl, wsynyr */
template<typename T>
void FFT3DFilterInvTransform::DecodeOverlapPlane( const float * __restrict inp0, float norm, T * __restrict dstp0, int dst_pitch, int planeBase, int maxval )
{
    int w, h;
    int ihx, ihy;
    T * __restrict dstp = dstp0;
    const float * __restrict inp = inp0;
    int xoffset = bh * bw - (bw - ow);
    int yoffset = bw * nox * bh - bw * (bh - oh); /* vertical offset of same block (overlap) */
    dst_pitch /= sizeof(T);

    ihy = 0; /* first top big non-overlapped) part */
    {
        for( h = 0; h < bh - oh; h++ )
        {
            inp = inp0 + h * bw;
            for( w = 0; w < bw - ow; w++ )   /* first half line of first block */
            {   
                if constexpr (std::is_integral<T>::value)
                    dstp[w] = std::min(maxval, std::max( 0, (int)(inp[w] * norm) + planeBase ) ); /* Copy each byte from float array to dest with windows */
                else 
                    dstp[w] = inp[w] * norm;
            }
            inp  += bw - ow;
            dstp += bw - ow;
            for( ihx = 1; ihx < nox; ihx++ ) /* middle horizontal half-blocks */
            {
                for( w = 0; w < ow; w++ )   /* half line of block */
                {
                    if constexpr (std::is_integral<T>::value)
                        dstp[w] = std::min(maxval, std::max( 0, (int)((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w]) * norm) + planeBase ) );  /* overlapped Copy */
                    else
                        dstp[w] = (inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w]) * norm;
                }
                inp  += xoffset + ow;
                dstp += ow;
                for( w = 0; w < bw - ow - ow; w++ )   /* first half line of first block */
                {
                    if constexpr (std::is_integral<T>::value)
                        dstp[w] = std::min(maxval, std::max( 0, (int)(inp[w] * norm) + planeBase ) );   /* Copy each byte from float array to dest with windows */
                    else
                        dstp[w] = inp[w] * norm;
                }
                inp  += bw - ow - ow;
                dstp += bw - ow - ow;
            }
            for( w = 0; w < ow; w++ )   /* last half line of last block */
            {
                if constexpr (std::is_integral<T>::value)
                    dstp[w] = std::min(maxval, std::max( 0,(int)(inp[w] * norm) + planeBase ) );
                else
                    dstp[w] = inp[w] * norm;
            }
            inp  += ow;
            dstp += ow;

            dstp += (dst_pitch - coverwidth);  /* Add the pitch of one line (in bytes) to the dest image. */
        }
    }

    for( ihy = 1; ihy < noy; ihy += 1 ) /* middle vertical */
    {
        for( h = 0; h < oh; h++ ) /* top overlapped part */
        {
            inp = inp0 + (ihy - 1) * (yoffset + (bh - oh) * bw) + (bh - oh) * bw + h * bw;

            float wsynyrh = wsynyr[h] * norm; /* remove from cycle for speed */
            float wsynylh = wsynyl[h] * norm;

            for( w = 0; w < bw - ow; w++ )   /* first half line of first block */
            {
                if constexpr (std::is_integral<T>::value)
                    dstp[w] = std::min(maxval, std::max( 0, (int)((inp[w] * wsynyrh + inp[w + yoffset] * wsynylh)) + planeBase ) );   /* y overlapped */
                else
                    dstp[w] = inp[w] * wsynyrh + inp[w + yoffset] * wsynylh;
            }
            inp  += bw - ow;
            dstp += bw - ow;
            for( ihx = 1; ihx < nox; ihx++ ) /* middle blocks */
            {
                for( w = 0; w < ow; w++ )   /* half overlapped line of block */
                {
                    if constexpr (std::is_integral<T>::value)
                        dstp[w] = std::min(maxval, std::max( 0, (int)(((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w]) * wsynyrh
                            + (inp[w + yoffset] * wsynxr[w] + inp[w + xoffset + yoffset] * wsynxl[w]) * wsynylh)) + planeBase ) );   /* x overlapped */
                    else
                        dstp[w] = (inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w]) * wsynyrh
                            + (inp[w + yoffset] * wsynxr[w] + inp[w + xoffset + yoffset] * wsynxl[w]) * wsynylh;
                }
                inp  += xoffset + ow;
                dstp += ow;
                for( w = 0; w < bw - ow - ow; w++ )   /* double minus - half non-overlapped line of block */
                {
                    if constexpr (std::is_integral<T>::value)
                        dstp[w] = std::min(maxval, std::max( 0, (int)((inp[w] * wsynyrh + inp[w + yoffset] * wsynylh )) + planeBase ) );
                    else
                        dstp[w] = inp[w] * wsynyrh + inp[w + yoffset] * wsynylh;
                }
                inp  += bw - ow - ow;
                dstp += bw - ow - ow;
            }
            for( w = 0; w < ow; w++ )   /* last half line of last block */
            {
                if constexpr (std::is_integral<T>::value)
                    dstp[w] = std::min(maxval, std::max( 0, (int)((inp[w] * wsynyrh + inp[w + yoffset] * wsynylh)) + planeBase ) );
                else
                    dstp[w] = inp[w] * wsynyrh + inp[w + yoffset] * wsynylh;
            }
            inp  += ow;
            dstp += ow;

            dstp += (dst_pitch - coverwidth);  /* Add the pitch of one line (in bytes) to the source image. */
        }
        /* middle  vertical non-ovelapped part */
        for( h = 0; h < (bh - oh - oh); h++ )
        {
            inp = inp0 + (ihy - 1) * (yoffset + (bh - oh) * bw) + (bh) * bw + h * bw + yoffset;
            for( w = 0; w < bw - ow; w++ )   /* first half line of first block */
            {
                if constexpr (std::is_integral<T>::value)
                    dstp[w] = std::min(maxval, std::max( 0, (int)(inp[w] * norm) + planeBase ) );
                else
                    dstp[w] = inp[w] * norm;
            }
            inp  += bw - ow;
            dstp += bw - ow;
            for( ihx = 1; ihx < nox; ihx++ ) /* middle blocks */
            {
                for( w = 0; w < ow; w++ )   /* half overlapped line of block */
                {
                    if constexpr (std::is_integral<T>::value)
                        dstp[w] = std::min(maxval, std::max( 0, (int)((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w] ) * norm) + planeBase ) );   /* x overlapped */
                    else
                        dstp[w] = (inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w]) * norm;
                }
                inp  += xoffset + ow;
                dstp += ow;
                for( w = 0; w < bw - ow - ow; w++ )   /* half non-overlapped line of block */
                {
                    if constexpr (std::is_integral<T>::value)
                        dstp[w] = std::min(maxval, std::max( 0, (int)(inp[w] * norm) + planeBase ) );
                    else
                        dstp[w] = inp[w] * norm;
                }
                inp  += bw - ow - ow;
                dstp += bw - ow - ow;
            }
            for( w = 0; w < ow; w++ )   /* last half line of last block */
            {
                if constexpr (std::is_integral<T>::value)
                    dstp[w] = std::min(maxval, std::max( 0, (int)(inp[w] * norm) + planeBase ) );
                else
                    dstp[w] = inp[w] * norm;
            }
            inp  += ow;
            dstp += ow;

            dstp += (dst_pitch - coverwidth);  /* Add the pitch of one line (in bytes) to the source image. */
        }
    }

    ihy = noy ; /* last bottom part */
    {
        for( h = 0; h < oh; h++ )
        {
            inp = inp0 + (ihy - 1) * (yoffset + (bh - oh) * bw) + (bh - oh) * bw + h * bw;
            for( w = 0; w < bw - ow; w++ )   /* first half line of first block */
            {
                if constexpr (std::is_integral<T>::value)
                    dstp[w] = std::min(maxval, std::max( 0, (int)(inp[w] * norm) + planeBase ) );
                else
                    dstp[w] = inp[w] * norm;
            }
            inp  += bw - ow;
            dstp += bw - ow;
            for( ihx = 1; ihx < nox; ihx++ ) /* middle blocks */
            {
                for( w = 0; w < ow; w++ )   /* half line of block */
                {
                    if constexpr (std::is_integral<T>::value)
                        dstp[w] = std::min(maxval, std::max( 0, (int)((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w]) * norm) + planeBase ) );  /* overlapped Copy */
                    else
                        dstp[w] = (inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w]) * norm;
                }
                inp  += xoffset + ow;
                dstp += ow;
                for( w = 0; w < bw - ow - ow; w++ )   /* half line of block */
                {
                    if constexpr (std::is_integral<T>::value)
                        dstp[w] = std::min(maxval, std::max( 0, (int)(inp[w] * norm) + planeBase ) );
                    else
                        dstp[w] = inp[w] * norm;
                }
                inp  += bw - ow - ow;
                dstp += bw - ow - ow;
            }
            for( w = 0; w < ow; w++ )   /* last half line of last block */
            {
                if constexpr (std::is_integral<T>::value)
                    dstp[w] = std::min(maxval, std::max( 0, (int)(inp[w] * norm) + planeBase ) );
                else
                    dstp[w] = inp[w] * norm;
            }
            inp  += ow;
            dstp += ow;

            dstp += (dst_pitch - coverwidth);  /* Add the pitch of one line (in bytes) to the source image. */
        }
    }
}
