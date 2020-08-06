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

template<typename T>
static void fft3d_memset(T *dst, T val, size_t count) {
    for (size_t i = 0; i < count; i++)
        dst[i] = val;
}

static void GetAnalysisWindow(int wintype, int ow, int oh, float *wanxl, float *wanxr, float *wanyl, float *wanyr) {
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

static void GetSynthesisWindow(int wintype, int ow, int oh, float *wsynxl, float *wsynxr, float *wsynyl, float *wsynyr) {
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

static void GetPatternWindow(int bw, int bh, int outwidth, int outpitchelems, float pcutoff, float *pwin) {
    for (int j = 0; j < bh; j++) {
        float fh2;
        if (j < bh / 2)
            fh2 = (j * 2.0f / bh) * (j * 2.0f / bh);
        else
            fh2 = ((bh - 1 - j) * 2.0f / bh) * ((bh - 1 - j) * 2.0f / bh);
        for (int i = 0; i < outwidth; i++) {
            float fw2 = (i * 2.0f / bw) * (j * 2.0f / bw);
            pwin[i + j * outpitchelems] = (fh2 + fw2) / (fh2 + fw2 + pcutoff * pcutoff);
        }
    }
}


//
template<typename T>
static void FramePlaneToCoverbuf( int plane, const VSFrameRef *src, T * __restrict coverbuf, int coverwidth, int coverheight, ptrdiff_t coverpitch, int mirw, int mirh, bool interlaced, const VSAPI *vsapi )
{
    const T * __restrict srcp = reinterpret_cast<const T *>(vsapi->getReadPtr(src, plane));
    int            src_height = vsapi->getFrameHeight(src, plane);
    int            src_width = vsapi->getFrameWidth(src, plane);
    ptrdiff_t      src_pitch = vsapi->getStride(src, plane) / sizeof(T);
    coverpitch /= sizeof(T);

    int width2 = src_width + src_width + mirw + mirw - 2;
    T * __restrict coverbuf1 = coverbuf + coverpitch * mirh;

    if( !interlaced ) /* progressive */
    {
        for(int h = mirh; h < src_height + mirh; h++ )
        {
            for(int w = 0; w < mirw; w++ )
            {
                coverbuf1[w] = coverbuf1[mirw + mirw - w]; /* mirror left border */
            }
            memcpy(coverbuf1 + mirw, srcp, src_width * sizeof(T)); /* copy line */
            for(int w = src_width + mirw; w < coverwidth; w++ )
            {
                coverbuf1[w] = coverbuf1[width2 - w]; /* mirror right border */
            }
            coverbuf1 += coverpitch;
            srcp      += src_pitch;
        }
    }
    else /* interlaced */
    {
        for(int h = mirh; h < src_height / 2 + mirh; h++ ) /* first field */
        {
            for(int w = 0; w < mirw; w++ )
            {
                coverbuf1[w] = coverbuf1[mirw + mirw - w]; /* mirror left border */
            }
            memcpy(coverbuf1 + mirw, srcp, src_width * sizeof(T)); /* copy line */
            for(int w = src_width + mirw; w < coverwidth; w++ )
            {
                coverbuf1[w] = coverbuf1[width2 - w]; /* mirror right border */
            }
            coverbuf1 += coverpitch;
            srcp      += src_pitch * 2;
        }

        srcp -= src_pitch;
        for(int h = src_height / 2 + mirh; h < src_height + mirh; h++ ) /* flip second field */
        {
            for(int w = 0; w < mirw; w++ )
            {
                coverbuf1[w] = coverbuf1[mirw + mirw - w]; /* mirror left border */
            }
            memcpy(coverbuf1 + mirw, srcp, src_width * sizeof(T)); /* copy line */
            for(int w = src_width + mirw; w < coverwidth; w++ )
            {
                coverbuf1[w] = coverbuf1[width2 - w]; /* mirror right border */
            }
            coverbuf1 += coverpitch;
            srcp      -= src_pitch * 2;
        }
    }

    T *pmirror = coverbuf1 - coverpitch * 2; /* pointer to vertical mirror */
    for(int h = src_height + mirh; h < coverheight; h++ )
    {
        memcpy( coverbuf1, pmirror, coverwidth * sizeof(T)); /* mirror bottom line by line */
        coverbuf1 += coverpitch;
        pmirror   -= coverpitch;
    }
    coverbuf1 = coverbuf;
    pmirror   = coverbuf1 + coverpitch * mirh * 2; /* pointer to vertical mirror */
    for(int h = 0; h < mirh; h++ )
    {
        memcpy( coverbuf1, pmirror, coverwidth * sizeof(T)); /* mirror bottom line by line */
        coverbuf1 += coverpitch;
        pmirror   -= coverpitch;
    }
}
//-----------------------------------------------------------------------
//
template<typename T>
static void CoverbufToFramePlane(const T * __restrict coverbuf, int coverwidth, int coverheight, ptrdiff_t coverpitch, VSFrameRef *dst, int mirw, int mirh, bool interlaced, const VSAPI *vsapi )
{
    T *__restrict dstp = reinterpret_cast<T *>(vsapi->getWritePtr(dst, 0));
    int      dst_height = vsapi->getFrameHeight(dst, 0);
    int      dst_width = vsapi->getFrameWidth(dst, 0);
    ptrdiff_t   dst_pitch = vsapi->getStride(dst, 0) / sizeof(T);
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

//-----------------------------------------------------------------------
/* put source bytes to float array of overlapped blocks
 * use analysis windows */
template<typename T>
static void InitOverlapPlane(float *__restrict inp0, const T *__restrict srcp0, ptrdiff_t src_pitch, float *__restrict wanxl, float *__restrict wanxr, float *__restrict wanyl, float *__restrict wanyr, int bw, int bh, int ow, int oh, int nox, int noy, int coverwidth, int planeBase) {
    int ihx, ihy;
    const T *__restrict srcp = srcp0;
    float ftmp;
    int xoffset = bh * bw - (bw - ow); /* skip frames */
    int yoffset = bw * nox * bh - bw * (bh - oh); /* vertical offset of same block (overlap) */
    src_pitch /= sizeof(T);

    float *__restrict inp = inp0;

    if constexpr (std::is_floating_point_v<T>)
        planeBase = 0;

    ihy = 0; /* first top (big non-overlapped) part */
    {
        for (int h = 0; h < oh; h++) {
            inp = inp0 + h * bw;
            for (int w = 0; w < ow; w++)   /* left part  (non-overlapped) row of first block */
            {
                inp[w] = float(wanxl[w] * wanyl[h] * (srcp[w] - planeBase));   /* Copy each byte from source to float array */
            }
            for (int w = ow; w < bw - ow; w++)   /* left part  (non-overlapped) row of first block */
            {
                inp[w] = float(wanyl[h] * (srcp[w] - planeBase));   /* Copy each byte from source to float array */
            }
            inp += bw - ow;
            srcp += bw - ow;
            for (ihx = 1; ihx < nox; ihx += 1) /* middle horizontal blocks */
            {
                for (int w = 0; w < ow; w++)   /* first part (overlapped) row of block */
                {
                    ftmp = float(wanyl[h] * (srcp[w] - planeBase));   /* Copy each byte from source to float array */
                    inp[w] = ftmp * wanxr[w]; /* cur block */
                    inp[w + xoffset] = ftmp * wanxl[w]; /* overlapped Copy - next block */
                }
                inp += ow;
                inp += xoffset;
                srcp += ow;
                for (int w = 0; w < bw - ow - ow; w++)   /* center part  (non-overlapped) row of first block */
                {
                    inp[w] = float(wanyl[h] * (srcp[w] - planeBase));   /* Copy each byte from source to float array */
                }
                inp += bw - ow - ow;
                srcp += bw - ow - ow;
            }
            for (int w = 0; w < ow; w++)   /* last part (non-overlapped) of line of last block */
            {
                inp[w] = float(wanxr[w] * wanyl[h] * (srcp[w] - planeBase));   /* Copy each byte from source to float array */
            }
            inp += ow;
            srcp += ow;
            srcp += (src_pitch - coverwidth);  /* Add the pitch of one line (in bytes) to the source image. */
        }
        for (int h = oh; h < bh - oh; h++) {
            inp = inp0 + h * bw;
            for (int w = 0; w < ow; w++)   /* left part  (non-overlapped) row of first block */
            {
                inp[w] = float(wanxl[w] * (srcp[w] - planeBase));   /* Copy each byte from source to float array */
            }
            for (int w = ow; w < bw - ow; w++)   /* left part  (non-overlapped) row of first block */
            {
                inp[w] = float((srcp[w] - planeBase));   /* Copy each byte from source to float array */
            }
            inp += bw - ow;
            srcp += bw - ow;
            for (ihx = 1; ihx < nox; ihx += 1) /* middle horizontal blocks */
            {
                for (int w = 0; w < ow; w++)   /* first part (overlapped) row of block */
                {
                    ftmp = float((srcp[w] - planeBase));  /* Copy each byte from source to float array */
                    inp[w] = ftmp * wanxr[w]; /* cur block */
                    inp[w + xoffset] = ftmp * wanxl[w]; /* overlapped Copy - next block */
                }
                inp += ow;
                inp += xoffset;
                srcp += ow;
                for (int w = 0; w < bw - ow - ow; w++)   /* center part  (non-overlapped) row of first block */
                {
                    inp[w] = float((srcp[w] - planeBase));   /* Copy each byte from source to float array */
                }
                inp += bw - ow - ow;
                srcp += bw - ow - ow;
            }
            for (int w = 0; w < ow; w++)   /* last part (non-overlapped) line of last block */
            {
                inp[w] = float(wanxr[w] * (srcp[w] - planeBase));   /* Copy each byte from source to float array */
            }
            inp += ow;
            srcp += ow;

            srcp += (src_pitch - coverwidth);  /* Add the pitch of one line (in bytes) to the source image. */
        }
    }

    for (ihy = 1; ihy < noy; ihy += 1) /* middle vertical */
    {
        for (int h = 0; h < oh; h++) /* top overlapped part */
        {
            inp = inp0 + (ihy - 1) * (yoffset + (bh - oh) * bw) + (bh - oh) * bw + h * bw;
            for (int w = 0; w < ow; w++)   /* first half line of first block */
            {
                ftmp = float(wanxl[w] * (srcp[w] - planeBase));
                inp[w] = ftmp * wanyr[h];   /* Copy each byte from source to float array */
                inp[w + yoffset] = ftmp * wanyl[h];   /* y overlapped */
            }
            for (int w = ow; w < bw - ow; w++)   /* first half line of first block */
            {
                ftmp = float((srcp[w] - planeBase));
                inp[w] = ftmp * wanyr[h];   /* Copy each byte from source to float array */
                inp[w + yoffset] = ftmp * wanyl[h];   /* y overlapped */
            }
            inp += bw - ow;
            srcp += bw - ow;
            for (ihx = 1; ihx < nox; ihx++) /* middle blocks */
            {
                for (int w = 0; w < ow; w++)   /* half overlapped line of block */
                {
                    ftmp = float((srcp[w] - planeBase));   /* Copy each byte from source to float array */
                    inp[w] = ftmp * wanxr[w] * wanyr[h];
                    inp[w + xoffset] = ftmp * wanxl[w] * wanyr[h];   /* x overlapped */
                    inp[w + yoffset] = ftmp * wanxr[w] * wanyl[h];
                    inp[w + xoffset + yoffset] = ftmp * wanxl[w] * wanyl[h];   /* x overlapped */
                }
                inp += ow;
                inp += xoffset;
                srcp += ow;
                for (int w = 0; w < bw - ow - ow; w++)   /* half non-overlapped line of block */
                {
                    ftmp = float((srcp[w] - planeBase));   /* Copy each byte from source to float array */
                    inp[w] = ftmp * wanyr[h];
                    inp[w + yoffset] = ftmp * wanyl[h];
                }
                inp += bw - ow - ow;
                srcp += bw - ow - ow;
            }
            for (int w = 0; w < ow; w++)   /* last half line of last block */
            {
                ftmp = float(wanxr[w] * (srcp[w] - planeBase)); /* Copy each byte from source to float array */
                inp[w] = ftmp * wanyr[h];
                inp[w + yoffset] = ftmp * wanyl[h];
            }
            inp += ow;
            srcp += ow;

            srcp += (src_pitch - coverwidth);  /* Add the pitch of one line (in bytes) to the source image. */
        }
        /* middle  vertical nonovelapped part */
        for (int h = 0; h < bh - oh - oh; h++) {
            inp = inp0 + (ihy - 1) * (yoffset + (bh - oh) * bw) + (bh)*bw + h * bw + yoffset;
            for (int w = 0; w < ow; w++)   /* first half line of first block */
            {
                ftmp = float(wanxl[w] * (srcp[w] - planeBase));
                inp[w] = ftmp;   /* Copy each byte from source to float array */
            }
            for (int w = ow; w < bw - ow; w++)   /* first half line of first block */
            {
                ftmp = float((srcp[w] - planeBase));
                inp[w] = ftmp;   /* Copy each byte from source to float array */
            }
            inp += bw - ow;
            srcp += bw - ow;
            for (ihx = 1; ihx < nox; ihx++) /* middle blocks */
            {
                for (int w = 0; w < ow; w++)   /* half overlapped line of block */
                {
                    ftmp = float((srcp[w] - planeBase));   /* Copy each byte from source to float array */
                    inp[w] = ftmp * wanxr[w];
                    inp[w + xoffset] = ftmp * wanxl[w];   /* x overlapped */
                }
                inp += ow;
                inp += xoffset;
                srcp += ow;
                for (int w = 0; w < bw - ow - ow; w++)   /* half non-overlapped line of block */
                {
                    ftmp = float((srcp[w] - planeBase));   /* Copy each byte from source to float array */
                    inp[w] = ftmp;
                }
                inp += bw - ow - ow;
                srcp += bw - ow - ow;
            }
            for (int w = 0; w < ow; w++)   /* last half line of last block */
            {
                ftmp = float(wanxr[w] * (srcp[w] - planeBase)); /* Copy each byte from source to float array */
                inp[w] = ftmp;
            }
            inp += ow;
            srcp += ow;

            srcp += (src_pitch - coverwidth);  /* Add the pitch of one line (in bytes) to the source image. */
        }

    }

    ihy = noy; /* last bottom  part */
    {
        for (int h = 0; h < oh; h++) {
            inp = inp0 + (ihy - 1) * (yoffset + (bh - oh) * bw) + (bh - oh) * bw + h * bw;
            for (int w = 0; w < ow; w++)   /* first half line of first block */
            {
                ftmp = float(wanxl[w] * wanyr[h] * (srcp[w] - planeBase));
                inp[w] = ftmp;   /* Copy each byte from source to float array */
            }
            for (int w = ow; w < bw - ow; w++)   /* first half line of first block */
            {
                ftmp = float(wanyr[h] * (srcp[w] - planeBase));
                inp[w] = ftmp;   /* Copy each byte from source to float array */
            }
            inp += bw - ow;
            srcp += bw - ow;
            for (ihx = 1; ihx < nox; ihx++) /* middle blocks */
            {
                for (int w = 0; w < ow; w++)   /* half line of block */
                {
                    float ftmp = float(wanyr[h] * (srcp[w] - planeBase));   /* Copy each byte from source to float array */
                    inp[w] = ftmp * wanxr[w];
                    inp[w + xoffset] = ftmp * wanxl[w];   /* overlapped Copy */
                }
                inp += ow;
                inp += xoffset;
                srcp += ow;
                for (int w = 0; w < bw - ow - ow; w++)   /* center part  (non-overlapped) row of first block */
                {
                    inp[w] = float(wanyr[h] * (srcp[w] - planeBase));   /* Copy each byte from source to float array */
                }
                inp += bw - ow - ow;
                srcp += bw - ow - ow;
            }
            for (int w = 0; w < ow; w++)   /* last half line of last block */
            {
                ftmp = float(wanxr[w] * wanyr[h] * (srcp[w] - planeBase));
                inp[w] = ftmp;   /* Copy each byte from source to float array */
            }
            inp += ow;
            srcp += ow;

            srcp += (src_pitch - coverwidth);  /* Add the pitch of one line (in bytes) to the source image. */
        }

    }
}


template<typename T>
static void DecodeOverlapPlane(const float *__restrict inp0, float norm, T *__restrict dstp0, ptrdiff_t dst_pitch, float *__restrict wsynxl, float *__restrict wsynxr, float *__restrict wsynyr, float *__restrict wsynyl, int bw, int bh, int ow, int oh, int nox, int noy, int coverwidth, int planeBase, int maxval) {
    int w, h;
    int ihx, ihy;
    T *__restrict dstp = dstp0;
    const float *__restrict inp = inp0;
    int xoffset = bh * bw - (bw - ow);
    int yoffset = bw * nox * bh - bw * (bh - oh); /* vertical offset of same block (overlap) */
    dst_pitch /= sizeof(T);

    ihy = 0; /* first top big non-overlapped) part */
    {
        for (h = 0; h < bh - oh; h++) {
            inp = inp0 + h * bw;
            for (w = 0; w < bw - ow; w++)   /* first half line of first block */
            {
                if constexpr (std::is_integral_v<T>)
                    dstp[w] = std::min(maxval, std::max(0, (int)(inp[w] * norm) + planeBase)); /* Copy each byte from float array to dest with windows */
                else
                    dstp[w] = inp[w] * norm;
            }
            inp += bw - ow;
            dstp += bw - ow;
            for (ihx = 1; ihx < nox; ihx++) /* middle horizontal half-blocks */
            {
                for (w = 0; w < ow; w++)   /* half line of block */
                {
                    if constexpr (std::is_integral_v<T>)
                        dstp[w] = std::min(maxval, std::max(0, (int)((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w]) * norm) + planeBase));  /* overlapped Copy */
                    else
                        dstp[w] = (inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w]) * norm;
                }
                inp += xoffset + ow;
                dstp += ow;
                for (w = 0; w < bw - ow - ow; w++)   /* first half line of first block */
                {
                    if constexpr (std::is_integral_v<T>)
                        dstp[w] = std::min(maxval, std::max(0, (int)(inp[w] * norm) + planeBase));   /* Copy each byte from float array to dest with windows */
                    else
                        dstp[w] = inp[w] * norm;
                }
                inp += bw - ow - ow;
                dstp += bw - ow - ow;
            }
            for (w = 0; w < ow; w++)   /* last half line of last block */
            {
                if constexpr (std::is_integral_v<T>)
                    dstp[w] = std::min(maxval, std::max(0, (int)(inp[w] * norm) + planeBase));
                else
                    dstp[w] = inp[w] * norm;
            }
            inp += ow;
            dstp += ow;

            dstp += (dst_pitch - coverwidth);  /* Add the pitch of one line (in bytes) to the dest image. */
        }
    }

    for (ihy = 1; ihy < noy; ihy += 1) /* middle vertical */
    {
        for (h = 0; h < oh; h++) /* top overlapped part */
        {
            inp = inp0 + (ihy - 1) * (yoffset + (bh - oh) * bw) + (bh - oh) * bw + h * bw;

            float wsynyrh = wsynyr[h] * norm; /* remove from cycle for speed */
            float wsynylh = wsynyl[h] * norm;

            for (w = 0; w < bw - ow; w++)   /* first half line of first block */
            {
                if constexpr (std::is_integral_v<T>)
                    dstp[w] = std::min(maxval, std::max(0, (int)((inp[w] * wsynyrh + inp[w + yoffset] * wsynylh)) + planeBase));   /* y overlapped */
                else
                    dstp[w] = inp[w] * wsynyrh + inp[w + yoffset] * wsynylh;
            }
            inp += bw - ow;
            dstp += bw - ow;
            for (ihx = 1; ihx < nox; ihx++) /* middle blocks */
            {
                for (w = 0; w < ow; w++)   /* half overlapped line of block */
                {
                    if constexpr (std::is_integral_v<T>)
                        dstp[w] = std::min(maxval, std::max(0, (int)(((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w]) * wsynyrh
                            + (inp[w + yoffset] * wsynxr[w] + inp[w + xoffset + yoffset] * wsynxl[w]) * wsynylh)) + planeBase));   /* x overlapped */
                    else
                        dstp[w] = (inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w]) * wsynyrh
                        + (inp[w + yoffset] * wsynxr[w] + inp[w + xoffset + yoffset] * wsynxl[w]) * wsynylh;
                }
                inp += xoffset + ow;
                dstp += ow;
                for (w = 0; w < bw - ow - ow; w++)   /* double minus - half non-overlapped line of block */
                {
                    if constexpr (std::is_integral_v<T>)
                        dstp[w] = std::min(maxval, std::max(0, (int)((inp[w] * wsynyrh + inp[w + yoffset] * wsynylh)) + planeBase));
                    else
                        dstp[w] = inp[w] * wsynyrh + inp[w + yoffset] * wsynylh;
                }
                inp += bw - ow - ow;
                dstp += bw - ow - ow;
            }
            for (w = 0; w < ow; w++)   /* last half line of last block */
            {
                if constexpr (std::is_integral_v<T>)
                    dstp[w] = std::min(maxval, std::max(0, (int)((inp[w] * wsynyrh + inp[w + yoffset] * wsynylh)) + planeBase));
                else
                    dstp[w] = inp[w] * wsynyrh + inp[w + yoffset] * wsynylh;
            }
            inp += ow;
            dstp += ow;

            dstp += (dst_pitch - coverwidth);  /* Add the pitch of one line (in bytes) to the source image. */
        }
        /* middle  vertical non-ovelapped part */
        for (h = 0; h < (bh - oh - oh); h++) {
            inp = inp0 + (ihy - 1) * (yoffset + (bh - oh) * bw) + (bh)*bw + h * bw + yoffset;
            for (w = 0; w < bw - ow; w++)   /* first half line of first block */
            {
                if constexpr (std::is_integral_v<T>)
                    dstp[w] = std::min(maxval, std::max(0, (int)(inp[w] * norm) + planeBase));
                else
                    dstp[w] = inp[w] * norm;
            }
            inp += bw - ow;
            dstp += bw - ow;
            for (ihx = 1; ihx < nox; ihx++) /* middle blocks */
            {
                for (w = 0; w < ow; w++)   /* half overlapped line of block */
                {
                    if constexpr (std::is_integral_v<T>)
                        dstp[w] = std::min(maxval, std::max(0, (int)((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w]) * norm) + planeBase));   /* x overlapped */
                    else
                        dstp[w] = (inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w]) * norm;
                }
                inp += xoffset + ow;
                dstp += ow;
                for (w = 0; w < bw - ow - ow; w++)   /* half non-overlapped line of block */
                {
                    if constexpr (std::is_integral_v<T>)
                        dstp[w] = std::min(maxval, std::max(0, (int)(inp[w] * norm) + planeBase));
                    else
                        dstp[w] = inp[w] * norm;
                }
                inp += bw - ow - ow;
                dstp += bw - ow - ow;
            }
            for (w = 0; w < ow; w++)   /* last half line of last block */
            {
                if constexpr (std::is_integral_v<T>)
                    dstp[w] = std::min(maxval, std::max(0, (int)(inp[w] * norm) + planeBase));
                else
                    dstp[w] = inp[w] * norm;
            }
            inp += ow;
            dstp += ow;

            dstp += (dst_pitch - coverwidth);  /* Add the pitch of one line (in bytes) to the source image. */
        }
    }

    ihy = noy; /* last bottom part */
    {
        for (h = 0; h < oh; h++) {
            inp = inp0 + (ihy - 1) * (yoffset + (bh - oh) * bw) + (bh - oh) * bw + h * bw;
            for (w = 0; w < bw - ow; w++)   /* first half line of first block */
            {
                if constexpr (std::is_integral_v<T>)
                    dstp[w] = std::min(maxval, std::max(0, (int)(inp[w] * norm) + planeBase));
                else
                    dstp[w] = inp[w] * norm;
            }
            inp += bw - ow;
            dstp += bw - ow;
            for (ihx = 1; ihx < nox; ihx++) /* middle blocks */
            {
                for (w = 0; w < ow; w++)   /* half line of block */
                {
                    if constexpr (std::is_integral_v<T>)
                        dstp[w] = std::min(maxval, std::max(0, (int)((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w]) * norm) + planeBase));  /* overlapped Copy */
                    else
                        dstp[w] = (inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w]) * norm;
                }
                inp += xoffset + ow;
                dstp += ow;
                for (w = 0; w < bw - ow - ow; w++)   /* half line of block */
                {
                    if constexpr (std::is_integral_v<T>)
                        dstp[w] = std::min(maxval, std::max(0, (int)(inp[w] * norm) + planeBase));
                    else
                        dstp[w] = inp[w] * norm;
                }
                inp += bw - ow - ow;
                dstp += bw - ow - ow;
            }
            for (w = 0; w < ow; w++)   /* last half line of last block */
            {
                if constexpr (std::is_integral_v<T>)
                    dstp[w] = std::min(maxval, std::max(0, (int)(inp[w] * norm) + planeBase));
                else
                    dstp[w] = inp[w] * norm;
            }
            inp += ow;
            dstp += ow;

            dstp += (dst_pitch - coverwidth);  /* Add the pitch of one line (in bytes) to the source image. */
        }
    }
}

FFT3DFilterTransform::FFT3DFilterTransform(bool pshow, VSNodeRef *node_, int plane_, int wintype, int bw_, int bh_, int ow_, int oh_, int px_, int py_, float pcutoff_, float degrid_, bool interlaced_, bool measure, int ncpu, VSCore *core, const VSAPI *vsapi) : node(node_), plane(plane_), bw(bw_), bh(bh_), ow(ow_), oh(oh_), px(px_), py(py_), pcutoff(pcutoff_), degrid(degrid_), interlaced(interlaced_), in(nullptr, nullptr), plan(nullptr, nullptr) {
    if (ow < 0)
        ow = bw / 3;
    if (oh < 0)
        oh = bh / 3;

    const VSVideoInfo *srcvi = vsapi->getVideoInfo(node);

    planeBase = (plane > 0 && srcvi->format.sampleType == stInteger && srcvi->format.colorFamily == cfYUV) ? (1 << (srcvi->format.bitsPerSample - 1)) : 0;

    nox = ((srcvi->width >> (plane ? srcvi->format.subSamplingW : 0)) - ow + (bw - ow - 1)) / (bw - ow);
    noy = ((srcvi->height >> (plane ? srcvi->format.subSamplingH : 0)) - oh + (bh - oh - 1)) / (bh - oh);

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
    coverpitch = ((coverwidth + 7) / 8) * 8 * srcvi->format.bytesPerSample;
    coverbuf = std::unique_ptr<uint8_t[]>(new uint8_t[coverheight * coverpitch]);

    int insize = bw * bh * nox * noy;
    in = std::unique_ptr<float[], decltype(&fftw_free)>(fftwf_alloc_real(insize), fftwf_free);
    outwidth = bw / 2 + 1;                  /* width (pitch) of complex fft block */
    outpitchelems = ((outwidth + 1) / 2) * 2;
    int outsize = outpitchelems * bh * nox * noy;   /* replace outwidth to outpitchelems here and below in v1.7 */

    int planFlags = (measure ? FFTW_MEASURE : FFTW_ESTIMATE) | FFTW_DESTROY_INPUT;
    int ndim[2] = { bh, bw }; 
    int idist = bw * bh;
    int odist = outpitchelems * bh;
    int inembed[2] = { bh, bw };
    int onembed[2] = { bh, outpitchelems };
    int howmanyblocks = nox * noy;

    dstvi = *srcvi;
    vsapi->getVideoFormatByID(&dstvi.format, pfGrayS, core);
    dstvi.width = outsize * 2; // 2 floats per complex number
    dstvi.height = 1;

    VSFrameRef *out = vsapi->newVideoFrame(&dstvi.format, dstvi.width, dstvi.height, nullptr, core);

    fftwf_plan_with_nthreads(ncpu);

    plan = std::unique_ptr<fftwf_plan_s, decltype(&fftwf_destroy_plan)>(fftwf_plan_many_dft_r2c(2, ndim, howmanyblocks,
        in.get(), inembed, 1, idist, reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(out, 0)), onembed, 1, odist, planFlags), fftwf_destroy_plan);

    fftwf_plan_with_nthreads(1);

    vsapi->freeFrame(out);

    // reset output format since it's only passed through
    if (pshow)
        outvi = *srcvi;
    else
        outvi = dstvi;
}

VSFrameRef *FFT3DFilterTransform::GetFrame(const VSFrameRef *src, VSCore *core, const VSAPI *vsapi) {
    const VSVideoFormat *fi = vsapi->getVideoFrameFormat(src);

    if (fi->bytesPerSample == 1) {
        FramePlaneToCoverbuf<uint8_t>(plane, src, reinterpret_cast<uint8_t *>(coverbuf.get()), coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, vsapi);
        InitOverlapPlane<uint8_t>(in.get(), reinterpret_cast<uint8_t *>(coverbuf.get()), coverpitch, wanxl.get(), wanxr.get(), wanyl.get(), wanyr.get(), bw, bh, ow, oh, nox, noy, coverwidth, planeBase);
    } else if (fi->bytesPerSample == 2) {
        FramePlaneToCoverbuf<uint16_t>(plane, src, reinterpret_cast<uint16_t *>(coverbuf.get()), coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, vsapi);
        InitOverlapPlane<uint16_t>(in.get(), reinterpret_cast<uint16_t *>(coverbuf.get()), coverpitch, wanxl.get(), wanxr.get(), wanyl.get(), wanyr.get(), bw, bh, ow, oh, nox, noy, coverwidth, planeBase);
    } else if (fi->bytesPerSample == 4) {
        FramePlaneToCoverbuf<float>(plane, src, reinterpret_cast<float *>(coverbuf.get()), coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, vsapi);
        InitOverlapPlane<float>(in.get(), reinterpret_cast<float *>(coverbuf.get()), coverpitch, wanxl.get(), wanxr.get(), wanyl.get(), wanyr.get(), bw, bh, ow, oh, nox, noy, coverwidth, planeBase);
    }

    VSFrameRef *dst = vsapi->newVideoFrame(&dstvi.format, dstvi.width, dstvi.height, src, core);
    fftwf_execute_dft_r2c(plan.get(), in.get(), reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)));
    return dst;
}

const VSFrameRef *VS_CC FFT3DFilterTransform::GetFrame(int n, int activation_reason, void *instance_data, void **frame_data, VSFrameContext *frame_ctx, VSCore *core, const VSAPI *vsapi) {
    FFT3DFilterTransform *data = reinterpret_cast<FFT3DFilterTransform *>(instance_data);
    if (activation_reason == arInitial) {
        vsapi->requestFrameFilter(n, data->node, frame_ctx);
    } else if (activation_reason == arAllFramesReady) {
        const VSFrameRef *src = vsapi->getFrameFilter(n, data->node, frame_ctx);
        VSFrameRef *dst = data->GetFrame(src, core, vsapi);
        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

const VSFrameRef *VS_CC FFT3DFilterTransform::GetPShowFrame(int n, int activation_reason, void *instance_data, void **frame_data, VSFrameContext *frame_ctx, VSCore *core, const VSAPI *vsapi) {
    FFT3DFilterTransform *data = reinterpret_cast<FFT3DFilterTransform *>(instance_data);
    if (activation_reason == arInitial) {
        vsapi->requestFrameFilter(n, data->node, frame_ctx);
    } else if (activation_reason == arAllFramesReady) {
        const VSFrameRef *src = vsapi->getFrameFilter(n, data->node, frame_ctx);
        VSFrameRef *dst = data->GetPShowInfo(src, core, vsapi);
        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

const VSFrameRef *FFT3DFilterTransform::GetGridSample(VSCore *core, const VSAPI *vsapi) {
    const VSVideoInfo *vi = vsapi->getVideoInfo(node);
    int bytesPerSample = vi->format.bytesPerSample;

    if (bytesPerSample == 1) {
        memset(coverbuf.get(), 255, coverheight * coverpitch);
        InitOverlapPlane(in.get(), reinterpret_cast<uint8_t *>(coverbuf.get()), coverpitch, wanxl.get(), wanxr.get(), wanyl.get(), wanyr.get(), bw, bh, ow, oh, nox, noy, coverwidth, 0);
    } else if (bytesPerSample == 2) {
        int maxval = (1 << vi->format.bitsPerSample) - 1;
        fft3d_memset(reinterpret_cast<uint16_t *>(coverbuf.get()), static_cast<uint16_t>(maxval), coverheight * coverpitch / 2);
        InitOverlapPlane(in.get(), reinterpret_cast<uint16_t *>(coverbuf.get()), coverpitch, wanxl.get(), wanxr.get(), wanyl.get(), wanyr.get(), bw, bh, ow, oh, nox, noy, coverwidth, 0);
    } else if (bytesPerSample == 4) {
        fft3d_memset(reinterpret_cast<float *>(coverbuf.get()), 1.f, coverheight * coverpitch / 4);
        InitOverlapPlane(in.get(), reinterpret_cast<float *>(coverbuf.get()), coverpitch, wanxl.get(), wanxr.get(), wanyl.get(), wanyr.get(), bw, bh, ow, oh, nox, noy, coverwidth, 0);
    }

    VSFrameRef *dst = vsapi->newVideoFrame(&dstvi.format, dstvi.width, dstvi.height, nullptr, core);
    fftwf_execute_dft_r2c(plan.get(), in.get(), reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(dst, 0)));
    return dst;
}

//-------------------------------------------------------------------------------------------
static void FindPatternBlock(const fftwf_complex *outcur0, int outwidth, int outpitchelems, int bh, int nox, int noy, int &px, int &py, const float *pwin, float degrid, const fftwf_complex *gridsample) {
    /* since v1.7 outwidth must be really an outpitchelems */
    float sigmaSquared = 1e15f;

    for (int by = 2; by < noy - 2; by++) {
        for (int bx = 2; bx < nox - 2; bx++) {
            const fftwf_complex *outcur = outcur0 + nox * by * bh * outpitchelems + bx * bh * outpitchelems;
            float sigmaSquaredcur = 0;
            float gcur = degrid * outcur[0][0] / gridsample[0][0]; /* grid (windowing) correction factor */
            for (int h = 0; h < bh; h++) {
                for (int w = 0; w < outwidth; w++) {
                    float grid0 = gcur * gridsample[w][0];
                    float grid1 = gcur * gridsample[w][1];
                    float corrected0 = outcur[w][0] - grid0;
                    float corrected1 = outcur[w][1] - grid1;
                    float psd = corrected0 * corrected0 + corrected1 * corrected1;
                    sigmaSquaredcur += psd * pwin[w]; /* windowing */
                }
                outcur += outpitchelems;
                pwin += outpitchelems;
                gridsample += outpitchelems;
            }
            pwin -= outpitchelems * bh; /* restore */
            if (sigmaSquaredcur < sigmaSquared) {
                px = bx;
                py = by;
                sigmaSquared = sigmaSquaredcur;
            }
        }
    }
}
//-------------------------------------------------------------------------------------------

static void SetPattern(const fftwf_complex *outcur, int outwidth, int outpitchelems, int bh, int nox, int noy, int px, int py, const float *pwin, float *pattern2d, float &psigma, float degrid, const fftwf_complex *gridsample) {
    outcur += nox * py * bh * outpitchelems + px * bh * outpitchelems;
    float sigmaSquared = 0;
    float weight = 0;

    for (int h = 0; h < bh; h++) {
        for (int w = 0; w < outwidth; w++) {
            weight += pwin[w];
        }
        pwin += outpitchelems;
    }
    pwin -= outpitchelems * bh; /* restore */

    float gcur = degrid * outcur[0][0] / gridsample[0][0]; /* grid (windowing) correction factor */

    for (int h = 0; h < bh; h++) {
        for (int w = 0; w < outwidth; w++) {
            float grid0 = gcur * gridsample[w][0];
            float grid1 = gcur * gridsample[w][1];
            float corrected0 = outcur[w][0] - grid0;
            float corrected1 = outcur[w][1] - grid1;
            float psd = corrected0 * corrected0 + corrected1 * corrected1;
            pattern2d[w] = psd * pwin[w]; /* windowing */
            sigmaSquared += pattern2d[w]; /* sum */
        }
        outcur += outpitchelems;
        pattern2d += outpitchelems;
        pwin += outpitchelems;
        gridsample += outpitchelems;
    }
    psigma = sqrt(sigmaSquared / (weight * bh * outwidth)); /* mean std deviation (sigma) */
}

void FFT3DFilterTransform::GetNoisePattern(int n, int &px, int &py, float *pattern2d, float &psigma, const fftwf_complex *gridsample, VSCore *core, const VSAPI *vsapi) {
    const VSFrameRef *src = vsapi->getFrame(n, node, nullptr, 0);
    VSFrameRef *dst = GetFrame(src, core, vsapi);
    vsapi->freeFrame(src);

    std::unique_ptr<float[]> pwin = std::unique_ptr<float[]>(new float[bh * outpitchelems]); /* pattern window array */
    GetPatternWindow(bw, bh, outwidth, outpitchelems, pcutoff, pwin.get());

    if (px == 0 && py == 0) /* try find pattern block with minimal noise sigma */
        FindPatternBlock(reinterpret_cast<const fftwf_complex *>(vsapi->getReadPtr(dst, 0)), outwidth, outpitchelems, bh, nox, noy, px, py, pwin.get(), degrid, gridsample);
    SetPattern(reinterpret_cast<const fftwf_complex *>(vsapi->getReadPtr(dst, 0)), outwidth, outpitchelems, bh, nox, noy, px, py, pwin.get(), pattern2d, psigma, degrid, gridsample);
}

VSFrameRef *FFT3DFilterTransform::GetPShowInfo(const VSFrameRef *src, VSCore *core, const VSAPI *vsapi) {
    // accept a lot of extra recalculation and allocation when visualizing
    // should be fast enough since it's all spatial anyway
    // requires px, py, pcutoff, degrid

    VSFrameRef *transformed = GetFrame(src, core, vsapi);
    const VSFrameRef *gridsample = GetGridSample(core, vsapi);

    std::unique_ptr<float[]> pwin = std::unique_ptr<float[]>(new float[bh * outpitchelems]); /* pattern window array */
    GetPatternWindow(bw, bh, outwidth, outpitchelems, pcutoff, pwin.get());

    int pxf, pyf;
    if (px == 0 && py == 0) { /* try find pattern block with minimal noise sigma */
        FindPatternBlock(reinterpret_cast<const fftwf_complex *>(vsapi->getReadPtr(transformed, 0)), outwidth, outpitchelems, bh, nox, noy, pxf, pyf, pwin.get(), degrid, reinterpret_cast<const fftwf_complex *>(vsapi->getReadPtr(gridsample, 0)));
    } else {
        pxf = px;
        pyf = py;
    }

    float psigma;
    std::unique_ptr<float[], decltype(&fftw_free)> pattern2d = std::unique_ptr<float[], decltype(&fftw_free)>(fftwf_alloc_real(bh * outpitchelems), fftwf_free);
    SetPattern(reinterpret_cast<const fftwf_complex *>(vsapi->getReadPtr(transformed, 0)), outwidth, outpitchelems, bh, nox, noy, pxf, pyf, pwin.get(), pattern2d.get(), psigma, degrid, reinterpret_cast<const fftwf_complex *>(vsapi->getReadPtr(gridsample, 0)));

    vsapi->freeFrame(transformed);
    vsapi->freeFrame(gridsample);

    VSFrameRef *dst = vsapi->copyFrame(src, core);

    VSMap *props = vsapi->getFramePropertiesRW(dst);

    vsapi->mapSetInt(props, "px", pxf, paReplace);
    vsapi->mapSetInt(props, "py", pyf, paReplace);
    vsapi->mapSetFloat(props, "sigma", psigma, paReplace);

    return dst;
}

void VS_CC FFT3DFilterTransform::Free(void *instance_data, VSCore *core, const VSAPI *vsapi) {
    FFT3DFilterTransform *data = reinterpret_cast<FFT3DFilterTransform *>(instance_data);
    vsapi->freeNode(data->node);
    delete data;
}

//-----------------------------------------------------------------------------------------

FFT3DFilterInvTransform::FFT3DFilterInvTransform(VSNodeRef *node_, const VSVideoInfo *srcvi, int plane, int wintype, int bw_, int bh_, int ow_, int oh_, bool interlaced_, bool measure, int ncpu, VSCore *core, const VSAPI *vsapi) : node(node_), bw(bw_), bh(bh_), ow(ow_), oh(oh_), interlaced(interlaced_), in(nullptr, nullptr), planinv(nullptr, nullptr) {
    if (ow < 0)
        ow = bw / 3;
    if (oh < 0)
        oh = bh / 3;

    planeBase = (plane > 0 && srcvi->format.sampleType == stInteger && srcvi->format.colorFamily == cfYUV) ? (1 << (srcvi->format.bitsPerSample - 1)) : 0;

    nox = ((srcvi->width >> (plane ? srcvi->format.subSamplingW : 0)) - ow + (bw - ow - 1)) / (bw - ow);
    noy = ((srcvi->height >> (plane ? srcvi->format.subSamplingH : 0)) - oh + (bh - oh - 1)) / (bh - oh);

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
    coverpitch = ((coverwidth + 7) / 8) * 8 * srcvi->format.bytesPerSample;
    coverbuf = std::unique_ptr<uint8_t[]>(new uint8_t[coverheight * coverpitch]);

    int insize = bw * bh * nox * noy;
    in = std::unique_ptr<float[], decltype(&fftw_free)>(fftwf_alloc_real(insize), fftwf_free);
    outwidth = bw / 2 + 1;                  /* width (pitch) of complex fft block */
    outpitchelems = ((outwidth + 1) / 2) * 2;
    norm = 1.0f / (bw * bh); /* do not forget set FFT normalization factor */

    // FFTW_PRESERVE_INPUT would be preferred but it's not implemented due to the infinite greatness of FFTW
    int planFlags = (measure ? FFTW_MEASURE : FFTW_ESTIMATE) | FFTW_DESTROY_INPUT; // needed since a read only frame is the source
    int ndim[2] = { bh, bw };
    int idist = bw * bh;
    int odist = outpitchelems * bh;
    int inembed[2] = { bh, bw };
    int onembed[2] = { bh, outpitchelems };
    int howmanyblocks = nox * noy;

    dstvi = *srcvi;
    dstvi.width = (srcvi->width >> (plane ? srcvi->format.subSamplingW : 0));
    dstvi.height = (srcvi->height >> (plane ? srcvi->format.subSamplingH : 0));
    vsapi->queryVideoFormat(&dstvi.format, cfGray, srcvi->format.sampleType, srcvi->format.bitsPerSample, 0, 0, core);

    const VSVideoInfo *inputvi = vsapi->getVideoInfo(node);
    VSFrameRef *src = vsapi->newVideoFrame(&inputvi->format, inputvi->width, inputvi->height, nullptr, core);

    fftwf_plan_with_nthreads(ncpu);

    planinv = std::unique_ptr<fftwf_plan_s, decltype(&fftwf_destroy_plan)>(fftwf_plan_many_dft_c2r(2, ndim, howmanyblocks,
        reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(src, 0)), onembed, 1, odist, in.get(), inembed, 1, idist, planFlags), fftwf_destroy_plan);

    fftwf_plan_with_nthreads(1);

    vsapi->freeFrame(src);
}

VSFrameRef *FFT3DFilterInvTransform::GetFrame(const VSFrameRef *src, VSCore *core, const VSAPI *vsapi) {
    VSFrameRef *modifiableSrc = vsapi->copyFrame(src, core);

    fftwf_execute_dft_c2r(planinv.get(), reinterpret_cast<fftwf_complex *>(vsapi->getWritePtr(modifiableSrc, 0)), in.get());

    vsapi->freeFrame(modifiableSrc);

    VSFrameRef *dst = vsapi->newVideoFrame(&dstvi.format, dstvi.width, dstvi.height, src, core);

    if (dstvi.format.bytesPerSample == 1) {
        DecodeOverlapPlane(in.get(), norm, reinterpret_cast<uint8_t *>(coverbuf.get()), coverpitch, wsynxl.get(), wsynxr.get(), wsynyr.get(), wsynyl.get(), bw, bh, ow, oh, nox, noy, coverwidth, planeBase, 255);
        CoverbufToFramePlane(reinterpret_cast<uint8_t *>(coverbuf.get()), coverwidth, coverheight, coverpitch, dst, mirw, mirh, interlaced, vsapi);
    } else if (dstvi.format.bytesPerSample == 2) {
        DecodeOverlapPlane(in.get(), norm, reinterpret_cast<uint16_t *>(coverbuf.get()), coverpitch, wsynxl.get(), wsynxr.get(), wsynyr.get(), wsynyl.get(), bw, bh, ow, oh, nox, noy, coverwidth, planeBase, (1 << dstvi.format.bitsPerSample) - 1);
        CoverbufToFramePlane(reinterpret_cast<uint16_t *>(coverbuf.get()), coverwidth, coverheight, coverpitch, dst, mirw, mirh, interlaced, vsapi);
    } else if (dstvi.format.bytesPerSample == 4) {
        DecodeOverlapPlane(in.get(), norm, reinterpret_cast<float *>(coverbuf.get()), coverpitch, wsynxl.get(), wsynxr.get(), wsynyr.get(), wsynyl.get(), bw, bh, ow, oh, nox, noy, coverwidth, planeBase, 1);
        CoverbufToFramePlane(reinterpret_cast<float *>(coverbuf.get()), coverwidth, coverheight, coverpitch, dst, mirw, mirh, interlaced, vsapi);
    }

    return dst;
}

const VSFrameRef *VS_CC FFT3DFilterInvTransform::GetFrame(int n, int activation_reason, void *instance_data, void **frame_data, VSFrameContext *frame_ctx, VSCore *core, const VSAPI *vsapi) {
    FFT3DFilterInvTransform *data = reinterpret_cast<FFT3DFilterInvTransform *>(instance_data);
    if (activation_reason == arInitial) {
        vsapi->requestFrameFilter(n, data->node, frame_ctx);
    } else if (activation_reason == arAllFramesReady) {
        const VSFrameRef *src = vsapi->getFrameFilter(n, data->node, frame_ctx);
        VSFrameRef *dst = data->GetFrame(src, core, vsapi);
        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

void VS_CC FFT3DFilterInvTransform::Free(void *instance_data, VSCore *core, const VSAPI *vsapi) {
    FFT3DFilterInvTransform *data = reinterpret_cast<FFT3DFilterInvTransform *>(instance_data);
    vsapi->freeNode(data->node);
    delete data;
}

FFT3DFilterPShow::FFT3DFilterPShow(VSNodeRef *node_, int plane_, int bw_, int bh_, int ow_, int oh_, bool interlaced_, VSCore *core, const VSAPI *vsapi) : node(node_), plane(plane_), bw(bw_), bh(bh_), ow(ow_), oh(oh_) {
    vi = vsapi->getVideoInfo(node);
}

template<typename T>
static void PutPatternOnly2(const T *src, T *dst, T emptyval, ptrdiff_t stride, int height, int bw, int bh, int ow, int oh, int px, int py) {
    stride /= sizeof(T);
    fft3d_memset<T>(dst, emptyval, stride * height);

    ptrdiff_t hoffset = stride * (bh - oh) * (py - 1);
    src += hoffset;
    dst += hoffset;
    ptrdiff_t woffset = (bw - ow) * (px - 1);


    for (int h = 0; h < bh; h++) {
        memcpy(dst + woffset, src + woffset, bw * sizeof(T));
        src += stride;
        dst += stride;
    }
}

VSFrameRef *FFT3DFilterPShow::GetFrame(const VSFrameRef *src, VSCore *core, const VSAPI *vsapi) {
    const VSMap *props = vsapi->getFramePropertiesRO(src);

    int pxf = vsapi->mapGetIntSaturated(props, "px", 0, nullptr);
    int pyf = vsapi->mapGetIntSaturated(props, "py", 0, nullptr);

    const VSFrameRef *srcs[3] = { plane == 0 ? nullptr : src, plane == 1 ? nullptr : src, plane == 2 ? nullptr : src };
    const int planesrc[3] = { 0, 1, 2 };

    VSFrameRef *dst = vsapi->newVideoFrame2(&vi->format, vi->width, vi->height, srcs, planesrc, src, core);

    int planeBase = (plane > 0 && vi->format.sampleType == stInteger && vi->format.colorFamily == cfYUV) ? (1 << (vi->format.bitsPerSample - 1)) : 0;

    if (vi->format.bytesPerSample == 1)
        PutPatternOnly2<uint8_t>(vsapi->getReadPtr(src, plane), vsapi->getWritePtr(dst, plane), 128, vsapi->getStride(src, plane), vsapi->getFrameHeight(src, plane), bw, bh, ow, oh, pxf, pyf);
    else if (vi->format.bytesPerSample == 2)
        PutPatternOnly2<uint16_t>(reinterpret_cast<const uint16_t *>(vsapi->getReadPtr(src, plane)), reinterpret_cast<uint16_t *>(vsapi->getWritePtr(dst, plane)), planeBase, vsapi->getStride(src, plane), vsapi->getFrameHeight(src, plane), bw, bh, ow, oh, pxf, pyf);
    else if (vi->format.bytesPerSample == 4)
        PutPatternOnly2<float>(reinterpret_cast<const float *>(vsapi->getReadPtr(src, plane)), reinterpret_cast<float *>(vsapi->getWritePtr(dst, plane)), 0, vsapi->getStride(src, plane), vsapi->getFrameHeight(src, plane), bw, bh, ow, oh, pxf, pyf);

    return dst;
}

const VSFrameRef *VS_CC FFT3DFilterPShow::GetFrame(int n, int activation_reason, void *instance_data, void **frame_data, VSFrameContext *frame_ctx, VSCore *core, const VSAPI *vsapi) {
    FFT3DFilterPShow *data = reinterpret_cast<FFT3DFilterPShow *>(instance_data);
    if (activation_reason == arInitial) {
        vsapi->requestFrameFilter(n, data->node, frame_ctx);
    } else if (activation_reason == arAllFramesReady) {
        const VSFrameRef *src = vsapi->getFrameFilter(n, data->node, frame_ctx);
        VSFrameRef *dst = data->GetFrame(src, core, vsapi);
        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

void VS_CC FFT3DFilterPShow::Free(void *instance_data, VSCore *core, const VSAPI *vsapi) {
    FFT3DFilterPShow *data = reinterpret_cast<FFT3DFilterPShow *>(instance_data);
    vsapi->freeNode(data->node);
    delete data;
}