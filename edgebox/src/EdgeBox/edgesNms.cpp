/*
 * edgesNms.cpp
 *
 *  Created on: May 16, 2016
 *      Author: patcho
 */

/*******************************************************************************
* Structured Edge Detection Toolbox      Version 3.01
* Code written by Piotr Dollar, 2014.
* Licensed under the MSR-LA Full Rights License [see license.txt]
*******************************************************************************/
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace EdgeBox
{

// return I[x,y] via bilinear interpolation
inline float interp( float *I, int h, int w, float x, float y ) {
    x = x < 0 ? 0 : (x > w - 1.001 ? w - 1.001 : x);
    y = y < 0 ? 0 : (y > h - 1.001 ? h - 1.001 : y);
    int x0 = int(x), y0 = int(y), x1 = x0 + 1, y1 = y0 + 1;
    float dx0 = x - x0, dy0 = y - y0, dx1 = 1 - dx0, dy1 = 1 - dy0;
    return I[x0 * h + y0] * dx1 * dy1 + I[x1 * h + y0] * dx0 * dy1 +
                 I[x0 * h + y1] * dx1 * dy0 + I[x1 * h + y1] * dx0 * dy0;
}

void edgesNms( float* edgeSrc, float * edgeDst, float* orient, int r, int s, float m, int h, int w, int nThreads )
{
    // suppress edges where edgeDst is stronger in orthogonal direction
#ifdef _OPENMP
    nThreads = nThreads < omp_get_max_threads() ? nThreads : omp_get_max_threads();
    #pragma omp parallel for num_threads(nThreads)
#endif
    for ( int x = 0; x < w; x++ ) for ( int y = 0; y < h; y++ ) {
            float e = edgeDst[x * h + y] = edgeSrc[x * h + y]; if (!e) continue; e *= m;
            float coso = cos(orient[x * h + y]), sino = sin(orient[x * h + y]);
            for ( int d = -r; d <= r; d++ ) if ( d ) {
                    float e0 = interp(edgeSrc, h, w, x + d * coso, y + d * sino);
                    if (e < e0) { edgeDst[x * h + y] = 0; break; }
                }
        }

    // suppress noisy edgeDst estimates near boundaries
    s = s > w / 2 ? w / 2 : s; s = s > h / 2 ? h / 2 : s;
    for ( int x = 0; x < s; x++ ) for ( int y = 0; y < h; y++ ) {
            edgeDst[x * h + y] *= x / float(s); edgeDst[(w - 1 - x)*h + y] *= x / float(s);
        }
    for ( int x = 0; x < w; x++ ) for ( int y = 0; y < s; y++ ) {
            edgeDst[x * h + y] *= y / float(s); edgeDst[x * h + (h - 1 - y)] *= y / float(s);
        }
}

} // EdgeBox

