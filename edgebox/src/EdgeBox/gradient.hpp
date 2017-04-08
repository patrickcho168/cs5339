/*
 * gradient.hpp
 *
 *  Created on: May 5, 2016
 *      Author: patcho
 */

#ifndef GRADIENT_HPP_
#define GRADIENT_HPP_

namespace EdgeBox
{

namespace internal
{

/**ConvConst**/

void convTriY( float *I, float *O, int h, int r, int s );

void convTri( float *I, float *O, int h, int w, int d, int r, int s );

void convTri1Y( float *I, float *O, int h, float p, int s );

void convTri1( float *I, float *O, int h, int w, int d, float p, int s );

/**Resample**/

void resample( float *A, float *B, int ha, int hb, int wa, int wb, int d, float r );

void resampleCoef( int ha, int hb, int &n, int *&yas, int *&ybs, float *&wts, int bd[2], int pad = 0 );

/**Impad**/

void imPad( int *A, int *B, int h, int w, int d, int pt, int pb,
            int pl, int pr, int flag, int val );

/**Gradient**/

// compute x and y gradients for just one column (uses sse)
void grad1( float *I, float *Gx, float *Gy, int h, int w, int x );

// compute x and y gradients at each location (uses sse)
void grad2( float *I, float *Gx, float *Gy, int h, int w, int d );

// build lookup table a[] s.t. a[x*n]~=acos(x) for x in [-1,1]
float* acosTable();

// compute gradient magnitude and orientation at each location (uses sse)
void gradMag( float *I, float *M, float *O, int h, int w, int d, bool full );

// normalize gradient magnitude at each location (uses sse)
void gradMagNorm( float *M, float *S, int h, int w, float norm );

// helper for gradHist, quantize O and M into O0, O1 and M0, M1 (uses sse)
void gradQuantize( float *O, float *M, int *O0, int *O1, float *M0, float *M1,
                   int nb, int n, float norm, int nOrients, bool full, bool interpolate );

// compute nOrients gradient histograms per bin x bin block of pixels
void gradHist( float *M, float *O, float *H, int h, int w,
               int bin, int nOrients, int softBin, bool full );

/******************************************************************************/

} // internal

} // EdgeBox

#endif /* GRADIENT_HPP_ */
