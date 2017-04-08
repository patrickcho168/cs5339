/*
 * edgesDetect.cpp
 *
 *  Created on: May 13, 2016
 *      Author: patcho
 */

/*******************************************************************************
* Structured Edge Detection Toolbox      Version 3.01
* Code written by Piotr Dollar, 2014.
* Licensed under the MSR-LA Full Rights License [see license.txt]
*******************************************************************************/
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include "edgesDetect.hpp"
#include "randomForest.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace EdgeBox
{

typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;
template<typename T> inline T min( T x, T y ) { return x < y ? x : y; }

// construct lookup array for mapping fids to channel indices
static uint32* buildLookup( int *dims, int w ) {
    int c, r, z, n = w * w * dims[2]; uint32 *cids = new uint32[n]; n = 0;
    for (z = 0; z < dims[2]; z++) for (c = 0; c < w; c++) for (r = 0; r < w; r++)
                cids[n++] = z * dims[0] * dims[1] + c * dims[0] + r;
    return cids;
}

// construct lookup arrays for mapping fids for self-similarity channel
static void buildLookupSs( uint32 *&cids1, uint32 *&cids2, int *dims, int w, int m ) {
    int i, j, z, z1, c, r; int locs[1024];
    int m2 = m * m, n = m2 * (m2 - 1) / 2 * dims[2], s = int(w / m / 2.0 + .5);
    cids1 = new uint32[n]; cids2 = new uint32[n]; n = 0;
    for (i = 0; i < m; i++) locs[i] = uint32((i + 1) * (w + 2 * s - 1) / (m + 1.0) - s + .5);
    for (z = 0; z < dims[2]; z++) for (i = 0; i < m2; i++) for (j = i + 1; j < m2; j++) {
                z1 = z * dims[0] * dims[1]; n++;
                r = i % m; c = (i - r) / m; cids1[n - 1] = z1 + locs[c] * dims[0] + locs[r];
                r = j % m; c = (j - r) / m; cids2[n - 1] = z1 + locs[c] * dims[0] + locs[r];
            }
}

void edgesDetect(RandomForest* model, int height, int width, int depth, float* image, float* chns, float* chnsSs, float* edge, uint32* ind) {

    // extract relevant fields from model and options
    const std::vector<uint8>& nSegs = model->nSegs;
    const std::vector<uint8>& segs = model->segs;
    const std::vector<uint16>& eBins = model->edgeBins;
    const std::vector<uint32>& fids = model->featureIds;
    const std::vector<float>& thrs = model->thresholds;
    const std::vector<uint32>& child = model->childs;
    const std::vector<uint32>& eBnds = model->edgeBoundaries;
    const int shrink = (int) model->options.shrinkNumber;
    const int imWidth = (int) model->options.patchSize;
    const int gtWidth = (int) model->options.patchInnerSize;
    const int nChns = (int) model->options.nChns;
    const int nCells = (int) model->options.selfsimilarityGridSize;
    const uint32 nChnFtrs = (uint32) model->options.nChnFtrs;
    const int stride = (int) model->options.stride;
    const int nTreesEval = (int) std::min(model->options.numberOfTreesToEvaluate, model->options.numberOfTrees);
    int sharpen = (int) model->options.sharpen;
    int nThreads = (int) model->options.numThreads;
    const int nBnds = int(model->edgeBoundaries.size() - 1) / int(model->thresholds.size());
    if ( sharpen > nBnds - 1 ) {
        sharpen = nBnds - 1;
    }

    // get dimensions and constants
    const int h = (int) height;
    const int w = (int) width;
    const int nTreeNodes = (int) model->featureIds.size() / model->options.numberOfTrees;
    const int nTrees = (int) model->options.numberOfTrees;
    const int h1 = (int) ceil((double(h - imWidth)) / stride);
    const int w1 = (int) ceil((double(w - imWidth)) / stride);
    const int h2 = h1 * stride + gtWidth;
    const int w2 = w1 * stride + gtWidth;
    const int imgDims[3] = {h, w, depth};
    const int outDims[3] = {h2, w2, 1};
    const int chnDims[3] = {h / shrink, w / shrink, nChns};

    // construct lookup tables
    uint32 *iids, *eids, *cids, *cids1, *cids2;
    iids = buildLookup( (int*)imgDims, gtWidth );
    eids = buildLookup( (int*)outDims, gtWidth );
    cids = buildLookup( (int*)chnDims, imWidth / shrink );
    buildLookupSs( cids1, cids2, (int*)chnDims, imWidth / shrink, nCells );

    // apply forest to all patches and store leaf inds
#ifdef _OPENMP
    nThreads = min(nThreads, omp_get_max_threads());
    #pragma omp parallel for num_threads(nThreads)
#endif
    for ( int c = 0; c < w1; c++ ) for ( int t = 0; t < nTreesEval; t++ ) {
            for ( int r0 = 0; r0 < 2; r0++ ) for ( int r = r0; r < h1; r += 2 ) {
                    int o = (r * stride / shrink) + (c * stride / shrink) * h / shrink;
                    // select tree to evaluate
                    int t1 = ((r + c) % 2 * nTreesEval + t) % nTrees;
                    uint32 k = t1 * nTreeNodes;
                    while ( child[k] ) { // while not leaf
                        // compute feature (either channel or self-similarity feature)
                        uint32 f = fids[k]; float ftr;
                        if ( f < nChnFtrs ) ftr = chns[cids[f] + o]; else
                            ftr = chnsSs[cids1[f - nChnFtrs] + o] - chnsSs[cids2[f - nChnFtrs] + o];
                        // compare ftr to threshold and move left or right accordingly
                        if ( ftr < thrs[k] ) k = child[k] - 1; else k = child[k];
                        k += t1 * nTreeNodes;
                    }
                    // store leaf index and update edge maps
                    ind[ r + c * h1 + t * h1 * w1 ] = k;
                }
        }

    // compute edge maps (avoiding collisions from parallel executions)
    if ( !sharpen ) for ( int c0 = 0; c0 < gtWidth / stride; c0++ ) {
#ifdef _OPENMP
            #pragma omp parallel for num_threads(nThreads)
#endif
            for ( int c = c0; c < w1; c += gtWidth / stride ) {
                for ( int r = 0; r < h1; r++ ) for ( int t = 0; t < nTreesEval; t++ ) {
                        uint32 k = ind[ r + c * h1 + t * h1 * w1 ];
                        float *edge1 = edge + (r * stride) + (c * stride) * h2;
                        int b0 = eBnds[k * nBnds], b1 = eBnds[k * nBnds + 1]; if (b0 == b1) continue;
                        for ( int b = b0; b < b1; b++ ) edge1[eids[eBins[b]]]++;
                    }
            }
        }

    // computed sharpened edge maps, snapping to local color values
    if ( sharpen ) {
        // compute neighbors array
        const int g = gtWidth; uint16 neighbor[4096 * 4];
        for ( int c = 0; c < g; c++ ) for ( int r = 0; r < g; r++ ) {
                int i = c * g + r; uint16 *neighbor1 = neighbor + i * 4;
                neighbor1[0] = c > 0 ? i - g : i; neighbor1[1] = c < g - 1 ? i + g : i;
                neighbor1[2] = r > 0 ? i - 1 : i; neighbor1[3] = r < g - 1 ? i + 1 : i;
            }
#ifdef _OPENMP
        #pragma omp parallel for num_threads(nThreads)
#endif
        for ( int c = 0; c < w1; c++ ) for ( int r = 0; r < h1; r++ ) {
                for ( int t = 0; t < nTreesEval; t++ ) {
                    // get current segment and copy into segMask
                    uint32 k = ind[ r + c * h1 + t * h1 * w1 ];
                    int m = nSegs[k]; if ( m == 1 ) continue;
                    uint8 segMask[4096];
                    // Get segmentation mask
                    // memcpy(segMask,&segs[k*g*g], g*g*sizeof(uint8));
                    std::copy( segs.begin() + g * g * k, segs.begin() + g * g * (k + 1), segMask );
                    // compute color model for each segment using every other pixel
                    // ns stores number of pixels in corresponding segment
                    // mus stores total color for each segment and each channel (Luv)
                    int ci, ri, s, z; float ns[100], mus[1000];
                    // Get original image
                    const float *image1 = image + (c * stride + (imWidth - g) / 2) * h + r * stride + (imWidth - g) / 2;
                    for ( s = 0; s < m; s++ ) { ns[s] = 0; for ( z = 0; z < depth; z++ ) mus[s * depth + z] = 0; }
                    for ( ci = 0; ci < g; ci += 2 ) for ( ri = 0; ri < g; ri += 2 ) {
                            s = segMask[ci * g + ri]; ns[s]++;
                            for ( z = 0; z < depth; z++ ) mus[s * depth + z] += image1[z * h * w + ci * h + ri];
                        }
                    for (s = 0; s < m; s++) for ( z = 0; z < depth; z++ ) mus[s * depth + z] /= ns[s];
                    // update segment segMask according to local color values
                    int b0 = eBnds[k * nBnds], b1 = eBnds[k * nBnds + sharpen];
                    for ( int b = b0; b < b1; b++ ) {
                        float vs[10], d, e, eBest = 1e10f;
                        int i, sBest = -1, ss[4];
                        for ( i = 0; i < 4; i++ ) ss[i] = segMask[neighbor[eBins[b] * 4 + i]];
                        for ( z = 0; z < depth; z++ ) {
                            vs[z] = image1[iids[eBins[b]] + z * h * w];
                        }
                        for ( i = 0; i < 4; i++ ) {
                            s = ss[i]; if (s == sBest) continue;
                            e = 0;
                            for ( z = 0; z < depth; z++ ) {
                                d = mus[s * depth + z] - vs[z];
                                e += d * d;
                            }
                            if ( e < eBest ) {
                                eBest = e;
                                sBest = s;
                            }
                        }
                        segMask[eBins[b]] = sBest;
                    }
                    // convert mask to edge maps (examining expanded set of pixels)
                    float *edge1 = edge + c * stride * h2 + r * stride; b1 = eBnds[k * nBnds + sharpen + 1];
                    for ( int b = b0; b < b1; b++ ) {
                        int i = eBins[b]; uint8 s = segMask[i]; uint16 *neighbor1 = neighbor + i * 4;
                        if ( s != segMask[neighbor1[0]] || s != segMask[neighbor1[1]] || s != segMask[neighbor1[2]] || s != segMask[neighbor1[3]] )
                            edge1[eids[i]]++;
                    }
                }
            }
    }

    // free memory
    delete [] iids; delete [] eids;
    delete [] cids; delete [] cids1; delete [] cids2;
}

} // EdgeBox

