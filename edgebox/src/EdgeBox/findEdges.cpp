/*
 * findEdges.cpp
 *
 *  Created on: Mar 24, 2016
 *      Author: patcho
 */

#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <typeinfo>
#include <iostream>
#include <fstream>
#include <string.h>
#include <chrono>
#include "gradient.hpp"
#include "sse.hpp"
#include "randomForest.hpp"
#include "edgesDetect.hpp"
#include "edgesNms.hpp"
#include "wrappers.hpp"

namespace EdgeBox
{

namespace internal
{

static const float PI = 3.14159265f;

// Constants for rgb2luv conversion and lookup table for y-> l conversion
static float* rgb2luv_setup( float z, float *mr, float *mg, float *mb,
                             float &minu, float &minv, float &un, float &vn )
{
    // set constants for conversion
    const float y0 = (float) ((6.0 / 29) * (6.0 / 29) * (6.0 / 29));
    const float a = (float) ((29.0 / 3) * (29.0 / 3) * (29.0 / 3));
    un = (float) 0.197833; vn = (float) 0.468331;
    mr[0] = (float) 0.430574 * z; mr[1] = (float) 0.222015 * z; mr[2] = (float) 0.020183 * z;
    mg[0] = (float) 0.341550 * z; mg[1] = (float) 0.706655 * z; mg[2] = (float) 0.129553 * z;
    mb[0] = (float) 0.178325 * z; mb[1] = (float) 0.071330 * z; mb[2] = (float) 0.939180 * z;
    float maxi = (float) 1.0 / 270; minu = -88 * maxi; minv = -134 * maxi;
    // build (padded) lookup table for y->l conversion assuming y in [0,1]
    static float lTable[1064]; static bool lInit = false;
    if ( lInit ) return lTable;
    float y, l;
    for (int i = 0; i < 1025; i++) {
        y = (float) (i / 1024.0);
        l = y > y0 ? 116 * (float)pow((double)y, 1.0 / 3.0) - 16 : y * a;
        lTable[i] = l * maxi;
    }
    for (int i = 1025; i < 1064; i++) lTable[i] = lTable[i - 1];
    lInit = true; return lTable;
}

// Convert from rgb to luv
static void rgb2luv( float *I, float *J, int n, float nrm ) {
    float minu, minv, un, vn, mr[3], mg[3], mb[3];
    float *lTable = rgb2luv_setup(nrm, mr, mg, mb, minu, minv, un, vn);
    float *L = J, *U = L + n, *V = U + n; float *R = I, *G = R + n, *B = G + n;
    for ( int i = 0; i < n; i++ ) {
        float r, g, b, x, y, z, l;
        r = (float) * R++; g = (float) * G++; b = (float) * B++;
        x = mr[0] * r + mg[0] * g + mb[0] * b;
        y = mr[1] * r + mg[1] * g + mb[1] * b;
        z = mr[2] * r + mg[2] * g + mb[2] * b;
        l = lTable[(int)(y * 1024)];
        *(L++) = l; z = 1 / (x + 15 * y + 3 * z + (float)1e-35);
        *(U++) = l * (13 * 4 * x * z - 13 * un) - minu;
        *(V++) = l * (13 * 9 * y * z - 13 * vn) - minv;
    }
}

// Convert from rgb to luv using sse
static void rgb2luv_sse( float *I, float *J, int n, float nrm ) {
    const int k = 256; float R[k], G[k], B[k];
    if ( ((size_t(R) & 15) || (size_t(G) & 15) || (size_t(B) & 15) || (size_t(I) & 15) || (size_t(J) & 15))
            || n % 4 > 0 ) { rgb2luv(I, J, n, nrm); return; }
    int i = 0, i1, n1; float minu, minv, un, vn, mr[3], mg[3], mb[3];
    float *lTable = rgb2luv_setup(nrm, mr, mg, mb, minu, minv, un, vn);
    while ( i < n ) {
        n1 = i + k; if (n1 > n) n1 = n; float *J1 = J + i; float *R1, *G1, *B1;
        R1 = ((float*)I) + i; G1 = R1 + n; B1 = G1 + n;
        // compute RGB -> XYZ
        for ( int j = 0; j < 3; j++ ) {
            __m128 _mr, _mg, _mb, *_J = (__m128*) (J1 + j * n);
            __m128 *_R = (__m128*) R1, *_G = (__m128*) G1, *_B = (__m128*) B1;
            _mr = SET(mr[j]); _mg = SET(mg[j]); _mb = SET(mb[j]);
            for ( i1 = i; i1 < n1; i1 += 4 ) * (_J++) = ADD( ADD(MUL(*(_R++), _mr),
                        MUL(*(_G++), _mg)), MUL(*(_B++), _mb));
        }
        {   // compute XZY -> LUV (without doing L lookup/normalization)
            __m128 _c15, _c3, _cEps, _c52, _c117, _c1024, _cun, _cvn;
            _c15 = SET(15.0f); _c3 = SET(3.0f); _cEps = SET(1e-35f);
            _c52 = SET(52.0f); _c117 = SET(117.0f), _c1024 = SET(1024.0f);
            _cun = SET(13 * un); _cvn = SET(13 * vn);
            __m128 *_X, *_Y, *_Z, _x, _y, _z;
            _X = (__m128*) J1; _Y = (__m128*) (J1 + n); _Z = (__m128*) (J1 + 2 * n);
            for ( i1 = i; i1 < n1; i1 += 4 ) {
                _x = *_X; _y = *_Y; _z = *_Z;
                _z = RCP(ADD(_x, ADD(_cEps, ADD(MUL(_c15, _y), MUL(_c3, _z)))));
                *(_X++) = MUL(_c1024, _y);
                *(_Y++) = SUB(MUL(MUL(_c52, _x), _z), _cun);
                *(_Z++) = SUB(MUL(MUL(_c117, _y), _z), _cvn);
            }
        }
        {   // perform lookup for L and finalize computation of U and V
            for ( i1 = i; i1 < n1; i1++ ) J[i1] = lTable[(int)J[i1]];
            __m128 *_L, *_U, *_V, _l, _cminu, _cminv;
            _L = (__m128*) J1; _U = (__m128*) (J1 + n); _V = (__m128*) (J1 + 2 * n);
            _cminu = SET(minu); _cminv = SET(minv);
            for ( i1 = i; i1 < n1; i1 += 4 ) {
                _l = *(_L++);
                *_U = SUB(MUL(_l, *_U), _cminu); _U++;
                *_V = SUB(MUL(_l, *_V), _cminv); _V++;
            }
        }
        i = n1;
    }
}

}

void loadBinaryModel( RandomForest* __rf, std::string dataFile ) {
    std::ifstream input(dataFile.c_str(), std::ios::in | std::ifstream::binary);

    int featureSize;
    input.read(reinterpret_cast<char*>(&featureSize), sizeof(int));
    __rf->featureIds.resize(featureSize);
    input.read(reinterpret_cast<char*>(__rf->featureIds.data()), featureSize * sizeof(unsigned int));

    input.read(reinterpret_cast<char*>(&featureSize), sizeof(int));
    __rf->thresholds.resize(featureSize);
    input.read(reinterpret_cast<char*>(__rf->thresholds.data()), featureSize * sizeof(float));

    input.read(reinterpret_cast<char*>(&featureSize), sizeof(int));
    __rf->childs.resize(featureSize);
    input.read(reinterpret_cast<char*>(__rf->childs.data()), featureSize * sizeof(unsigned int));

    input.read(reinterpret_cast<char*>(&featureSize), sizeof(int));
    __rf->nSegs.resize(featureSize);
    input.read(reinterpret_cast<char*>(__rf->nSegs.data()), featureSize * sizeof(unsigned char));

    input.read(reinterpret_cast<char*>(&featureSize), sizeof(int));
    __rf->segs.resize(featureSize);
    input.read(reinterpret_cast<char*>(__rf->segs.data()), featureSize * sizeof(unsigned char));

    input.read(reinterpret_cast<char*>(&featureSize), sizeof(int));
    __rf->edgeBoundaries.resize(featureSize);
    input.read(reinterpret_cast<char*>(__rf->edgeBoundaries.data()), featureSize * sizeof(unsigned int));

    input.read(reinterpret_cast<char*>(&featureSize), sizeof(int));
    __rf->edgeBins.resize(featureSize);
    input.read(reinterpret_cast<char*>(__rf->edgeBins.data()), featureSize * sizeof(unsigned short));

    input.read(reinterpret_cast<char*>(&__rf->numberOfTreeNodes), sizeof(int));
    input.read(reinterpret_cast<char*>(&__rf->options.numberOfOutputChannels), sizeof(int));
    input.read(reinterpret_cast<char*>(&__rf->options.patchSize), sizeof(int));
    input.read(reinterpret_cast<char*>(&__rf->options.patchInnerSize), sizeof(int));
    input.read(reinterpret_cast<char*>(&__rf->options.regFeatureSmoothingRadius), sizeof(int));
    input.read(reinterpret_cast<char*>(&__rf->options.ssFeatureSmoothingRadius), sizeof(int));
    input.read(reinterpret_cast<char*>(&__rf->options.shrinkNumber), sizeof(int));
    input.read(reinterpret_cast<char*>(&__rf->options.numberOfGradientOrientations), sizeof(int));
    input.read(reinterpret_cast<char*>(&__rf->options.gradientSmoothingRadius), sizeof(int));
    input.read(reinterpret_cast<char*>(&__rf->options.gradientNormalizationRadius), sizeof(int));
    input.read(reinterpret_cast<char*>(&__rf->options.selfsimilarityGridSize), sizeof(int));
    input.read(reinterpret_cast<char*>(&__rf->options.numberOfTrees), sizeof(int));
    input.read(reinterpret_cast<char*>(&__rf->options.numberOfTreesToEvaluate), sizeof(int));
    input.read(reinterpret_cast<char*>(&__rf->options.stride), sizeof(int));
    input.read(reinterpret_cast<char*>(&__rf->options.sharpen), sizeof(int));
    input.read(reinterpret_cast<char*>(&__rf->options.nChns), sizeof(int));
    input.read(reinterpret_cast<char*>(&__rf->options.nChnFtrs), sizeof(int));
    input.read(reinterpret_cast<char*>(&__rf->options.numThreads), sizeof(int));
}

void findEdges(RandomForest* __rf, cv::Mat nSrc, float* edgeFinal, float* orientFinal) {
    // Padding to increase both width and height by at least patchSize (usually 32)
    // and make both sides divisible by 4
    int originalWidth = nSrc.cols;
    int originalHeight = nSrc.rows;
    int padding = ( __rf->options.patchSize) / 2;
    int padding2 = padding + (4 - (nSrc.rows + 2 * padding) % 4) % 4;
    int padding3 = padding + (4 - (nSrc.cols + 2 * padding) % 4) % 4;
    cv::copyMakeBorder( nSrc, nSrc, padding, padding2, padding, padding3, cv::BORDER_REFLECT ); // TOP BOTTOM LEFT RIGHT

    // Convert to MATLAB Friendly Format for usage of pdollar code
    float* imgPadded = new float[nSrc.rows * nSrc.cols * 3];
    int nn = 0;
    for (int m = 2; m >= 0; m--) {
        for (int j = 0; j < nSrc.cols; j++) {
            for (int k = 0; k < nSrc.rows; k++) {
                cv::Vec3b temp = nSrc.at<cv::Vec3b>(k, j);
                imgPadded[nn] = temp.val[m];
                nn++;
            }
        }
    }

    // Convert to Luv
    int shrink = __rf->options.shrinkNumber;
    float* imgPaddedLuv = new float[nSrc.rows * nSrc.cols * 3];
    EdgeBox::internal::rgb2luv_sse(imgPadded, imgPaddedLuv, nSrc.rows * nSrc.cols, 1 / 255.0);

    // imgChannels stores all 13 channels of information
    float* imgChannels = new float[(nSrc.rows / shrink) * (nSrc.cols / shrink)*__rf->options.nChns];
    // Store 3 color channels in CIE-LUV color space in imgChannels
    EdgeBox::internal::resample( imgPaddedLuv, imgChannels, nSrc.rows, nSrc.rows / shrink, nSrc.cols, nSrc.cols / shrink, 3, 1.0f);

    // First run on original size
    float* imgTmp;
    int height = nSrc.rows;
    int width = nSrc.cols;
    bool memoryI = true;
    if (__rf->options.gradientSmoothingRadius > 0) {
        imgTmp = new float[(nSrc.rows) * (nSrc.cols) * 3];
        EdgeBox::internal::convTri( imgPaddedLuv, imgTmp, height, width, 3, __rf->options.gradientSmoothingRadius, 1 );
    } else {
        imgTmp = imgPaddedLuv;
        memoryI = false;
    }
    float* magnitude1 = new float[height * width];
    float* orientation1 = new float[height * width];
    EdgeBox::internal::gradMag( imgTmp, magnitude1, orientation1, height, width, 3, false );
    float* mag1 = new float[width * height];
    EdgeBox::internal::convTri( magnitude1, mag1, height, width, 1, __rf->options.gradientNormalizationRadius, 1 );
    EdgeBox::internal::gradMagNorm(magnitude1, mag1, height, width, 0.01);
    // Store Channel 5-8
    EdgeBox::internal::gradHist( magnitude1, orientation1, &imgChannels[(nSrc.rows / shrink) * (nSrc.cols / shrink) * 4], height, width, std::max(1, shrink), __rf->options.numberOfGradientOrientations, 0, false);
    // Store Channel 4
    EdgeBox::internal::resample( magnitude1, &imgChannels[(nSrc.rows / shrink) * (nSrc.cols / shrink) * 3], nSrc.rows, nSrc.rows / shrink, nSrc.cols, nSrc.cols / shrink, 1, 1.0f);
    delete [] magnitude1; delete [] orientation1; delete [] mag1;
    if (memoryI) {
        delete [] imgTmp;
    }

    // Then run on shrinked size
    float* imgTmpShrink;
    height = nSrc.rows / shrink;
    width  = nSrc.cols / shrink;
    if (__rf->options.gradientSmoothingRadius > 0) {
        imgTmpShrink = new float[(nSrc.rows / shrink) * (nSrc.cols / shrink) * 3];
        EdgeBox::internal::convTri( imgChannels, imgTmpShrink, height, width, 3, __rf->options.gradientSmoothingRadius, 1 );
        memoryI = true;
    } else {
        imgTmpShrink = imgChannels;
        memoryI = false;
    }
    float* orientation2 = new float[height * width];
    EdgeBox::internal::gradMag( imgTmpShrink, &imgChannels[width * height * 8], orientation2, height, width, 3, false );
    float* mag2 = new float[width * height];
    // Store Channel 9
    EdgeBox::internal::convTri( &imgChannels[width * height * 8], mag2, height, width, 1, __rf->options.gradientNormalizationRadius, 1 );
    EdgeBox::internal::gradMagNorm(&imgChannels[width * height * 8], mag2, height, width, 0.01);
    // Store Channel 10-13
    EdgeBox::internal::gradHist( &imgChannels[width * height * 8], orientation2, &imgChannels[width * height * 9], height, width, std::max(1, shrink / 2), __rf->options.numberOfGradientOrientations, 0, false);
    delete [] orientation2; delete [] mag2;
    if (memoryI) {
        delete [] imgTmpShrink;
    }

    delete [] imgPaddedLuv;
    int chnSm = round(__rf->options.regFeatureSmoothingRadius / shrink);
    int simSm = round(__rf->options.ssFeatureSmoothingRadius / shrink);
    float* chnsReg = new float[(nSrc.rows / shrink) * (nSrc.cols / shrink)*__rf->options.nChns];
    float* chnsSim = new float[(nSrc.rows / shrink) * (nSrc.cols / shrink)*__rf->options.nChns];
    if (chnSm > 1) {
        EdgeBox::internal::convTri( imgChannels, chnsReg, nSrc.rows / shrink, nSrc.cols / shrink, __rf->options.nChns, chnSm, 1 );
    } else {
        EdgeBox::internal::convTri1( imgChannels, chnsReg, nSrc.rows / shrink, nSrc.cols / shrink, __rf->options.nChns, (12 / chnSm) / (chnSm + 2) - 2, 1 );
    }
    if (simSm > 1) {
        EdgeBox::internal::convTri( imgChannels, chnsSim, nSrc.rows / shrink, nSrc.cols / shrink, __rf->options.nChns, simSm, 1 );
    } else {
        EdgeBox::internal::convTri1( imgChannels, chnsSim, nSrc.rows / shrink, nSrc.cols / shrink, __rf->options.nChns, (12 / simSm) / (simSm + 2) - 2, 1 );
    }

    int sharpen = __rf->options.sharpen;
    float* imgSharpen = new float[nSrc.rows * nSrc.cols * 3];
    if (sharpen) {
        float r = 1.0;
        EdgeBox::internal::resample( imgPadded, imgPadded, nSrc.rows, nSrc.rows, nSrc.cols, nSrc.cols, 3, 1.0 / 255);
        EdgeBox::internal::convTri1( imgPadded, imgSharpen, nSrc.rows, nSrc.cols, 3, (12 / r) / (r + 2) - 2, 1 );
    }

    const int stride = (int) __rf->options.stride;
    const int imWidth = (int) __rf->options.patchSize;
    const int gtWidth = (int) __rf->options.patchInnerSize;
    const int h1 = (int) ceil(double(nSrc.rows - imWidth) / stride);
    const int w1 = (int) ceil(double(nSrc.cols - imWidth) / stride);
    const int h2 = h1 * stride + gtWidth;
    const int w2 = w1 * stride + gtWidth;
    float* edgeTmp = new float[h2 * w2](); // Need to initialize edgeTmp to zeros
    unsigned int* inds = new unsigned int[h1 * w1 * __rf->options.numberOfTreesToEvaluate];
    edgesDetect(__rf, nSrc.rows, nSrc.cols, 3, imgSharpen, chnsReg, chnsSim, edgeTmp, inds);

    float t = powf(__rf->options.stride, 2) / powf(__rf->options.patchInnerSize, 2) / __rf->options.numberOfTreesToEvaluate; // 1/(numberOfVotes) = 1/256
    int r = __rf->options.patchInnerSize / 2;
    if (sharpen == 0) {
        t = t * 2; // Why?
    }
    else if (sharpen == 1) {
        t = t * 1.8;
    }
    else  {
        t = t * 1.66;
    }
    float* edgeTmp1 = new float[originalHeight * originalWidth];
    for (int j = 0; j < originalWidth * originalHeight; j++) {
        int wid = j / originalHeight;
        int hei = j % originalHeight;
        edgeTmp1[j] = edgeTmp[(r + wid) * h2 + hei + r] * t;
    }
    int rad = 1;
    float* edgeTmp2 = new float[originalHeight * originalWidth];
    EdgeBox::internal::convTri1( edgeTmp1, edgeTmp2, originalHeight, originalWidth, 1, (12 / rad) / (rad + 2) - 2, 1 );

    // Calculate final Orientation
    EdgeBox::internal::convTri( edgeTmp2, edgeTmp1, originalHeight, originalWidth, 1, 4, 1 );
    float* ox = new float[originalWidth * originalHeight];
    float* oy = new float[originalWidth * originalHeight];
    EdgeBox::internal::grad2( edgeTmp1, ox, oy, originalHeight, originalWidth, 1 );
    float* oxx = new float[originalWidth * originalHeight];
    float* oyx = new float[originalWidth * originalHeight];
    float* oxy = new float[originalWidth * originalHeight];
    float* oyy = new float[originalWidth * originalHeight];
    EdgeBox::internal::grad2(ox, oxx, oyx, originalHeight, originalWidth, 1);
    EdgeBox::internal::grad2(oy, oxy, oyy, originalHeight, originalWidth, 1);
    for (int i = 0; i < originalWidth * originalHeight; i++) {
        orientFinal[i] = fmod(atan((oyy[i] * (oxy[i] <= 0 ? 1 : -1)) / (oxx[i] + 1e-5)) + EdgeBox::internal::PI, EdgeBox::internal::PI); // coarse edge normal orientation (0=left, pi/2=up)
    }
    edgesNms( edgeTmp2, edgeFinal, orientFinal, 2, 0, 1, originalHeight, originalWidth, __rf->options.numThreads);

    delete [] imgPadded; delete [] imgSharpen; delete [] imgChannels;
    delete [] chnsReg; delete [] chnsSim;
    delete [] oxx; delete [] oyx; delete [] oxy; delete [] oyy; delete [] ox; delete [] oy;
    delete [] edgeTmp; delete [] edgeTmp1; delete [] edgeTmp2; delete [] inds;
}

} // EdgeBox
