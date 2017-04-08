/*
 * randomForest.hpp
 *
 *  Created on: May 13, 2016
 *      Author: patcho
 */

#ifndef RANDOMFOREST_HPP_
#define RANDOMFOREST_HPP_

#include <vector>

namespace EdgeBox
{

/*! random forest options, e.g. number of trees */
struct RandomForestOptions
{
    // model params

    int numberOfOutputChannels; /*!< number of edge orientation bins for output */

    int patchSize;              /*!< width of image patches */
    int patchInnerSize;         /*!< width of predicted part inside patch*/

    // feature params

    int regFeatureSmoothingRadius;    /*!< radius for smoothing of regular features
                                       *   (using convolution with triangle filter) */

    int ssFeatureSmoothingRadius;     /*!< radius for smoothing of additional features
                                       *   (using convolution with triangle filter) */

    int shrinkNumber;                 /*!< amount to shrink channels */

    int numberOfGradientOrientations; /*!< number of orientations per gradient scale */

    int gradientSmoothingRadius;      /*!< radius for smoothing of gradients
                                       *   (using convolution with triangle filter) */

    int gradientNormalizationRadius;  /*!< gradient normalization radius */
    int selfsimilarityGridSize;       /*!< number of self similarity cells */

    // detection params
    int numberOfTrees;            /*!< number of trees in forest to train */
    int numberOfTreesToEvaluate;  /*!< number of trees to evaluate per location */

    int stride;                   /*!< stride at which to compute edges */
    int sharpen = 0;     // ADDED
    int nChns = 13;      // ADDED
    int nChnFtrs = 3328; // ADDED
    int numThreads = 4;
};

/*! random forest used to detect edges */
struct RandomForest
{
    RandomForestOptions options;

    int numberOfTreeNodes;

    std::vector <unsigned int> featureIds;     /*!< feature coordinate thresholded at k-th node */
    std::vector <float> thresholds;   /*!< threshold applied to featureIds[k] at k-th node */
    std::vector <unsigned int> childs;         /*!< k --> child[k] - 1, child[k] */

    std::vector <unsigned char> nSegs;
    std::vector <unsigned char> segs;
    std::vector <unsigned int> edgeBoundaries;
    std::vector <unsigned short> edgeBins;
};

} // EdgeBox

#endif /* RANDOMFOREST_HPP_ */
