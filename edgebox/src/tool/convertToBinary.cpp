/*
 * convertToBinary.cpp
 *
 *  Created on: May 18, 2016
 *      Author: patcho
 */

#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <fstream>

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
    int numThreads = 4;  // ADDED

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

void loadModel( RandomForest* __rf, std::string inputFile )
{
    cv::FileStorage modelFile(inputFile, cv::FileStorage::READ);
    // CV_Assert( modelFile.isOpened() );
    __rf->options.stride = modelFile["options"]["stride"];
    __rf->options.shrinkNumber
        = modelFile["options"]["shrinkNumber"];
    __rf->options.patchSize
        = modelFile["options"]["patchSize"];
    __rf->options.patchInnerSize
        = modelFile["options"]["patchInnerSize"];

    __rf->options.numberOfGradientOrientations
        = modelFile["options"]["numberOfGradientOrientations"];
    __rf->options.gradientSmoothingRadius
        = modelFile["options"]["gradientSmoothingRadius"];
    __rf->options.regFeatureSmoothingRadius
        = modelFile["options"]["regFeatureSmoothingRadius"];
    __rf->options.ssFeatureSmoothingRadius
        = modelFile["options"]["ssFeatureSmoothingRadius"];
    __rf->options.gradientNormalizationRadius
        = modelFile["options"]["gradientNormalizationRadius"];

    __rf->options.selfsimilarityGridSize
        = modelFile["options"]["selfsimilarityGridSize"];

    __rf->options.numberOfTrees
        = modelFile["options"]["numberOfTrees"];
    __rf->options.numberOfTreesToEvaluate
        = modelFile["options"]["numberOfTreesToEvaluate"];

    __rf->options.numberOfOutputChannels =
        2*(__rf->options.numberOfGradientOrientations + 1) + 3;
    //--------------------------------------------

    cv::FileNode childs = modelFile["childs"];
    cv::FileNode featureIds = modelFile["featureIds"];

    std::vector <int> currentTree;

    for(cv::FileNodeIterator it = childs.begin();
        it != childs.end(); ++it)
    {
        (*it) >> currentTree;
        std::copy(currentTree.begin(), currentTree.end(),
            std::back_inserter(__rf->childs));
    }

    for(cv::FileNodeIterator it = featureIds.begin();
        it != featureIds.end(); ++it)
    {
        (*it) >> currentTree;
        std::copy(currentTree.begin(), currentTree.end(),
            std::back_inserter(__rf->featureIds));
    }

    cv::FileNode thresholds = modelFile["thresholds"];
    std::vector <float> fcurrentTree;

    for(cv::FileNodeIterator it = thresholds.begin();
        it != thresholds.end(); ++it)
    {
        (*it) >> fcurrentTree;
        std::copy(fcurrentTree.begin(), fcurrentTree.end(),
            std::back_inserter(__rf->thresholds));
    }

    cv::FileNode edgeBoundaries = modelFile["edgeBoundaries"];
    cv::FileNode edgeBins = modelFile["edgeBins"];

    for(cv::FileNodeIterator it = edgeBoundaries.begin();
        it != edgeBoundaries.end(); ++it)
    {
        (*it) >> currentTree;
        std::copy(currentTree.begin(), currentTree.end(),
            std::back_inserter(__rf->edgeBoundaries));
    }

    for(cv::FileNodeIterator it = edgeBins.begin();
        it != edgeBins.end(); ++it)
    {
        (*it) >> currentTree;
        std::copy(currentTree.begin(), currentTree.end(),
            std::back_inserter(__rf->edgeBins));
    }

    cv::FileNode nSegs = modelFile["nSegs"];
    cv::FileNode segsFile = modelFile["segs"];

    for(cv::FileNodeIterator it = nSegs.begin();
        it != nSegs.end(); ++it)
    {
        (*it) >> currentTree;
        std::copy(currentTree.begin(), currentTree.end(),
            std::back_inserter(__rf->nSegs));
    }

    for(cv::FileNodeIterator it = segsFile.begin();
        it != segsFile.end(); ++it)
    {
        (*it) >> currentTree;
        std::copy(currentTree.begin(), currentTree.end(),
            std::back_inserter(__rf->segs));
    }
    __rf->numberOfTreeNodes = int( __rf->childs.size() ) / __rf->options.numberOfTrees;
}

// Set read to zero to read from generated outputFile (binary). Set compareResults to 1 to compare binary file with yml file.
// Set read to one to generate outputFile (binary) form inputFile (yml)
int main() {
    bool read=0;
    bool compareResults = 1;
    std::string inputFile = "data.yml.gz";
    std::string outputFile = "data.dat";
    if (read) {
        RandomForest* __rf = new RandomForest;
        std::ifstream input(outputFile, std::ios::in | std::ifstream::binary);

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
        input.read(reinterpret_cast<char*>(__rf->nSegs,data()), featureSize * sizeof(unsigned char));

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

        // COMPARE RESULTS
        if (compareResults) {
            RandomForest __rfTrue;
            loadModel(&__rfTrue, inputFile);
            for (int i=0; i<__rfTrue.featureIds.size(); i++) {
                assert(__rfTrue.featureIds.at(i) == __rf->featureIds.at(i));
            }
            for (int i=0; i<__rfTrue.thresholds.size(); i++) {
                assert(__rfTrue.thresholds.at(i) == __rf->thresholds.at(i));
            }
            for (int i=0; i<__rfTrue.childs.size(); i++) {
                assert(__rfTrue.childs.at(i) == __rf->childs.at(i));
            }
            for (int i=0; i<__rfTrue.nSegs.size(); i++) {
                assert(__rfTrue.nSegs.at(i) == __rf->nSegs.at(i));
            }
            for (int i=0; i<__rfTrue.segs.size(); i++) {
                assert(__rfTrue.segs.at(i) == __rf->segs.at(i));
            }
            for (int i=0; i<__rfTrue.edgeBoundaries.size(); i++) {
                assert(__rfTrue.edgeBoundaries.at(i) == __rf->edgeBoundaries.at(i));
            }
            for (int i=0; i<__rfTrue.edgeBins.size(); i++) {
                assert(__rfTrue.edgeBins.at(i) == __rf->edgeBins.at(i));
            }
            assert(__rfTrue.numberOfTreeNodes == __rf->numberOfTreeNodes);
            assert(__rfTrue.options.numberOfOutputChannels == __rf->options.numberOfOutputChannels);
            assert(__rfTrue.options.patchSize == __rf->options.patchSize);
            assert(__rfTrue.options.patchInnerSize == __rf->options.patchInnerSize);
            assert(__rfTrue.options.regFeatureSmoothingRadius == __rf->options.regFeatureSmoothingRadius);
            assert(__rfTrue.options.ssFeatureSmoothingRadius == __rf->options.ssFeatureSmoothingRadius);
            assert(__rfTrue.options.shrinkNumber == __rf->options.shrinkNumber);
            assert(__rfTrue.options.numberOfGradientOrientations == __rf->options.numberOfGradientOrientations);
            assert(__rfTrue.options.gradientSmoothingRadius == __rf->options.gradientSmoothingRadius);
            assert(__rfTrue.options.gradientNormalizationRadius == __rf->options.gradientNormalizationRadius);
            assert(__rfTrue.options.selfsimilarityGridSize == __rf->options.selfsimilarityGridSize);
            assert(__rfTrue.options.numberOfTrees == __rf->options.numberOfTrees);
            assert(__rfTrue.options.numberOfTreesToEvaluate == __rf->options.numberOfTreesToEvaluate);
            assert(__rfTrue.options.stride == __rf->options.stride);
            assert(__rfTrue.options.sharpen == __rf->options.sharpen);
            assert(__rfTrue.options.nChns == __rf->options.nChns);
            assert(__rfTrue.options.nChnFtrs == __rf->options.nChnFtrs);
            assert(__rfTrue.options.numThreads == __rf->options.numThreads);
        }
        input.close();
    }
    else {
        RandomForest __rf;
        loadModel( &__rf, inputFile );
        std::ofstream fout("data.dat", std::ios::out | std::ios::binary);

        int featureIdsSize = __rf.featureIds.size();
        fout.write((char*)&(featureIdsSize), sizeof(int));
        fout.write((char*)&__rf.featureIds[0], featureIdsSize * sizeof(unsigned int));

        int thresholdsSize = __rf.thresholds.size();
        fout.write((char*)&(thresholdsSize), sizeof(int));
        fout.write((char*)&__rf.thresholds[0], thresholdsSize * sizeof(float));

        int childSize = __rf.childs.size();
        fout.write((char*)&(childSize), sizeof(int));
        fout.write((char*)&__rf.childs[0], childSize * sizeof(unsigned int));

        int nSegsSize = __rf.nSegs.size();
        fout.write((char*)&(nSegsSize), sizeof(int));
        fout.write((char*)&__rf.nSegs[0], nSegsSize * sizeof(unsigned char));

        int segsSize = __rf.segs.size();
        fout.write((char*)&(segsSize), sizeof(int));
        fout.write((char*)&__rf.segs[0], segsSize * sizeof(unsigned char));

        int edgeBoundariesSize = __rf.edgeBoundaries.size();
        fout.write((char*)&(edgeBoundariesSize), sizeof(int));
        fout.write((char*)&__rf.edgeBoundaries[0], edgeBoundariesSize * sizeof(unsigned int));

        int edgeBinsSize = __rf.edgeBins.size();
        fout.write((char*)&(edgeBinsSize), sizeof(int));
        fout.write((char*)&__rf.edgeBins[0], edgeBinsSize * sizeof(unsigned short));

        fout.write((char*)&(__rf.numberOfTreeNodes), sizeof(int));
        fout.write((char*)&(__rf.options.numberOfOutputChannels),sizeof(int));
        fout.write((char*)&(__rf.options.patchSize),sizeof(int));
        fout.write((char*)&(__rf.options.patchInnerSize),sizeof(int));
        fout.write((char*)&(__rf.options.regFeatureSmoothingRadius),sizeof(int));
        fout.write((char*)&(__rf.options.ssFeatureSmoothingRadius),sizeof(int));
        fout.write((char*)&(__rf.options.shrinkNumber),sizeof(int));
        fout.write((char*)&(__rf.options.numberOfGradientOrientations),sizeof(int));
        fout.write((char*)&(__rf.options.gradientSmoothingRadius),sizeof(int));
        fout.write((char*)&(__rf.options.gradientNormalizationRadius),sizeof(int));
        fout.write((char*)&(__rf.options.selfsimilarityGridSize),sizeof(int));
        fout.write((char*)&(__rf.options.numberOfTrees),sizeof(int));
        fout.write((char*)&(__rf.options.numberOfTreesToEvaluate),sizeof(int));
        fout.write((char*)&(__rf.options.stride),sizeof(int));
        fout.write((char*)&(__rf.options.sharpen),sizeof(int));
        fout.write((char*)&(__rf.options.nChns),sizeof(int));
        fout.write((char*)&(__rf.options.nChnFtrs),sizeof(int));
        fout.write((char*)&(__rf.options.numThreads),sizeof(int));
        fout.close();
    }
    return 0;
}
