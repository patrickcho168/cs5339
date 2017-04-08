#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "../EdgeBox/edgeBoxes.hpp"
#include "../EdgeBox/evaluation.hpp"

int main(int argc, char** argv) {
    std::string structuredEdgeModel = "../data/data.dat";
    std::string annotationsFolder = "../data/Annotations";
    std::string imagesFolder = "../data/JPEGImages";
    float minReqIOU = 0.5;
    int showImage = 1;
    EdgeBox::runVocEvaluation(structuredEdgeModel, EdgeBox::EDGEBOXMODEL_50, annotationsFolder, imagesFolder, minReqIOU, showImage);
    return 0;
}

