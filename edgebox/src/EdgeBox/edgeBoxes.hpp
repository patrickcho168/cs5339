/*
 * edgeBoxes.hpp
 *
 *  Created on: May 16, 2016
 *      Author: patcho
 */

#ifndef EDGEBOXES_HPP_
#define EDGEBOXES_HPP_
#include <opencv2/imgproc/imgproc.hpp>
#include "randomForest.hpp"
#include <string>
#include <vector>

namespace EdgeBox
{

// Each candidate consists of a bounding box and a score
struct Candidate {
    Candidate(const cv::Rect& box, const float score) : box(box), score(score) {}
    const cv::Rect box;
    const float score;
};

enum EdgeBoxModel
{
    EDGEBOXMODEL_FAST,
    EDGEBOXMODEL_50,
    EDGEBOXMODEL_70,
    EDGEBOXMODEL_90
};

class EdgeBox
{
private:
    class EdgeBoxImpl;
public:
    EdgeBox();
    void getEdgeModel(const std::string binaryFile);
    void loadEdgeBoxPredeterminedParameters(const EdgeBoxModel type);
    void loadEdgeBoxManualParameters(const std::string config);

    // Returns edgeFinal which contains the edge magnitude and has same dimensions as original image
    // Returns orientationFinal which contains coarse edge normal orientation (0=left, pi/2=up) and has same dimensions as original image
    void runEdgeDetection(const cv::Mat& nSrc, cv::Mat& edgeFinal, cv::Mat& orientationFinal); 

    // Runs EdgeBox
    void runEdgeBoxes(const cv::Mat& src, std::vector<Candidate>& candidates);
    
    EdgeBoxImpl* edgeBoxPImpl;
    ~EdgeBox();
};

} // EdgeBox

#endif /* EDGEBOXES_HPP_ */
