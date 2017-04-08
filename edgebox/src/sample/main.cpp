#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "../EdgeBox/edgeBoxes.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <image-file> <min-score>" << std::endl;
        return 1;
    }
    const std::string imageFile = argv[1];
    const double minScore = std::atof(argv[2]);

    cv::Mat src = cv::imread(imageFile);
    if (src.empty()) {
        std::cout << "Error: Cannot read image file " << imageFile << std::endl;
        return 1;
    }

    EdgeBox::EdgeBox eb;
    eb.getEdgeModel("../data/data.dat");
    // Either preset params
    eb.loadEdgeBoxPredeterminedParameters(EdgeBox::EDGEBOXMODEL_50);
    // Or manually set params
    // eb.loadEdgeBoxManualParameters("../data/config.txt");

    cv::Mat Edge, Orientation;
    eb.runEdgeDetection(src, Edge, Orientation);

    std::vector<EdgeBox::Candidate> candidates;
    eb.runEdgeBoxes(src, candidates);

    cv::RNG rng;
    for (unsigned int i = 0; i < candidates.size(); i++) {
        const double score = candidates[i].score;
        if (score >= minScore) {
            const cv::Rect& rect = candidates[i].box;
            cv::Point P1(rect.x, rect.y);
            cv::Point P2(rect.x + rect.width, rect.y + rect.height);
            unsigned int val = rng;
            cv::Scalar color((val & 0xFF), ((val >> 8) & 0xFF), ((val >> 16) & 0xFF));
            cv::rectangle(src, P1, P2, color);
        }
    }

    cv::imshow("eb", src);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}

