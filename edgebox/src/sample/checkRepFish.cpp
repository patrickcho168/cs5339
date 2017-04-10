#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <vector>
#include "../EdgeBox/edgeBoxes.hpp"
#include "json/json.h"
#include "json/json-forwards.h"

int main(int argc, char** argv) {

    std::string bbFolder = "../../BBGT/";
    std::vector<std::string> fishTypes;
    fishTypes.push_back("ALB_BBGT.txt");
    fishTypes.push_back("BET_BBGT.txt");
    fishTypes.push_back("DOL_BBGT.txt");
    fishTypes.push_back("LAG_BBGT.txt");
    fishTypes.push_back("SHARK_BBGT.txt");
    fishTypes.push_back("YFT_BBGT.txt");

    std::string picFolder = "../../train/";
    std::vector<std::string> fishFolders;
    fishFolders.push_back("ALB/");
    fishFolders.push_back("BET/");
    fishFolders.push_back("DOL/");
    fishFolders.push_back("LAG/");
    fishFolders.push_back("SHARK/");
    fishFolders.push_back("YFT/");

    for (unsigned int j=0; j<fishTypes.size(); ++j) {
        std::string nextFile = bbFolder + fishTypes[j];
        std::cout << nextFile << std::endl;
        Json::Value root;

        std::ifstream ifile(nextFile, std::ifstream::binary);
        ifile >> root;

        for( auto itr = root.begin() ; itr != root.end() ; itr++ ) {
            std::string imageFile = picFolder + fishFolders[j] + (*itr)["filename"].asString();
            auto rootAnno = (*itr)["annotations"];
            cv::Mat src = cv::imread(imageFile);
            if (src.empty()) {
                std::cout << "Error: Cannot read image file " << imageFile << std::endl;
                return 1;
            }
            cv::RNG rng;
            for ( auto itr2 = rootAnno.begin(); itr2 != rootAnno.end(); itr2++) {
                auto nextRootAnno = (*itr2);
                if (nextRootAnno["iou"].asDouble() > 0.5) {
                    double x = nextRootAnno["x"].asDouble();
                    double y = nextRootAnno["y"].asDouble();
                    double w = nextRootAnno["width"].asDouble();
                    double h = nextRootAnno["height"].asDouble();
                    cv::Point P1(x, y);
                    cv::Point P2(x + w, y + h);
                    unsigned int val = rng;
                    cv::Scalar color((val & 0xFF), ((val >> 8) & 0xFF), ((val >> 16) & 0xFF));
                    cv::rectangle(src, P1, P2, color);
                }
            }
            cv::imshow("eb", src);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    
    }

    return 0;
}

