#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <dirent.h>
#include "../EdgeBox/edgeBoxes.hpp"
#include "json/json.h"
#include "json/json-forwards.h"

void listFile(char* dir, std::vector<std::string>& allFiles) {
    DIR *pDIR;
    struct dirent *entry;
    if ((pDIR = opendir(dir))) {
        while ((entry = readdir(pDIR))) {
            if ( strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 )
                allFiles.push_back(entry->d_name);
        }
        closedir(pDIR);
    }
}

int main(int argc, char** argv) {
    // if (argc != 3) {
    //     std::cout << "Usage: " << argv[0] << " <image-file> <min-score>" << std::endl;
    //     return 1;
    // }

    std::string testFolder = "../../test_stg2/";
    char folder[1024];
    strcpy(folder, testFolder.c_str());
    std::vector<std::string> allFiles;
    listFile(folder, allFiles);
    sort(allFiles.begin(), allFiles.end());

    EdgeBox::EdgeBox eb;
    eb.getEdgeModel("../data/data.dat");
    // Either preset params
    eb.loadEdgeBoxPredeterminedParameters(EdgeBox::EDGEBOXMODEL_50);
    // Or manually set params
    // eb.loadEdgeBoxManualParameters("../data/config.txt");

    Json::Value vec(Json::arrayValue);

    for (unsigned int i = 0; i < allFiles.size(); i++) {

        const std::string imageFile = testFolder + allFiles.at(i);
        std::cout << imageFile << std::endl;
        // const std::string imageFile = argv[1];
        const double minScore = 0.01;

        cv::Mat src = cv::imread(imageFile);
        if (src.empty()) {
            std::cout << "Error: Cannot read image file " << imageFile << std::endl;
            return 1;
        }

        cv::Mat Edge, Orientation;
        eb.runEdgeDetection(src, Edge, Orientation);

        std::vector<EdgeBox::Candidate> candidates;
        eb.runEdgeBoxes(src, candidates);

        // cv::RNG rng;
        Json::Value onePic;
        onePic["filename"] = allFiles.at(i);
        Json::Value annotations(Json::arrayValue);


        for (unsigned int i = 0; i < candidates.size(); i++) {
            const double score = candidates[i].score;
            if (score >= minScore) {
                const cv::Rect& rect = candidates[i].box;
                Json::Value oneAnnotation;
                oneAnnotation["height"] = rect.height;
                oneAnnotation["width"] = rect.width;
                oneAnnotation["x"] = rect.x;
                oneAnnotation["y"] = rect.y;
                // cv::Point P1(rect.x, rect.y);
                // cv::Point P2(rect.x + rect.width, rect.y + rect.height);
                // unsigned int val = rng;
                // cv::Scalar color((val & 0xFF), ((val >> 8) & 0xFF), ((val >> 16) & 0xFF));
                // cv::rectangle(src, P1, P2, color);
                annotations.append(oneAnnotation);
            }
        }
        onePic["annotations"] = annotations;
        vec.append(onePic);

        // cv::imshow("eb", src);
        // cv::waitKey(0);
        // cv::destroyAllWindows();
    }

    std::ofstream file_id;
    file_id.open("test_stg2.txt");

    file_id << vec;

    file_id.close();

    return 0;
}

