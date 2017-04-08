/*
 * evaluation.cpp
 *
 *  Created on: Apr 14, 2016
 *      Author: patcho
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#if (CV_MAJOR_VERSION >= 3)
#include <opencv2/imgcodecs/imgcodecs.hpp>
#endif
#include <iostream>
#include <chrono>
#include <dirent.h>
#include <vector>
#include "evaluation.hpp"
#include "edgeBoxes.hpp"
#include "pugixml.hpp"


namespace EdgeBox
{

// bounding box data structures and routines
typedef struct { int c, r, w, h; float s; } Box;
typedef std::vector<Box> Boxes;

/////// 3 Helper Functions for File Reading for VOC2007 Evaluation
std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::istringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

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

static float boxesOverlap( Box &a, Box &b ) {
    float areai, areaj, areaij;
    int r0, r1, c0, c1, r1i, c1i, r1j, c1j;
    r1i = a.r + a.h; c1i = a.c + a.w; if ( a.r >= r1i || a.c >= c1i ) return 0;
    r1j = b.r + b.h; c1j = b.c + b.w; if ( a.r >= r1j || a.c >= c1j ) return 0;
    areai = (float) a.w * a.h; r0 = std::max(a.r, b.r); r1 = std::min(r1i, r1j);
    areaj = (float) b.w * b.h; c0 = std::max(a.c, b.c); c1 = std::min(c1i, c1j);
    areaij = (float) std::max(0, r1 - r0) * std::max(0, c1 - c0);
    return areaij / (areai + areaj - areaij);
}

// For calculating Intersection Over Union
// @param boxes stores vector of Rect returned by EdgeBox Algorithm
// @param gtBoxes stores vector of boxes that are human labelled
// @param correctBoxResult stores the number of gtBoxes that are labelled with IOU of at least minReqIOU (default 0.5)
static int getIOU(cv::Mat& img, const std::vector<Candidate> candidates, const Boxes gtBoxes, const float minReqIOU, const int showImage) {
    int correctBoxes = 0;
    // create output bbs and output to Matlab
    int n = (int) candidates.size();
    double totalIOU = 0.0;
    for (unsigned int j = 0; j < gtBoxes.size(); j++) {
        Box currentGtBox = gtBoxes.at(j);
        cv::Point pgt1(currentGtBox.c, currentGtBox.r);
        cv::Point pgt2(currentGtBox.c + currentGtBox.w, currentGtBox.r + currentGtBox.h);
        int maxIOUBox = 0;
        double maxIOU = 0;
        for (int i = 0; i < n; i++) {
            cv::Rect rect = candidates[i].box;
            Box box;
            box.c = rect.x;
            box.r = rect.y;
            box.w = rect.width;
            box.h = rect.height;
            double currentIOU = boxesOverlap(currentGtBox, box);
            if (j == 0) {
                cv::Point p1(box.c, box.r);
                cv::Point p2(box.c + box.w, box.r + box.h);
                // if (showImage) {
                //   rectangle(img, p1, p2, Scalar(0, 0, 0)); // Show all rectangles
                // }
            }
            if (currentIOU > maxIOU) {
                maxIOU = currentIOU;
                maxIOUBox = i;
            }
        }
        totalIOU += maxIOU;
        cv::Rect finalBox = candidates[maxIOUBox].box;
        cv::Point p1(finalBox.x, finalBox.y);
        cv::Point p2(finalBox.x + finalBox.width, finalBox.y + finalBox.height);
        if (maxIOU >= minReqIOU) {
            correctBoxes++;
            if (showImage) {
                cv::rectangle(img, pgt1, pgt2, cv::Scalar(0, 255, 0));
                cv::rectangle(img, p1, p2, cv::Scalar(255, 0, 0));
            }
        }
        else {
            if (showImage) {
                cv::rectangle(img, pgt1, pgt2, cv::Scalar(0, 0, 255));
            }
        }
    }
    if (showImage) {
        cv::imshow("Boxed Image", img);
    }
    return correctBoxes;
}

void runVocEvaluation(const std::string modelPath, const EdgeBoxModel edgeBoxParam, const std::string annotationFolder, const std::string imgFolder, const float minReqIOU, const int showImage) {
    EdgeBox eb;
    eb.getEdgeModel(modelPath);
    eb.loadEdgeBoxPredeterminedParameters(edgeBoxParam);
    char folder[1024];
    strcpy(folder, annotationFolder.c_str());
    std::vector<std::string> allFiles;
    listFile(folder, allFiles);
    sort(allFiles.begin(), allFiles.end());
    std::string xmlFolder = annotationFolder + "/";
    float totalTime = 0;
    int totalImages = 0;
    int totalBoxes = 0;
    int totalCorrectBoxes = 0;
    int totalFiles = 0;
    std::vector<std::string> allImgFiles;
    std::vector<Boxes> allBBox;
    for (unsigned int i = 0; i < allFiles.size(); i++) {
        if (allFiles[i].substr( allFiles[i].length() - 4 ) != ".xml") {
            continue;
        }
        totalFiles++;
        pugi::xml_document doc;
        std::string xmlFileName = xmlFolder + allFiles.at(i);
        doc.load_file(xmlFileName.c_str());
        pugi::xml_node annotations = doc.child("annotation");
        Boxes gtBoxes;
        for (pugi::xml_node object = annotations.child("object"); object; object = object.next_sibling("object")) {
            Box newGtBox;
            int difficult = std::stoi(object.child_value("difficult"));
            if (difficult) {
                continue;
            }
            int xmin = std::stoi(object.child("bndbox").child_value("xmin"));
            int xmax = std::stoi(object.child("bndbox").child_value("xmax"));
            int ymin = std::stoi(object.child("bndbox").child_value("ymin"));
            int ymax = std::stoi(object.child("bndbox").child_value("ymax"));
            newGtBox.c = xmin;
            newGtBox.r = ymin;
            newGtBox.w = xmax - xmin;
            newGtBox.h = ymax - ymin;
            gtBoxes.push_back(newGtBox);
        }
        allBBox.push_back(gtBoxes);

        std::string imageName = imgFolder + '/' + allFiles.at(i).substr(0, 6) + ".jpg";
        allImgFiles.push_back(imageName);
    }
    for (int i = 0; i < totalFiles; i++) {
        std::string imageName = allImgFiles[i];
        Boxes gtBoxes = allBBox[i];
        std::cout << imageName << std::endl;
        auto t_start = std::chrono::high_resolution_clock::now();
        std::vector<Candidate> candidates;
        cv::Mat src = cv::imread(imageName, -1);
        if (!src.data) {
            std::cerr << "Could not open image file" << std::endl;
            exit(1);
        }
        eb.runEdgeBoxes(src, candidates);
        auto t_end = std::chrono::high_resolution_clock::now();
        double edgeBoxDuration = (std::chrono::duration<double, std::milli>(t_end - t_start).count()) / 1000;
        std::cout << "Total Time Taken: " << edgeBoxDuration << "s" << std::endl;
        totalTime += edgeBoxDuration;
        totalImages += 1;
        int correctBoxes = getIOU(src, candidates, gtBoxes, minReqIOU, showImage);
        totalCorrectBoxes += correctBoxes;
        totalBoxes += gtBoxes.size();
        std::cout << "Total Time Taken for " << totalImages << " images: " << totalTime << std::endl;
        std::cout << "Recall: " << totalCorrectBoxes << "/" << totalBoxes << std::endl << std::endl;
        if (showImage) {
            cv::waitKey(0);
        }
    }
    std::cout << "Total Time Taken for " << totalImages << " images: " << totalTime << std::endl;
    std::cout << "Recall: " << totalCorrectBoxes << "/" << totalBoxes << std::endl;
}

} // EdgeBox