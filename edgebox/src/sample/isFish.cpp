#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <vector>
#include <dirent.h>
#include "../EdgeBox/edgeBoxes.hpp"
#include "json/json.h"
#include "json/json-forwards.h"

typedef struct { int c, r, w, h; float s; } Box;

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

void PrintJSONValue( const Json::Value &val )
{
    if( val.isString() ) {
        printf( "string(%s)", val.asString().c_str() ); 
    } else if( val.isBool() ) {
        printf( "bool(%d)", val.asBool() ); 
    } else if( val.isInt() ) {
        printf( "int(%d)", val.asInt() ); 
    } else if( val.isUInt() ) {
        printf( "uint(%u)", val.asUInt() ); 
    } else if( val.isDouble() ) {
        printf( "double(%f)", val.asDouble() ); 
    }
    else 
    {
        printf( "unknown type=[%d]", val.type() ); 
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

bool PrintJSONTree( const Json::Value &root, unsigned short depth /* = 0 */) 
{
    depth += 1;
    printf( " {type=[%d], size=%d}", root.type(), root.size() ); 

    if( root.size() > 0 ) {
        printf("\n");
        for( auto itr = root.begin() ; itr != root.end() ; itr++ ) {
            // Print depth. 
            for( int tab = 0 ; tab < depth; tab++) {
               printf("-"); 
            }
            printf(" subvalue(");
            PrintJSONValue(itr.key());
            printf(") -");
            PrintJSONTree( *itr, depth); 
        }
        return true;
    } else {
        printf(" ");
        PrintJSONValue(root);
        printf( "\n" ); 
    }
    return true;
}

int main(int argc, char** argv) {

    std::string bbFolder = "../../BB/";
    std::vector<std::string> fishTypes;
    fishTypes.push_back("ALB_BB.txt");
    fishTypes.push_back("BET_BB.txt");
    fishTypes.push_back("DOL_BB.txt");
    fishTypes.push_back("LAG_BB.txt");
    // fishTypes.push_back("NoF_BB.txt");
    // fishTypes.push_back("OTHER_BB.txt");
    fishTypes.push_back("SHARK_BB.txt");
    fishTypes.push_back("YFT_BB.txt");

    std::string annoFolder = "../../annotations/";
    std::vector<std::string> annotations;
    annotations.push_back("alb_labels.json");
    annotations.push_back("bet_labels.json");
    annotations.push_back("dol_labels.json");
    annotations.push_back("lag_labels.json");
    annotations.push_back("shark_labels.json");
    annotations.push_back("yft_labels.json");
    // fishTypes.push_back("YFT_BB.txt");

    for (unsigned int j=0; j<fishTypes.size(); ++j) {
        std::string nextFile = bbFolder + fishTypes[j];
        std::cout << nextFile << std::endl;
        Json::Value root;

        std::ifstream ifile(nextFile, std::ifstream::binary);
        ifile >> root;

        std::string nextAnnoFile = annoFolder + annotations[j];
        Json::Value annoRoot;

        std::ifstream afile(nextAnnoFile, std::ifstream::binary);
        afile >> annoRoot;

        for( auto itr = root.begin() ; itr != root.end() ; itr++ ) {
            for( auto itr2 = annoRoot.begin() ; itr2 != annoRoot.end() ; itr2++ ) {
                if ((*itr)["filename"] == (*itr2)["filename"] || "../data/train/SHARK/" + (*itr)["filename"].asString() == (*itr2)["filename"].asString() || "../data/train/YFT/" + (*itr)["filename"].asString() == (*itr2)["filename"].asString()) {
                    auto rootAnno = (*itr)["annotations"];
                    auto annoRootAnno = (*itr2)["annotations"];
                    // float biggest = 0.0;
                    for ( auto itr3 = rootAnno.begin(); itr3 != rootAnno.end(); itr3++) {
                        auto nextRootAnno = (*itr3);
                        float largestIou = 0.0;
                        Box proposal;
                        proposal.c = nextRootAnno["x"].asDouble();
                        proposal.r = nextRootAnno["y"].asDouble();
                        proposal.w = nextRootAnno["width"].asDouble();
                        proposal.h = nextRootAnno["height"].asDouble();
                        for ( auto itr4 = annoRootAnno.begin(); itr4 != annoRootAnno.end(); itr4++) {
                            // biggest overlap
                            auto nextAnnoRootAnno = (*itr4);
                            Box groundTruth;
                            groundTruth.c = nextAnnoRootAnno["x"].asDouble();
                            groundTruth.r = nextAnnoRootAnno["y"].asDouble();
                            groundTruth.w = nextAnnoRootAnno["width"].asDouble();
                            groundTruth.h = nextAnnoRootAnno["height"].asDouble();
                            float overlap = boxesOverlap(groundTruth, proposal);
                            if (overlap > largestIou) {
                                largestIou = overlap;
                            }
                        }
                        // if (largestIou > biggest) {
                        //     biggest = largestIou;
                        // }
                        // std::cout << largestIou << std::endl;
                        root[itr.key().asInt()]["annotations"][itr3.key().asInt()]["iou"] = largestIou;
                    }
                    // std::cout << biggest << std::endl;
                    break;
                }
            }
        }

        std::ofstream file_id;
        file_id.open(std::to_string(j) + ".txt");

        file_id << root;

        file_id.close();
    
    }

    return 0;
}

