#EdgeBoxes C++

```
@inproceedings{ZitnickECCV14edgeBoxes,
    author    = {C. Lawrence Zitnick and Piotr Doll\'ar},
    title     = {Edge Boxes: Locating Object Proposals from Edges},
    booktitle = {ECCV},
    year      = {2014},
}
```

## 1. Introduction

This is a C++ implementation of the EdgeBox algorithm based on Zitnick and Dollar's [paper](http://web.bii.a-star.edu.sg/~zhangxw/files/EdgeBoxes_ECCV2014.pdf). There are two main components:

### Structured Edge Detection

For Structured Edge Detection, a Random Forest model is required. This model can be trained on MatLab. Follow instructions [here](https://github.com/pdollar/edges) for more details on how to train your own model. The same site provides a pretrained model that we have used here. Three scripts have also been provided (modelConvert.m, modelConvertDemo.m, convertToBinary.cpp) to convert the trained model from .mat into .yml and .bin format.

Structured Edge Detection can be used independent of EdgeBox as an alternative edge detection algorithm. See function runEdgeDetection in edgeBoxes.cpp for more details. An important parameter in SE detection is the sharpening parameter. Set to 0 for higher speed and set to 2 for sharper edges. See SE [paper](http://arxiv.org/pdf/1406.5549.pdf) for more details. Pertinently, the multi-scale version of the SE detection is not implemented here since this version slows down the algorithm and does not improve EdgeBox results. Adding this functionality can be done easily if needed.

We briefly describe here the SE algorithm. The Random Forest consists of 8 Random Trees, 4 of which are used at test time to predict 16x16 binary edge maps from 32x32 image patches. The 32x32 image patch is first expanded to 13 channels and then downsampled by a factor of 2. This gives a total of 32x32x13/4=3328 features. Another set of features, called the self-similarity features, are derived by downsampling each channel to a 5x5 dimension and taking the difference between each pair of pixels within each channel. This leads to another (25C2)x13=3900 features. Adding the two together gives a grand total of 7228 features. At each node, a single feature compared with a threshold to determine whether to go to the left or the right of the Random Tree. At the leaf of the tree, the predicted binary edge map is found. Given an image, the 32x32 image patch is used with a stride of 2. Hence, each pixel gets 16x16xnumberOfTrees/(stride^2)=16x16x4/4=256 votes on whether the pixel is an edge pixel.

During training, the feature and threshold at each node of each tree needs to be determined. Firstly, not all features are tested at each node. This adds the required randomness in the trees to ensure that the trees remain de-correlated. Additionally, we need a criteria to decide what split works best. The information gain formula, unfortunately, does not go well with structured output like binary edge maps. Instead, the similarity of two pairs of pixels is used as an intermediate output. However, this gives 256C2 output features which is too large. Instead, only 256 of such output features are chosen which are then further decreased to 5 dimensions through PCA. Finally, these 5 dimensions are used to cluster outputs into 2 categories and the information gain is then calculated.

### EdgeBoxes

For EdgeBox, there are many more parameters. The most pertinent ones are:
A. alpha (usually set to 0.65 unless desired IOU is more than 0.7)
B. beta (usually set to desiredIOU+0.05)
C. minScore (this is the minimum score that the a box proposal needs to have before it is refined. Hence, higher minScore results in faster algorithm. Usually set to 0.01. Set to 0.02 for faster algorithm.)
D. sharpen (set to 2 for higher recall and set to 0 for faster algorithm. Can set to 1 also but this setting has not been tested.).
E. edgeMergeThr (default=0.5. Increase for speed up.)
F. clusterMinMag (default=0.5. Increase for speed up.)
G. maxAspectRatio (1/maxAspectRatio <= width/height of window <= maxAspectRatio. Set to 1 if only square boxes are allowed. Set sameWidthHeight as one if you do not want to refine the box).
H. minBoxArea (minimum window size)
I. maxBoxArea (maximum window size. Set to 0 if no max size.)

In particular, 4 settings have been recommended by the [paper](http://web.bii.a-star.edu.sg/~zhangxw/files/EdgeBoxes_ECCV2014.pdf) as they give good recall results for VOC2007 images. They are:
A. EdgeBoxesFast (50% desired IOU, 0.1s/img)
B. EdgeBoxes50   (50% desired IOU, 0.25s/img)
C. EdgeBoxes70   (70% desired IOU, 0.25s/img)
D. EdgeBoxes90   (90% desired IOU, 2.5s/img)

We also briefly describe the EdgeBoxes algorithm here. Given the edge magnitude and orientation from the SE detection, we compute an affinity between every 2 edge pixels with magnitude higher than a predefined threshold. Affinity between any 2 pixels that are further than 2 pixels apart is 0. Otherwise, it is calculated using formula 1 in the paper. Intuitively, if the orientations of the edges are close to the angle made between the edges, then these edges lie in roughly a straight line and have high affinity. Else, the pixels have low affinity. For each bounding box, a score is given based on formulas 2,3 and 4. Intuitively, if all edges are tightly enclosed by the bounding box, a high score is given. This happens if the edge pixels within the bounding box have little affinity with edge pixels outside the bounding box.

## 2. Compile 

```
mkdir build && cd build
cmake ../
make
```

## 3. Usage

```
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "../EdgeBox/edgeBoxes.hpp" // Header file can be found in edgebox directory

int main() {
    EdgeBox::EdgeBox eb;
    // Get SE Binary Model
    std::string seFile = "../data/data.dat";
    eb.getEdgeModel(seFile); // get from edgebox directory
    
    // Only needed if doing EdgeBoxes
    // Either load predetermined parameters for edgebox
    // EDGEBOXMODEL_FAST for EdgeBoxFast, EDGEBOXMODEL_50 for EdgeBox50, EDGEBOXMODEL_70 for EDGEBOXMODEL_90, 4 for EdgeBox90
    eb.loadEdgeBoxPredeterminedParameters(EdgeBox::EDGEBOXMODEL_50);
    // Or load own parameters for edgebox
    eb.loadEdgeBoxManualParameters("../data/config.txt");
    
    // Load Image
    cv::Mat src = imread("abc.png",-1);
    
    // Only Run Edge Detection
    cv::Mat Edge, Orientation;
    eb.runEdgeDetection(src,Edge,Orientation);
    
    // Run EdgeBoxes
    std::vector<EdgeBox::Candidate> candidates;
    eb.runEdgeBoxes(src, candidates);
    
    // Visualize EdgeBoxes
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
    waitKey(0);
    cv::destroyAllWindows();
    
    // Run algo on VOC dataset: http://pjreddie.com/projects/pascal-voc-dataset-mirror/
    std::string annotationsFolder = "Annotations";
    std::string jpgImagesFolder = "JPGImages";
    int showImage = 0; // change to 1 to see images
    eb.runVocEvaluation(seFile, EdgeBox::EDGEBOXMODEL_50, annotationsFolder, jpgImagesFolder, showImage);
    return 0;
}
```


## 4. Evaluation Results and Comparison With Paper

| EdgeBox Type  | Number of Threads | Desired IOU   | Recall (Paper)       | Time per image (Paper) |
|:-------------:|:-----------------:|:-------------:|:--------------------:|:----------------------:|
| EdgeBoxFast   | 1                 | 0.5           | 93.6% (not reported) | 0.21s (not reported)   |
| EdgeBoxFast   | 4                 | 0.5           | 93.6% (not reported) | 0.09s (0.09s)          |
| EdgeBox50     | 4                 | 0.5           | 96.3% (96%)          | 0.25s (0.25s)          |
| EdgeBox70     | 4                 | 0.7           | 77.9% (76%)          | 0.24s (0.25s)          |
| EdgeBox90     | 4                 | 0.9           | 27.0% (28%)          | 3.48s (2.5s)           |

Tested using 3.00GHz Intel Core i7-5960X (octacore) Ubuntu 3.16.0 16GB RAM

EdgeBox90 only tested on 340 images. Others tested on 9963 images.

## 5. Queries?

Please email Patrick (patcho168-at-gmail.com).
