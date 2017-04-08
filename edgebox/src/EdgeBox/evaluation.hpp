/*
 * edgeBoxes.hpp
 *
 *  Created on: May 16, 2016
 *      Author: patcho
 */

#ifndef EVALUATION_HPP_
#define EVALUATION_HPP_
#include <string>
#include "randomForest.hpp"
#include "edgeBoxes.hpp"

namespace EdgeBox
{

void runVocEvaluation(const std::string modelPath, const EdgeBoxModel edgeBoxParam, const std::string annotationFolder, const std::string imgFolder, const float minReqIOU, const int showImage);

} // EdgeBox

#endif /* EVALUATION_HPP_ */
