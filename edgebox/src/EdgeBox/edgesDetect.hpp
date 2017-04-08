/*
 * edgesDetect.hpp
 *
 *  Created on: May 16, 2016
 *      Author: patcho
 */

#ifndef EDGESDETECT_HPP_
#define EDGESDETECT_HPP_

struct RandomForest;

namespace EdgeBox
{

void edgesDetect(RandomForest* model, int height, int width, int depth, float* image, float* chns, float* chnsSs, float* edge, unsigned int* ind);

} // EdgeBox



#endif /* EDGESDETECT_HPP_ */
