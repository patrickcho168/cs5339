/*
 * findEdges.hpp
 *
 *  Created on: May 16, 2016
 *      Author: patcho
 */

#ifndef FINDEDGES_HPP_
#define FINDEDGES_HPP_
#include <string>

struct RandomForest;

namespace EdgeBox
{

void loadBinaryModel( RandomForest* __rf, std::string dataFile );
void findEdges(RandomForest* __rf, cv::Mat nSrc, float* edgeFinal, float* orientFinal);

} // EdgeBox

#endif /* FINDEDGES_HPP_ */
