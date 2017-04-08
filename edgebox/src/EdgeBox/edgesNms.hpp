/*
 * edgesNms.hpp
 *
 *  Created on: May 16, 2016
 *      Author: patcho
 */

#ifndef EDGESNMS_HPP_
#define EDGESNMS_HPP_

namespace EdgeBox
{

void edgesNms( float* edgeSrc, float * edgeDst, float* orient, int r, int s, float m, int h, int w, int nThreads );

} // EdgeBox

#endif /* EDGESNMS_HPP_ */
