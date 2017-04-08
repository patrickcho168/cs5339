/*
 * wrappers.hpp
 *
 *  Created on: May 5, 2016
 *      Author: patcho
 */

/*******************************************************************************
* Piotr's Computer Vision Matlab Toolbox      Version 3.00
* Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
* Licensed under the Simplified BSD License [see external/bsd.txt]
*******************************************************************************/
#include <cstring>
#include <cstdlib>
#include "wrappers.hpp"

namespace EdgeBox
{

namespace internal
{

// platform independent aligned memory allocation (see also alFree)
void* alMalloc( size_t size, int alignment ) {
    const size_t pSize = sizeof(void*), a = alignment - 1;
    void *raw = wrMalloc(size + a + pSize);
    void *aligned = (void*) (((size_t) raw + pSize + a) & ~a);
    *(void**) ((size_t) aligned - pSize) = raw;
    return aligned;
}

// platform independent alignned memory de-allocation (see also alMalloc)
void alFree(void* aligned) {
    void* raw = *(void**)((char*)aligned - sizeof(void*));
    wrFree(raw);
}

} // internal

} // EdgeBox
