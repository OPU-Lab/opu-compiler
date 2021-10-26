
/*
Passing variables / arrays between cython and cpp
Example from 
http://docs.cython.org/src/userguide/wrapping_CPlusPlus.html
Adapted to include passing of multidimensional arrays
*/

#include <vector>
#include <cmath> 
#include <iostream>

namespace cc {
    class CppClass {
    public:
        CppClass();
        ~CppClass();
        std::vector< std::vector< std::vector< std::vector< std::vector<float> > > > > frexp_ret(std::vector< std::vector< std::vector< std::vector<float> > > > sv);
    };
}