# distutils: language = c++
# distutils: sources = CppClass.cpp

# Cython interface file for wrapping the object
#
#

from libcpp.vector cimport vector

# c++ interface to cython
cdef extern from "CppClass.h" namespace "cc":
  cdef cppclass CppClass:
        CppClass()
        vector[vector[vector[vector[vector[float]]]]] frexp_ret(vector[vector[vector[vector[float]]]])

# creating a cython wrapper class
cdef class PyCppClass:
    cdef CppClass *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.thisptr = new CppClass()
    def __dealloc__(self):
        del self.thisptr
    def frexp_ret(self, sv):
        return self.thisptr.frexp_ret(sv)