#pragma once

#include "cuda_vector.hpp"
#include "cuda_error.hpp"

namespace cuda {

//constructor
template <typename T>
vector<T>::vector(unsigned numElems)
        : elem(0), sz{numElems}
{
    error err = cudaMallocManaged(&elem, sz * sizeof(T));
    err.update();
}

//destructor
template <typename T>
vector<T>::~vector()
{
    error err = cudaFree(elem);
    err.update();
}

//copy constructor
template <typename T>
vector<T>::vector(const vector<T>& v)
        : elem(0), sz{v.sz}
{
    error err = cudaMallocManaged(&elem, sz);
}

//copy assignment
template <typename T>
vector<T>& vector<T>::operator=(const vector<T>& v)
{
    T* p;
    error err;
    err = cudaMallocManaged(&p, v.sz * sizeof(T));
    for (unsigned i = 0; i < v.sz; ++i)
        p[i] = v.elem[i];
    err  = cudaFree(elem);
    elem = p;
    sz   = v.sz;
    return *this;
}

//move constructor
template <typename T>
vector<T>::vector(vector<T>&& v)
        : elem{v.elem}, sz{v.sz}
{
    v.elem = nullptr;
    v.sz   = 0;
}

//move assignment
template <typename T>
vector<T>& vector<T>::operator=(vector<T>&& v)
{
    elem = v.elem;
    sz   = v.sz;
    v.elem = nullptr;
    v.sz   = 0;
    return *this;
}


template <typename T>
HOST_DEVICE_CALLABLE_INLINE
T& vector<T>::operator[](unsigned i)
{
    return elem[i];
}


template <typename T>
HOST_DEVICE_CALLABLE_INLINE
const T& vector<T>::operator[](unsigned i) const
{
    return elem[i];
}

//To support range-for loop we must define begin() and end()
template <typename T>
HOST_DEVICE_CALLABLE
T* vector<T>::begin()
{
    return &elem[0];
}

template <typename T>
HOST_DEVICE_CALLABLE
T* vector<T>::end()
{
    return &elem[sz];
}

template <typename T>
HOST_DEVICE_CALLABLE
unsigned vector<T>::size() const
{
    return sz;
}

}