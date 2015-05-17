#pragma once

#include <cstdlib>
#include "cuda_managed.hpp"
#include "device_callable.hpp"

namespace cuda {

template <typename T>
class vector : public managed {
public:
    vector(unsigned numElems);
    ~vector();

    vector(const vector<T> &v);
    vector<T> &operator=(const vector<T>& v);

    vector(vector<T>&& v);
    vector<T> &operator=(vector<T>&& v);

    HOST_DEVICE_CALLABLE_INLINE       T& operator[](unsigned i);
    HOST_DEVICE_CALLABLE_INLINE const T& operator[](unsigned i) const;

    HOST_DEVICE_CALLABLE T* begin();
    HOST_DEVICE_CALLABLE T* end();

    HOST_DEVICE_CALLABLE unsigned size() const;

private:
    T* elem;
    unsigned sz;
};

}

#include "cuda_vector-impl.hpp"

