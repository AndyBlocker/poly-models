#include "common/tensor.hpp"
#include <numeric>
#include <stdexcept>
#include <cmath>
#include <iostream>

template<typename T>
Tensor<T>::Tensor()
{
}

template<typename T>
Tensor<T>::Tensor(const std::vector<int>& shape)
    : shape_(shape)
{
    if(shape_.empty()) {
        throw std::runtime_error("Tensor shape cannot be empty.");
    }
    int total = 1;
    for(auto dim : shape_) {
        if(dim <= 0) {
            throw std::runtime_error("Tensor shape dimension must be positive.");
        }
        total *= dim;
    }
    data_.resize(total, static_cast<T>(0));
}

// template<typename T>
// Tensor<T>::Tensor(int n, int c, int h, int w)
// {
//     shape_ = {n, c, h, w};
//     if(n<=0 || c<=0 || h<=0 || w<=0) {
//         throw std::runtime_error("Tensor dimensions must be positive.");
//     }
//     data_.resize(n*c*h*w, static_cast<T>(0));
// }

template<typename T>
const std::vector<int>& Tensor<T>::shape() const {
    return shape_;
}

template<typename T>
int Tensor<T>::size(int dim) const {
    return shape_.at(dim);
}

template<typename T>
int Tensor<T>::total_size() const {
    return static_cast<int>(data_.size());
}

template<typename T>
T* Tensor<T>::data() {
    return data_.data();
}

template<typename T>
const T* Tensor<T>::data() const {
    return data_.data();
}

template<typename T>
T& Tensor<T>::operator[](int idx) {
    return data_[idx];
}

template<typename T>
const T& Tensor<T>::operator[](int idx) const {
    return data_[idx];
}

template<typename T>
T& Tensor<T>::at4d(int n, int c, int h, int w) {
    int C = shape_[1];
    int H = shape_[2];
    int W = shape_[3];
    int index = ((n*C + c)*H + h)*W + w;
    return data_[index];
}

template<typename T>
const T& Tensor<T>::at4d(int n, int c, int h, int w) const {
    int C = shape_[1];
    int H = shape_[2];
    int W = shape_[3];
    int index = ((n*C + c)*H + h)*W + w;
    return data_[index];
}

template class Tensor<float>;
template class Tensor<double>;
template class Tensor<int>;
template class Tensor<unsigned char>;