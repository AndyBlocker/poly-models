#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__

#include <vector>
#include <stdexcept>
#include <string>

template<typename T>
class Tensor {
public:
    Tensor();

    explicit Tensor(const std::vector<int>& shape);

    // Tensor(int n, int c, int h, int w);

    const std::vector<int>& shape() const;

    int size(int dim) const;

    int total_size() const;

    T* data();
    const T* data() const;

    T& operator[](int idx);
    const T& operator[](int idx) const;

    T& at4d(int n, int c, int h, int w);
    const T& at4d(int n, int c, int h, int w) const;

private:
    std::vector<int> shape_;
    std::vector<T> data_;
};

#endif