#include <iostream>
#include <string>
#include <npy.hpp>
#include <vector>
#include <torch/torch.h>
int main(){
    
    std::string file_path = "save_data.npy";
    
    npy::npy_data<_Float32> data_np = npy::read_npy<_Float32>(file_path);
    
    std::vector<_Float32> data_vector = data_np.data;
    std::vector<unsigned long> data_shape_UnLo = data_np.shape;
    std::cout << "vector shape :"<<data_shape_UnLo<<std::endl;
    std::vector<long> data_shape_Lo;

    size_t dim_num = data_shape_UnLo.size();
    torch::Tensor tensor = torch::from_blob(data_vector.data(), {static_cast<long>(data_vector.size())}, torch::kFloat32);
    for (size_t i = 0; i < dim_num; i++)
    {
        data_shape_Lo.push_back(static_cast<long>(data_shape_UnLo[i]));   
    }
    
    tensor.resize_({data_shape_Lo});
    std::cout << "tensor shape :"<<tensor.sizes()<<std::endl;
    torch::Device device(torch::kCUDA);
    torch::Tensor tensor_gpu = tensor.to(device);
    // std::cout << tensor_gpu << std::endl;
    return 0;
}