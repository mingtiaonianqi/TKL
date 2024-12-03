#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <npy.hpp>



extern "C" {
    torch::jit::script::Module loadModel(const std::string& modelPath);
    torch::Tensor callModel(torch::jit::script::Module& module, std::string input_data_path);
    void releaseModel(torch::jit::script::Module& module);

    torch::jit::script::Module loadModel(const std::string& modelPath) {
    try {
        torch::jit::script::Module module = torch::jit::load(modelPath);
        return module;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        throw;
    }
    }

    torch::Tensor callModel(torch::jit::script::Module& module, std::string input_data_path) {
        try {
            npy::npy_data<_Float32> data_np = npy::read_npy<_Float32>(input_data_path);
    
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
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(tensor_gpu); 
            auto output_tuple = module.forward(inputs).toTuple();
            torch::Tensor output = output_tuple->elements()[0].toTensor();
            std::cout << "output shape :"<<output.sizes()<<std::endl;
            
            return output;
        } catch (const c10::Error& e) {
            std::cerr << "Error calling the model: " << e.what() << std::endl;
            throw;
        }
    }

    void releaseModel(torch::jit::script::Module& module) {
        // 这里可以进行一些清理资源的操作，目前LibTorch会自动管理一些内存
        // 但如果有其他相关资源需要清理，可以在这里添加代码
        module.~Module();
    }
}