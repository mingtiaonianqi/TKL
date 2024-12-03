#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>


extern "C" {
    torch::jit::script::Module loadModel(const std::string& modelPath);
    torch::Tensor callModel(torch::jit::script::Module& module, std::string input_data_path);
    void releaseModel(torch::jit::script::Module& module);
}
int main(){
    std::string modelPath = "/DLL/pointnet2-sem-seg-changed/best_model.pt";
    torch::jit::script::Module model = loadModel(modelPath);
    model.to(torch::kCUDA);
    std::string input_data_path = "/DLL/pointnet2-sem-seg-changed/save_data.npy";
    torch::Tensor output = callModel(model,input_data_path);
    std::cout << output[0][10] <<std::endl;
    releaseModel(model);
    return 0;
}