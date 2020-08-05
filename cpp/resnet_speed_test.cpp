#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

#include <iostream>
#include <memory>
#include <chrono>

int main(int argc, const char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
        module.to(at::kCUDA);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "ok\n";

    // Create a vector of inputs.
    torch::jit::IValue inp = torch::rand({64, 3, 224, 224}).cuda();
    std::vector<torch::jit::IValue> inputs;

    inputs.push_back(inp);

    auto start = std::chrono::steady_clock::now();
    // Execute the model and turn its output into a tensor.
    for (size_t i = 0; i < 1000; i++)
    {
        at::Tensor output = module.forward(inputs).toTensor();
    }
    auto end = std::chrono::steady_clock::now();
    
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "time consuming: "
              << time_used.count()
              << '\n';

    /**
     * Libtorch run pytorch Resnet34 model in 1000 loops:
     *  1 Bat:  721 Mib,   5.70 sec
     * 32 Bat: 1781 Mib,  84.04 sec
     * 64 Bat: 2799 Mib, 170.60 Sec
     */
}