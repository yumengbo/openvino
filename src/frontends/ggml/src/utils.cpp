#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/frontend/ggml/node_context.hpp"

namespace ov {
namespace frontend {
namespace ggml {

std::string getCurrentTime() {
    std::time_t now = std::time(nullptr);
    char buf[100];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return buf;
}

// void logToFile(const std::string& message) {
//     std::ofstream logFile;
//     std::string filename = "/home/yumeng/Code/ov-ggml-frontend/openvino/build/log.txt";
//     logFile.open(filename, std::ios_base::app); 
//     if (logFile.is_open()) {
//         logFile << "[" << getCurrentTime() << "] " << message << std::endl;
//         logFile.close();
//     } else {
//         std::cerr << "Unable to open log file: " << filename << std::endl;
//     }
// }

// void dump_ov_model (const std::shared_ptr<ov::Model> model) {
//     logToFile("========= Model =========");
//     logToFile(model->get_friendly_name());

//     for (const auto& node : model->get_ops()) {
//         logToFile("=== Node ====");
//         logToFile(node->get_friendly_name());
//         logToFile("Type:");
//         logToFile(node->get_type_name());
//         for (const auto& input : node->inputs()) {
//             logToFile("    Input Shape: ");
//             logToFile(input.get_shape().to_string());
//         }
//         for (const auto& output : node->outputs()) {
//             logToFile("    Output Shape: ");
//             logToFile(output.get_shape().to_string());
//         }
//     }
// }

void num_inputs_check(const NodeContext& context, size_t min_inputs, size_t max_inputs) {
    auto input_size = context.get_input_size();
    FRONT_END_OP_CONVERSION_CHECK(input_size >= min_inputs, "Got less inputs than expected");
    FRONT_END_OP_CONVERSION_CHECK(input_size <= max_inputs, "Got more inputs than expected");
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov

