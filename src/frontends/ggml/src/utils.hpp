#pragma once

#include "openvino/frontend/ggml/node_context.hpp"

namespace ov {
namespace frontend {
namespace ggml {

void dump_ov_model (const std::shared_ptr<ov::Model> model) ;

// void logToFile(const std::string& message);

void num_inputs_check(const NodeContext& context, size_t min_inputs, size_t max_inputs);

namespace op {
template <typename T>
OutputVector translate_1to1_match_2_inputs(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    return {std::make_shared<T>(context.get_input(0), context.get_input(1))};
}
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov

