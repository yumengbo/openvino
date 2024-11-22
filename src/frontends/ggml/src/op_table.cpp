#include "op_table.hpp"
#include "utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"

using namespace ov::op;
namespace ov {
namespace frontend {
namespace ggml {
namespace op {

#define GGML_OP_CONVERTER(op) OutputVector op(const NodeContext& node)

GGML_OP_CONVERTER(translate_add);

}  // namespace op

const std::unordered_map<std::string, CreatorFunction> get_supported_ops() {
    return {
        // {"GGML_OP_ADD", CreatorFunction(op::translate_add)}
        {"GGML_OP_ADD", op::translate_1to1_match_2_inputs<v1::Add>},
        // {"GGML_OP_ADD1", op::translate_1to1_match_2_inputs<v1::Add>},
        {"GGML_OP_DIV", op::translate_1to1_match_2_inputs<v1::Divide>},
        {"GGML_OP_MUL", op::translate_1to1_match_2_inputs<v1::Multiply>},
        {"GGML_OP_SUB", op::translate_1to1_match_2_inputs<v1::Subtract>}
        
    };
};


}  // namespace ggml
}  // namespace frontend
}  // namespace ov