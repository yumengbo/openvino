#include "openvino/op/add.hpp"

#include "openvino/frontend/ggml/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

// Not used at present
OutputVector translate_add_common(const NodeContext& context, bool inplace) {
    num_inputs_check(context, 2, 3);
    Output<Node> lhs;
    Output<Node> rhs;
    auto dtype0 = context.get_input_type(0);
    auto dtype1 = context.get_input_type(1);
    if (inplace) {
        lhs = context.get_input(0);
        rhs = context.get_input(1);
    } else {
        lhs = context.get_input(0);
        rhs = context.get_input(1);
        // std::tie(lhs, rhs) = get_inputs_with_promoted_types(context, 0, 1);
    }

    auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);
    return {add};
}

OutputVector translate_add(const NodeContext& context) {
    return translate_add_common(context, false);
};

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov