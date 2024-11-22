#include "openvino/op/reshape.hpp"

#include <cstdint>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/frontend/ggml/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_reshape(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    Output<Node> input = context.get_input(0);
    auto pshape = context.get_input_shape(0);
    auto new_shape_node = std::make_shared<ov::op::v0::Constant>(element::i64, pshape.to_shape());
    Output<Node> res = std::make_shared<ov::op::v1::Reshape>(input, new_shape_node, false);
    return {res};
};


}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov