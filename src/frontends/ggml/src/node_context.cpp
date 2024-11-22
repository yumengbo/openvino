#include "openvino/frontend/ggml/node_context.hpp"
#include "translate_session.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {

using namespace ov::op;

Output<Node> NodeContext::get_input(int idx) const {
    auto input_name = m_decoder->get_input_name(idx);
    return m_tensor_map->at(input_name);
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
