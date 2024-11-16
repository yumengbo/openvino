#pragma once

#include "openvino/frontend/node_context.hpp"
#include "openvino/frontend/ggml/decoder.hpp"

namespace ov {
namespace frontend {
namespace ggml {

class NodeContext : public frontend::NodeContext {

private:
    std::shared_ptr<GgmlDecoder> m_decoder;

};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
