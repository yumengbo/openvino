#pragma once

#include "openvino/frontend/node_context.hpp"
#include "openvino/frontend/ggml/decoder.hpp"

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type()),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_translate_session(translate_session) {}

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    size_t get_input_size() const override {
        return m_decoder->get_input_size();
    }

    Any get_input_type(size_t index) const {
        return m_decoder->get_input_type(index);
    }

    PartialShape get_input_shape(size_t index) const {
        return m_decoder->get_input_shape(index);
    }

    Output<Node> get_input(int idx) const override;

    Output<Node> get_input(const std::string& name) const override {
        return m_tensor_map->at(name);
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name();
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    TranslateSession* m_translate_session;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
