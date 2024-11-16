#include "input_model.hpp"
#include "place.hpp"
#include "openvino/frontend/ggml/decoder.hpp"

namespace ov {
namespace frontend {
namespace ggml {

InputModel::InputModel(const std::shared_ptr<GgmlGraphIterator>& graph_iterator) 
    : m_graph_iterator(graph_iterator) {
    // const auto& inputs = m_model_decoder->inputs();
    // for (size_t i = 0; i < inputs.size(); ++i) {
    //     auto in_place = std::make_shared<ggml::Place>(*this, inputs[i]);
    //     m_name_to_place.emplace(std::to_string(inputs[i]), std::dynamic_pointer_cast<frontend::Place>(in_place));
    //     for (const auto &name : in_place->get_names()) {
    //         m_name_to_place.emplace(name, std::dynamic_pointer_cast<frontend::Place>(in_place));
    //     }
    //     m_inputs.push_back(in_place);
    // }
    // const auto& outputs = m_model_decoder->outputs();
    // for (size_t i = 0; i < outputs.size(); ++i) {
    //     auto out_place = std::make_shared<ggml::Place>(*this, outputs[i]);
    //     m_name_to_place.emplace(std::to_string(outputs[i]), std::dynamic_pointer_cast<frontend::Place>(out_place));
    //     for (const auto &name : out_place->get_names()) {
    //         m_name_to_place.emplace(name, std::dynamic_pointer_cast<frontend::Place>(out_place));
    //     }
    //     m_outputs.push_back(out_place);
    // }
}

std::vector<ov::frontend::Place::Ptr> InputModel::get_inputs() const {
    return m_inputs;
}

std::vector<ov::frontend::Place::Ptr> InputModel::get_outputs() const {
    return m_outputs;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
