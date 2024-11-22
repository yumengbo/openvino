#include "input_model.hpp"
#include "place.hpp"
#include "utils.hpp"
#include "openvino/frontend/ggml/decoder.hpp"
#include "openvino/util/log.hpp"

namespace ov {
namespace frontend {
namespace ggml {

InputModel::InputModel(const std::shared_ptr<GgmlGraphIterator>& graph_iterator) 
    : m_graph_iterator(graph_iterator) {
    load_places();
    // dump_input_model();
}

void InputModel::load_places() {
    std::set<std::string> all_op_names;
    size_t index = 0;

    m_inputs.clear();
    m_outputs.clear();
    for (; !m_graph_iterator->is_end(); m_graph_iterator->next()) {
        auto node_decoder = m_graph_iterator->get_decoder();
        auto op_name = node_decoder->get_op_name();
        auto op_type = node_decoder->get_op_type();

        auto op_place = std::make_shared<OpPlace>(*this, std::dynamic_pointer_cast<GgmlDecoder>(node_decoder));
        all_op_names.insert(op_name);
        m_op_places.push_back(op_place);
        m_op_places_map[op_name] = op_place;

        // Get model inputs
        // Get input from the first OP
        // Get input with flag GGML_TENSOR_FLAG_INPUT
        auto input_tensor_vector = op_place->get_input_tensor();
        for (size_t i = 0; i < op_place->get_input_tensor().size(); i++) {
            if (index == 0 || input_tensor_vector[i]->is_model_input()) {
                m_inputs.push_back(std::dynamic_pointer_cast<frontend::Place>(input_tensor_vector[i]));
            }
        }

        // Get model outputs
        // Get input from the last OP
        // Get output with flag GGML_TENSOR_FLAG_OUTPUT
        auto output_tensor_vector = op_place->get_output_tensor();
        for (size_t i = 0; i < op_place->get_output_tensor().size(); i++) {
            if (index == m_graph_iterator->size() - 1 || output_tensor_vector[i]->is_model_output()) {
                m_outputs.push_back(std::dynamic_pointer_cast<frontend::Place>(output_tensor_vector[i]));
            }
        }
        
        index++;
    }
}

std::vector<ov::frontend::Place::Ptr> InputModel::get_inputs() const {
    return m_inputs;
}

std::vector<ov::frontend::Place::Ptr> InputModel::get_outputs() const {
    return m_outputs;
}

std::vector<std::shared_ptr<OpPlace>> InputModel::get_op_places() const {
    return m_op_places;
}

// void InputModel::dump_input_model() const {
//     logToFile("====== Graph ======");
//     for (size_t i = 0; i < m_op_places.size(); i++) {
//         logToFile("OP: ");
//         logToFile(m_op_places[i]->get_operation_name());
//     }
//     for (size_t i = 0; i < m_inputs.size(); i++) {
//         logToFile("Input shape: ");
//         logToFile(std::dynamic_pointer_cast<TensorPlace>(m_inputs[i])->get_partial_shape().to_string().c_str());
//         logToFile("Input type:");
//         logToFile(std::dynamic_pointer_cast<TensorPlace>(m_inputs[i])->get_element_type().to_string().c_str());
//         logToFile("Input name: ");
//         logToFile(std::dynamic_pointer_cast<TensorPlace>(m_inputs[i])->get_tensor_name().c_str());
//     }
//     for (size_t i = 0; i < m_outputs.size(); i++) {
//         logToFile("Output shape: ");
//         logToFile(std::dynamic_pointer_cast<TensorPlace>(m_outputs[i])->get_partial_shape().to_string().c_str());
//         logToFile("Output type:");
//         logToFile(std::dynamic_pointer_cast<TensorPlace>(m_outputs[i])->get_element_type().to_string().c_str());
//         logToFile("Output name: ");
//         logToFile(std::dynamic_pointer_cast<TensorPlace>(m_outputs[i])->get_tensor_name().c_str());
//     }
//     logToFile("===================");
// }

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
