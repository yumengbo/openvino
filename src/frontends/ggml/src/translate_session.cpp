#include "translate_session.hpp"
#include "input_model.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {

using namespace ov::op;

TranslateSession::TranslateSession(const frontend::InputModel::Ptr& input_model,
                                    const std::unordered_map<std::string, CreatorFunction>& translator_map)
                                    :m_input_model(input_model),
                                    m_translator_map(translator_map),
                                    m_ov_model(nullptr) {}

std::shared_ptr<Model> TranslateSession::get_converted_model() {
    if (m_ov_model) {
        return m_ov_model;
    }
    m_ov_model = translate_graph(m_input_model, m_ov_model);
    return m_ov_model;
}

std::shared_ptr<Model> TranslateSession::translate_graph(const frontend::InputModel::Ptr& input_model,
                                                        std::shared_ptr<ov::Model>& ov_model){
    ov::ParameterVector params;
    ov::ResultVector results;
    auto tensor_map = std::make_shared<TensorMap>();
    std::shared_ptr<Model> resulting_model;  
    
    const auto& ggml_model = std::dynamic_pointer_cast<InputModel>(input_model);
    FRONT_END_GENERAL_CHECK(ggml_model, "nullptr for InputModel is given for translation into OV Model");
    const auto& operation_places = ggml_model->get_op_places();
    const auto& model_inputs = ggml_model->get_inputs();
    const auto& model_outputs = ggml_model->get_outputs();

    // Process model inputs
    for (const auto& input_place : model_inputs) {
        auto input_tensor_place = std::dynamic_pointer_cast<TensorPlace>(input_place);
        auto input_shape = input_tensor_place->get_partial_shape();
        auto input_type = input_tensor_place->get_element_type();
        auto input_tensor_name = input_tensor_place->get_tensor_name();
        auto param = std::make_shared<v0::Parameter>(input_type, input_shape);
        param->set_friendly_name(input_tensor_name);
        params.push_back(param);
        (*tensor_map)[input_tensor_name] = param;
    }

    // Transform operation
    for (const auto& operation_place : operation_places) {
        auto operation_decoder = operation_place->get_decoder();
        auto operation_name = operation_place->get_operation_name();
        if (tensor_map->find(operation_name) != tensor_map->end()) {
            // Already processed
            continue;
        }
        auto raw_inputs = operation_place->get_input_tensor();
        // TODO
        for (size_t i = 0; i < raw_inputs.size(); ++i) {
            auto input = raw_inputs.at(i);
            auto input_name = input->get_tensor_name();
            // Not processed inputs
            if (tensor_map->find(input_name) == tensor_map->end()) {
                auto input_shape = input->get_partial_shape();
                auto input_type = input->get_element_type();
                auto param = std::make_shared<v0::Parameter>(input_type, input_shape);
                (*tensor_map)[input_name] = param;
            }
        }

        ov::OutputVector converted_outputs;
        auto operation_type = operation_place->get_operation_type();
        auto it = m_translator_map.find(operation_type);
        if (it != m_translator_map.end()) {
            try {
                NodeContext node_context(operation_decoder, tensor_map, this);
                converted_outputs = it->second(node_context);
            } catch (const std::exception& ex) {
                // TODO
            }
        } else {
            // TODO
        }

        const auto& node_outputs = operation_place->get_output_tensor(); 
        // Check converted output size
        FRONT_END_OP_CONVERSION_CHECK(node_outputs.size() == converted_outputs.size(),
                                          "Number of ",
                                          operation_name,
                                          " outputs greater than number of converted outputs, which are",
                                          node_outputs.size(),
                                          " and ",
                                          converted_outputs.size(),
                                          " respectively.");
        for (size_t i = 0; i < node_outputs.size(); ++i) {
            auto output = node_outputs.at(i);
            auto output_name = output->get_tensor_name();
            if (tensor_map->find(output_name) == tensor_map->end()) {
                (*tensor_map)[output_name] = converted_outputs[i];
            }
        }
    }

    for (auto& output_p : input_model->get_outputs()) {
        auto ggml_output_place = std::dynamic_pointer_cast<TensorPlace>(output_p);
        FRONT_END_GENERAL_CHECK(ggml_output_place, "Only place produced by Jax Frontend is supported.");
        auto ggml_output_name = ggml_output_place->get_tensor_name();
        if (tensor_map->find(ggml_output_name) != tensor_map->end()) {
            auto result = std::make_shared<v0::Result>(tensor_map->at(ggml_output_name));
            results.push_back(result);
        }
    }

    resulting_model = std::make_shared<Model>(results, params);
    return resulting_model;
}



}  // namespace ggml
}  // namespace frontend
}  // namespace ov
