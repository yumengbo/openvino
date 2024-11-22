#include "place.hpp"
#include "utils.hpp"

#include "input_model.hpp"

namespace ov {
namespace frontend {
namespace ggml {

OpPlace::OpPlace(const ov::frontend::InputModel& input_model, std::shared_ptr<GgmlDecoder> op_decoder)
    : m_input_model(input_model),
      m_op_decoder(op_decoder) {
    for (size_t i = 0; i < m_op_decoder->get_input_size(); i++) {
        auto input_place = std::make_shared<TensorPlace>(m_op_decoder->get_input_shape(i),
                                                        m_op_decoder->get_input_type(i),
                                                        m_op_decoder->get_op_name(),
                                                        m_op_decoder->get_input_name(i),
                                                        m_op_decoder->is_graph_input(i),
                                                        false);
        m_input_tensors.push_back(input_place);
    }
    for (size_t i = 0; i < m_op_decoder->get_output_size(); i++) {
        auto output_place = std::make_shared<TensorPlace>(m_op_decoder->get_output_shape(i),
                                                        m_op_decoder->get_output_type(i),
                                                        m_op_decoder->get_op_name(),
                                                        m_op_decoder->get_output_name(i),
                                                        false,
                                                        m_op_decoder->is_graph_output(i));
        m_output_tensors.push_back(output_place);
    }
}

std::vector<std::shared_ptr<TensorPlace>> OpPlace::get_input_tensor() const {
    return m_input_tensors;
}

std::vector<std::shared_ptr<TensorPlace>> OpPlace::get_output_tensor() const {
    return m_output_tensors;
}

const std::string& OpPlace::get_operation_name() const {
    return m_op_decoder->get_op_name();
}

const std::string& OpPlace::get_operation_type() const {
    return m_op_decoder->get_op_type();
}

std::shared_ptr<GgmlDecoder> OpPlace::get_decoder() const {
    return m_op_decoder;
}

TensorPlace::TensorPlace(const ov::PartialShape& pshape,
                    ov::Any type,
                    const std::string& operation_name,
                    const std::string& tensor_name,
                    bool is_input,
                    bool is_output)
                    : m_pshape(pshape),
                      m_operation_name(operation_name),
                      m_tensor_name(tensor_name),
                      m_is_input(is_input),
                      m_is_output(is_output) {
    FRONT_END_GENERAL_CHECK(type.is<ov::element::Type>(),
                            "GGML Frontend doesn't support provided model type. Please provide supported model "
                            "object using API.");
    m_type = type.as<ov::element::Type>();
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
