#pragma once

#include "openvino/frontend/input_model.hpp"
#include "openvino/frontend/place.hpp"
#include "openvino/frontend/ggml/decoder.hpp"

namespace ov {
namespace frontend {
namespace ggml {

class InputModel;
class TensorPlace;
class OpPlace;

class OpPlace : public ov::frontend::Place {
    friend class ::ov::frontend::ggml::InputModel;

public:
    OpPlace(const ov::frontend::InputModel& input_model, std::shared_ptr<GgmlDecoder> op_decoder);

    ~OpPlace() override = default;

    std::vector<std::shared_ptr<TensorPlace>> get_input_tensor() const;

    std::vector<std::shared_ptr<TensorPlace>> get_output_tensor() const;

    const std::string& get_operation_name() const;

    const std::string& get_operation_type() const;

    std::shared_ptr<GgmlDecoder> get_decoder() const;

private:
    const ov::frontend::InputModel& m_input_model;
    std::shared_ptr<GgmlDecoder> m_op_decoder;
    std::vector<std::shared_ptr<TensorPlace>> m_input_tensors;
    std::vector<std::shared_ptr<TensorPlace>> m_output_tensors;
};


class TensorPlace: public ov::frontend::Place {
    public:
        TensorPlace(const ov::PartialShape& pshape,
                    ov::Any type,
                    const std::string& operation_name,
                    const std::string& tensor_name,
                    bool is_input,
                    bool is_output);

        // Internal usage
        const PartialShape& get_partial_shape() const {
            return m_pshape;
        }

        const element::Type& get_element_type() const {
            return m_type;
        }

        const std::string& get_operation_name() const {
            return m_operation_name;
        }

        const std::string& get_tensor_name() const {
            return m_tensor_name;
        }

        void set_partial_shape(const PartialShape& pshape) {
            m_pshape = pshape;
        }

        void set_element_type(const element::Type& type) {
            m_type = type;
        }

        bool is_model_input() {
            return m_is_input;
        }

        bool is_model_output() {
            return m_is_output;
        }

    private:
        PartialShape m_pshape;
        element::Type m_type;
        // store original node name from which tensor place is created
        std::string m_operation_name;
        std::string m_tensor_name;
        bool m_is_input = false;
        bool m_is_output = false;
    };


}  // namespace ggml
}  // namespace frontend
}  // namespace ov
