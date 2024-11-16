#include "place.hpp"

#include "input_model.hpp"

namespace ov {
namespace frontend {
namespace ggml {

Place::Place(const ov::frontend::InputModel& input_model, size_t tensor_index)
    : m_input_model(input_model),
      m_tensor_index(tensor_index) {

}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
