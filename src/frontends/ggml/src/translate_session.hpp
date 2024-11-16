#pragma once

#include "input_model.hpp"
#include "openvino/frontend/ggml/node_context.hpp"

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession {
public:
    TranslateSession(const frontend::InputModel::Ptr& input_model,
                     const std::map<std::string, CreatorFunction>& translator_map);
    ~TranslateSession();

    std::shared_ptr<Model> get_converted_model();
    std::shared_ptr<Model> translate_graph(const frontend::InputModel::Ptr& input_model);

private:
    const frontend::InputModel::Ptr m_input_model;
    const std::map<std::string, CreatorFunction>& m_translator_map;
    std::shared_ptr<Model> m_ov_model;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
