#include "openvino/frontend/ggml/frontend.hpp"
#include "openvino/frontend/ggml/graph_iterator.hpp"

#include "input_model.hpp"
#include "op_table.hpp"
#include "translate_session.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {

using ov::frontend::tensorflow::GraphIterator;
using ov::frontend::tensorflow::ggml::GgmlGraphIterator;

FrontEnd::FrontEnd() {}

std::shared_ptr<Model> FrontEnd::convert(const InputModel::Ptr& model) const{
    //TODO
    auto ggml_model = std::dynamic_pointer_cast<ggml::InputModel>(model);
    FRONT_END_GENERAL_CHECK(ggml_model, "Invalid input model");
    std::shared_ptr<Model> converted_model;
    const auto& supported_ops = get_supported_ops();
    {
        TranslateSession translate_session(model, supported_ops);
        converted_model = translate_session.get_converted_model();
        // dump_ov_model(converted_model);
        serialize_example(converted_model);
    }
    return converted_model;
}

void FrontEnd::convert(const std::shared_ptr<ov::Model>& partiallyConverted) const {
    FRONT_END_NOT_IMPLEMENTED(convert);
}

std::shared_ptr<ov::Model> FrontEnd::convert_partially(const ov::frontend::InputModel::Ptr& model) const {
    //TODO
    std::shared_ptr<ov::Model> converted_model;
    return converted_model;
}

std::shared_ptr<ov::Model> FrontEnd::decode(const InputModel::Ptr& model) const {
    //TODO
    std::shared_ptr<ov::Model> decoded_model;
    return decoded_model;
}

void FrontEnd::normalize(const std::shared_ptr<ov::Model>& model) const {
    //TODO
}

void FrontEnd::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    //TODO
}

ov::frontend::InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& variants) const {
    FRONT_END_GENERAL_CHECK(variants[0].is<std::shared_ptr<GraphIterator>>(),
                            "GGML Frontend doesn't support provided model type. Please provide supported model "
                            "object using API.");
    auto graph_iterator = variants[0].as<std::shared_ptr<GraphIterator>>();
    auto ggml_graph_iterator = std::dynamic_pointer_cast<GgmlGraphIterator>(graph_iterator);
    FRONT_END_GENERAL_CHECK(ggml_graph_iterator, "Couldn't cast ov::Any to GraphIterator");
    return std::make_shared<ggml::InputModel>(ggml_graph_iterator);
}


}  // namespace ggml
}  // namespace frontend
}  // namespace ov
