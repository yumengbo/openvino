#pragma once

#include "openvino/frontend/graph_iterator.hpp"

namespace ov {
namespace frontend {
namespace tensorflow { // To be Removed
namespace ggml {

// TODO: Directly include from openvino
class GgmlGraphIterator : public GraphIterator {
public:
    
    virtual size_t size() const override;

    virtual void reset() override;

    virtual void next() override;

    virtual bool is_end() const override;

    virtual std::shared_ptr<DecoderBase> get_decoder() const override;

    virtual std::vector<std::string> get_input_names() const override;

    virtual std::vector<std::string> get_output_names() const override;

    virtual std::shared_ptr<GraphIterator> get_body_graph_iterator(const std::string& func_name) const override;

    virtual std::map<std::string, std::string> get_input_names_map() const override{
        return {};
    }

    virtual std::map<std::string, std::string> get_output_names_map() const override{
        return {};
    }
    
};

}
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
