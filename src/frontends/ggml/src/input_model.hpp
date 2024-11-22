#pragma once

#include "openvino/frontend/input_model.hpp"
#include "place.hpp"
#include "openvino/frontend/ggml/graph_iterator.hpp"


namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd;
class GgmlDecoder;
using ov::frontend::tensorflow::ggml::GgmlGraphIterator;

class InputModel : public ov::frontend::InputModel {
    friend class ::ov::frontend::ggml::FrontEnd;

public:
    explicit InputModel(const std::shared_ptr<GgmlGraphIterator>& graph_iterator);

    std::vector<frontend::Place::Ptr> get_inputs() const override;
    std::vector<frontend::Place::Ptr> get_outputs() const override;
    std::vector<std::shared_ptr<OpPlace>> get_op_places() const;

    void dump_input_model() const;

private:
    void load_places();
    std::shared_ptr<GgmlGraphIterator> m_graph_iterator;
    std::unordered_map<std::string, std::shared_ptr<frontend::Place>> m_name_to_place;
    std::vector<std::shared_ptr<frontend::Place>> m_inputs;
    std::vector<std::shared_ptr<frontend::Place>> m_outputs;       
    std::vector<std::shared_ptr<OpPlace>> m_op_places;
    std::map<std::string, std::shared_ptr<OpPlace>> m_op_places_map;                                                                                                   
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
