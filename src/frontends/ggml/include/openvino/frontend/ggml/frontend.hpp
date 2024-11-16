// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/frontend.hpp"
#include "openvino/frontend/ggml/visibility.hpp"

namespace ov {
namespace frontend {
namespace ggml {

class GGML_API FrontEnd : public ov::frontend::FrontEnd { 
public:
    using Ptr = std::shared_ptr<FrontEnd>;
    FrontEnd();
    
    std::shared_ptr<Model> convert(const InputModel::Ptr& model) const override;
    void convert(const std::shared_ptr<ov::Model>& partiallyConverted) const override;
    std::shared_ptr<ov::Model> convert_partially(const InputModel::Ptr& model) const override;
    std::shared_ptr<ov::Model> decode(const InputModel::Ptr& model) const override;
    void normalize(const std::shared_ptr<ov::Model>& model) const override;
    
    std::string get_name() const override {
        return "ggml";
    }

    void add_extension(const std::shared_ptr<ov::Extension>& extension) override;

protected:
    ov::frontend::InputModel::Ptr load_impl(const std::vector<ov::Any>& variants) const override;
    
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
