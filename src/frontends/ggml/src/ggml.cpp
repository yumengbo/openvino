// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/ggml/frontend.hpp"
#include "openvino/frontend/ggml/visibility.hpp"
#include "openvino/frontend/manager.hpp"

GGML_C_API ov::frontend::FrontEndVersion get_api_version() {
    return OV_FRONTEND_API_VERSION;
}

GGML_C_API void* get_front_end_data() {
    auto res = new ov::frontend::FrontEndPluginInfo();
    res->m_name = "ggml";
    res->m_creator = []() {
        return std::make_shared<ov::frontend::ggml::FrontEnd>();
    };
    return res;
}
