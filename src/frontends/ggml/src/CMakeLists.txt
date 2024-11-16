ov_add_frontend(NAME ggml
                LINKABLE_FRONTEND
                DISABLE_CPP_INSTALL
                FILEDESCRIPTION "FrontEnd to load and convert GGML graph"
                LINK_LIBRARIES openvino::util openvino::core::dev)