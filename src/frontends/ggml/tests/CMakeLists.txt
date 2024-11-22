set(TARGET_NAME ov_ggml_frontend_tests)

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            openvino_ggml_frontend
        LINK_LIBRARIES
            gtest
            gtest_main
            openvino::runtime::dev
            common_test_utils
            frontend_shared_test_classes
        INCLUDES
            "${CMAKE_CURRENT_SOURCE_DIR}/../include"
        ADD_CLANG_FORMAT
        LABELS
            OV UNIT GGML_FE
)
