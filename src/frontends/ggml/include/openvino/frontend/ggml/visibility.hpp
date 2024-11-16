#pragma once

#include <openvino/core/visibility.hpp>

#ifdef OPENVINO_STATIC_LIBRARY
#    define GGML_API
#    define GGML_C_API
#else
#    ifdef openvino_ggml_frontend_EXPORTS
#        define GGML_API   OPENVINO_CORE_EXPORTS
#        define GGML_C_API OPENVINO_EXTERN_C OPENVINO_CORE_EXPORTS
#    else
#        define GGML_API   OPENVINO_CORE_IMPORTS
#        define GGML_C_API OPENVINO_EXTERN_C OPENVINO_CORE_IMPORTS
#    endif  // openvino_ggml_frontend_EXPORTS
#endif      // OPENVINO_STATIC_LIBRARY