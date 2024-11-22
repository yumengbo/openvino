#include "basic_api.hpp"

using namespace ov::frontend;
using namespace ov::frontend::ggml::tests;

using GGMLBasicTest = FrontEndBasicTest;

static const std::vector<std::string> models{
    std::string("2in_2out/2in_2out.pb"),
};

INSTANTIATE_TEST_SUITE_P(GGMLBasicTest,
                         FrontEndBasicTest,
                         ::testing::Combine(::testing::Values(GGML_FE),
                                            ::testing::Values(std::string(TEST_GGML_MODELS_DIRNAME)),
                                            ::testing::ValuesIn(models)),
                         FrontEndBasicTest::getTestCaseName);
