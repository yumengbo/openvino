# GGML Frontend Build Instructions

To build the GGML frontend, follow these steps:

1. Create a build directory:
    ```sh
    mkdir build
    ```

2. Navigate to the build directory:
    ```sh
    cd build
    ```

3. Run CMake with the `ENABLE_OV_GGML_FRONTEND` option enabled:
    ```sh
    cmake -DENABLE_OV_GGML_FRONTEND=ON ..
    ```

4. Build the project:
    ```sh
    cmake --build .
    ```

## Output

The `.so` files of the GGML frontend can be found in the following directory:
```
${openvino_repo_dir}/bin/intel64/Release
```

The files include:
- `libopenvino_ggml_frontend.so`
- `libopenvino_ggml_frontend.so.2025.0.0`
- `libopenvino_ggml_frontend.so.2500`