# Vendored Vulkan loader

Khronos Vulkan-Loader **v1.3.296**, built from source on G27WS0011
(Ubuntu 20.04, gcc, glibc 2.31 — max GLIBC requirement is 2.7, so the
binary runs on any of the lab workstations).

## Why

Ubuntu 20.04 ships `libvulkan.so.1` **1.2.131**, which segfaults inside
`vkEnumerateInstanceVersion` / `vkEnumerateInstanceLayerProperties` on
G27WS0011 (RTX 2080 Ti, NVIDIA driver 535.230.02) with **every** ICD
(NVIDIA and lavapipe both crash — it is the loader, not the driver).
SAPIEN's bundled glfw `dlopen`s the **unversioned** `libvulkan.so`, which
`LD_PRELOAD` cannot intercept (the dynamic linker matches dlopen requests
by soname), so the launchers put this directory first on the dlopen search
path via `LD_LIBRARY_PATH` instead. The existing `libX11`/`libxcb`
preloads (uv-CPython symbol interposition) are unchanged and independent.

## Rebuild

```bash
git clone --depth 1 --branch vulkan-sdk-1.3.296.0 \
    https://github.com/KhronosGroup/Vulkan-Loader.git /tmp/vkloader
git clone --depth 1 --branch vulkan-sdk-1.3.296.0 \
    https://github.com/KhronosGroup/Vulkan-Headers.git /tmp/vkheaders
cmake -S /tmp/vkheaders -B /tmp/vkheaders/build \
    -DCMAKE_INSTALL_PREFIX=/tmp/vkheaders-dist
cmake --build /tmp/vkheaders/build --target install --parallel
cmake -S /tmp/vkloader -B /tmp/vkloader/build -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=OFF -DBUILD_WSI_XCB_SUPPORT=ON -DBUILD_WSI_XLIB_SUPPORT=ON \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_PREFIX_PATH=/tmp/vkheaders-dist
cmake --build /tmp/vkloader/build --parallel
cp -a /tmp/vkloader/build/loader/libvulkan.so* deployment/vendor/vulkan/
```

Requires cmake >= 3.17 (Ubuntu 20.04's 3.16 is too old; `uv pip install
cmake` into the repo venv was used here).

## Optional: fix the whole machine (needs root)

```bash
sudo cp -a deployment/vendor/vulkan/libvulkan.so* /usr/local/lib/ && sudo ldconfig
```

`/usr/local/lib` precedes `/usr/lib` in the ldconfig cache, so every
application on the machine (including `vulkaninfo`) then uses the fixed
loader and the launcher's `LD_LIBRARY_PATH` becomes a harmless no-op.
