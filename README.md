# OpenCV Rust Demo: CPU vs GPU Performance Comparison

A Rust demonstration program that showcases OpenCV's CPU vs GPU (OpenCL) performance comparison using real image processing operations with a focus on rusticl platform support (OpenCL 3.0).

## Features

- **Real Image Processing**: Implements a complete image processing pipeline with:
  - Grayscale conversion
  - Gaussian blur (noise reduction)
  - Canny edge detection 
- **CPU vs GPU Comparison**: Measures performance differences between CPU (Mat) and GPU (UMat) processing
- **Rusticl Platform Support**: Specifically configures and targets the rusticl platform for OpenCL 3.0 support
- **OpenCL Detection**: Automatically detects OpenCL availability and lists platforms/devices with rusticl detection
- **Timing Measurements**: Precise timing using `std::time::Instant` for performance comparison
- **Error Handling**: Robust error handling throughout the pipeline

## Prerequisites

- Rust (latest stable version)
- OpenCV 4.x installed on your system
- OpenCL runtime for GPU acceleration (preferably with rusticl support for OpenCL 3.0)

### Installing OpenCV

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install libopencv-dev clang libclang-dev
```

**macOS:**
```bash
brew install opencv
```

**Windows:**
Follow the [OpenCV installation guide](https://docs.opencv.org/4.x/d3/d52/tutorial_windows_install.html) for Windows.

### OpenCL Runtime (for GPU acceleration)

The demo is configured to use the rusticl platform (OpenCL 3.0). Install Mesa with rusticl support:

**Ubuntu/Debian:**
```bash
sudo apt install mesa-opencl-icd
```

## Usage

1. **Clone and build:**
```bash
git clone <repository-url>
cd opencv-opencl-demo
cargo build --release
```

2. **Run with an image:**
```bash
cargo run --release path/to/your/image.jpg
```

3. **Example with system image:**
```bash
# Using a sample system image
cargo run --release /usr/share/gtk-doc/html/totem/home.png
```

## Sample Output

Without OpenCL:
```
Warning: rusticl platform not found!
OpenCL is disabled

CPU pipeline: 8.436ms
```

With OpenCL and rusticl platform:
```
Platform #0: rusticl
  Device #0: AMD Radeon RX 6800 XT (OpenCL 3.0)
  -> This is the rusticl platform with OpenCL 3.0 support
rusticl platform found - configured to use via OPENCV_OPENCL_DEVICE

OpenCL is enabled

=== ACTIVE OpenCL DEVICE ===
Device Name: AMD Radeon RX 6800 XT
Device Version: OpenCL 3.0
=============================

CPU pipeline: 12.342ms
OpenCL pipeline: 4.187ms
```

## Image Processing Pipeline

The demo implements the following computer vision pipeline:

1. **Grayscale Conversion** (`cvtColor`):
   - Converts BGR color image to grayscale
   - Purpose: Reduces data dimensionality for subsequent processing

2. **Gaussian Blur** (`GaussianBlur`):
   - Kernel size: 7×7
   - Standard deviation: 1.5
   - Purpose: Reduces image noise before edge detection

3. **Canny Edge Detection** (`Canny`):
   - Low threshold: 0.0
   - High threshold: 50.0
   - Aperture size: 3
   - Purpose: Detects edges in the image

## Performance Characteristics

- **CPU Processing**: Uses OpenCV's `Mat` class with standard CPU operations
- **GPU Processing**: Uses OpenCV's `UMat` class with OpenCL acceleration via rusticl
- **Timing**: Measures end-to-end pipeline execution time
- **Iterations**: Performs 10 iterations per test for accurate timing
- **OpenCL Device Selection**: Automatically configures to use rusticl platform via `OPENCV_OPENCL_DEVICE` environment variable

## Code Structure

- `main.rs`: Entry point with OpenCL detection, rusticl configuration, and timing
- `cpu_pipeline()`: CPU-based image processing using `Mat`
- `gpu_pipeline()`: GPU-based image processing using `UMat`  
- `time_it()`: Utility function for precise timing measurements

## OpenCL Support

The demo automatically:
- Sets `OPENCV_OPENCL_DEVICE` environment variable to target rusticl platform
- Detects OpenCL platform availability with emphasis on rusticl
- Lists all available OpenCL devices and identifies rusticl platform
- Displays active OpenCL device information when available
- Falls back gracefully to CPU-only mode if OpenCL is unavailable
- Uses OpenCV's transparent UMat system for GPU acceleration

## Dependencies

```toml
[dependencies]
opencv = "0.95.0"
```

## Platform Support

- **Linux**: Full support with OpenCL (rusticl recommended)
- **macOS**: Full support with OpenCL  
- **Windows**: Full support with OpenCL

## Troubleshooting

**OpenCV not found:**
- Ensure OpenCV is properly installed
- Set `OPENCV_LINK_LIBS`, `OPENCV_LINK_PATHS`, and `OPENCV_INCLUDE_PATHS` if needed

**OpenCL not available:**
- Install OpenCL runtime for your GPU
- For rusticl support, ensure Mesa with rusticl is installed
- Check that your GPU supports OpenCL
- The demo will still work in CPU-only mode

**Rusticl platform not found:**
- Ensure Mesa with rusticl support is installed
- Check that your GPU driver supports OpenCL 3.0
- The demo will fall back to other available OpenCL platforms

**Compilation errors:**
- Ensure you have a C++ compiler installed
- On Ubuntu: `sudo apt install build-essential`
- On macOS: Install Xcode command line tools

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please feel free to submit pull requests or open issues for bugs and feature requests. 
