# Stereovision Depth to ROIs using Segmentation

This project combines **Segment-Anything-2 (SAM2)** with the **Stereolabs ZED-Stereocamera**, leveraging the Python API provided by the ZED SDK to extract depth information and segment Regions of Interest (ROIs) in real-time.

Optional integration with **GroundingDINO** allows for the use of text prompts to generate bounding boxes as input for SAM2, enhancing segmentation flexibility.

For more details about the ZED Python API, visit the [Stereolabs documentation](https://www.stereolabs.com/docs/app-development/python/install).

---

## Overview

This project requires the integration of multiple models and dependencies to achieve real-time segmentation and depth estimation. Below is an outline of the components and their purposes:

- **SAM2**: Instance segmentation using flexible prompts for guidance.
- **GroundingDINO**: Text-based object detection to generate bounding box prompts for SAM2.
- **Nakama Pyzed Wrapper**: Streamlined interaction with the ZED stereocamera.

---

## Installation

### Prerequisites: Setting up CUDA

**Note:**
- For the ZED camera, CUDA 12.1 is required to use Python SDK 4.1. The SDK installer handles this automatically.
- For other models, any CUDA version ≥ 11.3 can be used.

After setting up CUDA, ensure the correct version is being used by configuring the `CUDA_HOME` environment variable:

```bash
export CUDA_HOME=/path/to/desired_cuda_version
```

To verify the CUDA version:

```bash
which nvcc
```

The output should point to the desired CUDA version, e.g., `/usr/local/cuda-12.6/bin/nvcc`. Use the corresponding path to set `CUDA_HOME`. Confirm the setup:

```bash
echo $CUDA_HOME
```

---

### SAM2

**Note:** It is recommended to install SAM2 before GroundingDINO, as SAM2's installer includes required dependencies for GroundingDINO.

#### Installation

1. Install requirements:

   ```bash
   pip install -e .
   ```

2. Download checkpoints and configurations:

   ```bash
   cd checkpoints
   ./download_ckpts.sh
   ```

---

### GroundingDINO

#### Installation

1. Change to the `GroundingDINO` directory:

   ```bash
   cd GroundingDINO/
   ```

2. Install required dependencies:

   ```bash
   pip install -e .
   ```

3. Download pre-trained model weights:

   ```bash
   mkdir weights
   cd weights
   wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
   cd ..
   ```

---

### Nakama Pyzed Wrapper for the ZED Camera

#### Installation

1. Install the ZED SDK (details [here](https://www.stereolabs.com/en-nl/developers/release)).

   **Note:** This project has been tested with ZED SDK v4.1, which integrates the AI mode for depth estimation. The installer will prompt you to install CUDA 12 if it is not already configured.

2. Install the ZED Python API:

   - Globally:
     [Python API Installation](https://www.stereolabs.com/docs/app-development/python/install)
   - Within a virtual environment:
     [Python API Virtual Env](https://www.stereolabs.com/docs/app-development/python/virtual_env)

3. Install the Nakama Pyzed Wrapper as a package or clone it into your project:
   [Nakama Pyzed Wrapper](https://bitbucket.org/ctw-bw/nakama_pyzed_wrapper/src/master/)

---

## ZED Camera Prediction

The main pipeline code is located at:

```bash
scripts/sam2_track_zed.py
```

### Configuration

1. General configurations are in the `configurations` folder. Update paths to match your machine setup.

2. Nakama Pyzed Wrapper-specific settings are in:

   ```bash
   scripts/pyzed_wrapper/wrapper_settings.py
   
