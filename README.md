# segment-anything-2 real-time
Run all released versions of segment-anything-2 (SAM2) in combination with the Stereolabs ZED-Stereocamera using the associated python API released in ZED SDK.

More information of the Python Api on the Stereolabs website: https://www.stereolabs.com/docs/app-development/python/install


## Current outline
Different models (and dependencies) are needed to get the project working.
- - -

### Nakama Pyzed Wrapper for the ZED camera

#### Installation
For installation of the wrapper we need to have the ZED SDK configured (see details here: https://www.stereolabs.com/en-nl/developers/release)

This project has been tested for the ZED SDK v4.1. This version integrated the AI-mode for depth estimation. Note that CUDA 12 is required for usage of this SDK, but the installer will prompt you if CUDA 12 is not set up correctly.

Next you can either run the <em><strong>get_python_api.py</strong></em> globally: https://www.stereolabs.com/docs/app-development/python/install

Or Install it within a virtual environment: https://www.stereolabs.com/docs/app-development/python/virtual_env

After that the wrapper package can be installed as a package, or cloned into your project
For more information follow: https://bitbucket.org/ctw-bw/nakama_pyzed_wrapper/src/master/

- - -

### Grounding DINO
Grounding dino is used to estimate inital boundingboxes in the first frame.
#### Installation
step 1:
Set specific path to cuda used to build the model
```bash
export CUDA_HOME=/path/to/desired_cuda_version
```
Step 2: Change the current directory to the GroundingDINO folder.
```bash
cd GroundingDINO/
```

step 3: Install the required dependencies in the current directory.
```bash
pip install -e .
```

step 4: Download pre-trained model weights.
``` bash
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```
- - -

### SAM2
#### Installation
step 1: install requirements
```bash
pip install -e .
```

step2: Download Checkpoints and configurations

Then, we need to download a model checkpoint.
```bash
cd checkpoints
./download_ckpts.sh
```


- - -
### ZED-Camera prediction 

The main code combining the current pipeline is placed in:
```bash
scripts/sam2_track_zed.py
```

Configurations (.yaml) are placed in the <strong><em>configurations</strong></em> folder.

Configurations for the <strong><em>Nakama pyzed wrapper</strong></em> are placed in:
```bash
scripts/pyzed_wrapper/wrapper_settings.py
```
