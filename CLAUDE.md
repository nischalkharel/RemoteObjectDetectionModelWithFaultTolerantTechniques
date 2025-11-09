READ FIRST!!! THIS IS VERY IMPORTANT.
- THIS IS BEING WRITTEN BY NISCHAL KHAREL, the user.

## DEVELOPMENT ENVIRONMENT SETUP - CRITICAL INFO
**VS Code Terminal Location:** This VS Code is running on Windows PC (nisch@NischalLabPC)
**Target Device:** Jetson Xavier NX (shrenx@ubuntu) - accessed via SSH
**IMPORTANT:** Claude cannot directly execute commands on the Xavier. When testing or running commands on the Xavier device, Claude will ask Nischal to run them and provide the terminal output back.

Working directory structure:
- **PC:** `C:\Users\nisch\Documents\JetsonXavierRun\RemoteObjectDetectionModelWithFaultTolerantTechniques\` (for code development)
- **Xavier:** `~/Nischal_NVBitFi/RemoteObjectDetectionWithFaultTolerantTechniques/` (project files)
- **Xavier:** `~/Nischal_NVBitFi/nvbit_release/` (NVBit 1.5.5 installation - SEPARATE directory)
I could not get NVBitFi to be compatable with Orin Nano. So I had to pivid to Jetson Xavier NX. I finally got it pulled down and installed and ran the ./test.sh and it seems like it worked. Now we need to edit setup_and_run_nvbitfi.sh according to how this new platform supports it. Currently the CUDA on this xavier is 10.2. and NVBit is 1.5.5. This is 

Here is more info about the board.
(nvbitfi_env) shrenx@ubuntu:/usr/local/cuda/samples/1_Utilities/deviceQuery$ ./deviceQuery
./deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "Xavier"
  CUDA Driver Version / Runtime Version          10.2 / 10.2
  CUDA Capability Major/Minor version number:    7.2
  Total amount of global memory:                 7775 MBytes (8152907776 bytes)
  ( 6) Multiprocessors, ( 64) CUDA Cores/MP:     384 CUDA Cores
  GPU Max Clock rate:                            1109 MHz (1.11 GHz)
  Memory Clock rate:                             1109 Mhz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 524288 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            Yes
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 0 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 10.2, CUDA Runtime Version = 10.2, NumDevs = 1
Result = PASS




About where NVBitFi is currently located and how it should work. There is a folder in this project that I have put to show you an example of how NVBit and NVBitFi are setup in xavier. The folder is called nvbit_release and inside tools you will find nvbitFi's info and also the test.sh and everything. that is not a machine learning model and it is a simple model so you will not need to worry too much about it. Make sure to get all the directories right when using it anywhere.


I want to make a very simple pipeline. 
This project will be here: NOTE that nvbit_release is in a different directory than other files. Make sure to be very careful with where everything is.
(nvbitfi_env) shrenx@ubuntu:~/Nischal_NVBitFi/RemoteObjectDetectionWithFaultTolerantTechniques$ ls
aggregate_fault_results.py  notechnique_inference.py  run_evaluations.py        validation_dataset_list.txt
CLAUDE.md                   output                    run_single_inference.py
compare_results.py          Plane_Ship_Detection      setup_and_run_nvbitfi.sh
Images                      requirements.txt          tmr_inference.py
(nvbitfi_env) shrenx@ubuntu:~/Nischal_NVBitFi/RemoteObjectDetectionWithFaultTolerantTechniques$ cd Plane_Ship_Detection/
(nvbitfi_env) shrenx@ubuntu:~/Nischal_NVBitFi/RemoteObjectDetectionWithFaultTolerantTechniques/Plane_Ship_Detection$ ls
Plane_Ship_Model.pt
(nvbitfi_env) shrenx@ubuntu:~/Nischal_NVBitFi/RemoteObjectDetectionWithFaultTolerantTechniques/Plane_Ship_Detection$ cd ../Images
(nvbitfi_env) shrenx@ubuntu:~/Nischal_NVBitFi/RemoteObjectDetectionWithFaultTolerantTechniques/Images$ ls
Validation_Images  Validation_Labels
(nvbitfi_env) shrenx@ubuntu:~/Nischal_NVBitFi/RemoteObjectDetectionWithFaultTolerantTechniques/Images$ cd ..
(nvbitfi_env) shrenx@ubuntu:~/Nischal_NVBitFi/RemoteObjectDetectionWithFaultTolerantTechniques$ cd ..
(nvbitfi_env) shrenx@ubuntu:~/Nischal_NVBitFi$ ls
hello  hello.cu  nvbit-Linux-aarch64-1.5.5.tar.bz2  nvbit_release  RemoteObjectDetectionWithFaultTolerantTechniques
(nvbitfi_env) shrenx@ubuntu:~/Nischal_NVBitFi$ cd nvbit_release/
(nvbitfi_env) shrenx@ubuntu:~/Nischal_NVBitFi/nvbit_release$ ls
core  EULA.txt  LICENSE  README.md  test-apps  tools
(nvbitfi_env) shrenx@ubuntu:~/Nischal_NVBitFi/nvbit_release$ cd tools
(nvbitfi_env) shrenx@ubuntu:~/Nischal_NVBitFi/nvbit_release/tools$ ls
instr_count     instr_count_cuda_graph  mem_printf  mov_replace  opcode_hist
instr_count_bb  Makefile                mem_trace   nvbitfi      record_reg_vals
(nvbitfi_env) shrenx@ubuntu:~/Nischal_NVBitFi/nvbit_release/tools$



also we will need to activate conda environment nvbitfi_env. 


ok the rest of the pipeline should be.
dont worry about NoFaultRuns. I will take care of that. the Golden Runs will be saved in output directory and stuff before we do the fault runs. you can check run_evaluations.py to understand how we will output golden runs to the output folder so just assume that will be good before we do any fault runs.

Next change setup_and_run_nvbitfi.py and also help me make the necessary things in nvbitfi's pipeline so we can run the thing with fault injected. we can initially try to use their compare thing to know if it is SDC or masked and so on. 
but I do want the outputs from the fault runs to be in output folder as well. so I can see them too. it can be in nvfitfi's pipeline too but I just want it in output as well. 

make sure to keep everything simple and followable. no need to make a bunch of random files. this is just for an experiment so no need to be too extra. I only have notechnique and tmr for now but later I will have more techniques so whereever I will need to update after adding more techniques, make a comment saying TODO: ADD TECHNIQUES HERE!!


# Project Guidelines

## User Preferences
- **Keep it simple**: No unnecessary files, scripts, or automation
- **Focus on the experiment**: Don't create extra documentation, READMEs, or setup files unless absolutely necessary
- **Clean workspace**: Delete temporary files and avoid clutter
- **Minimal approach**: One script to do the job, not multiple helper scripts

## Project Goal
**Evaluate fault tolerance techniques for YOLOv8 object detection under GPU fault injection using NVBitFI**

### Techniques Being Tested:
1. `notechnique` - Baseline (no fault tolerance)
2. `tmr` - Triple Modular Redundancy (Run 3, Vote 1)

### Experimental Setup:
- Platform: Jetson Xavier NX (sm_72, CUDA 10.2)
- NVBit Version: 1.5.5
- Model: YOLOv8n (Plane/Ship Detection)
- Dataset: 112 validation images
- Fault Injection: NVBitFI (10,000 injections per technique)
- Fault Model: Single-bit flips in general-purpose (GP) instructions
- Conda Environment: nvbitfi_env

### Key Files:
- `notechnique_inference.py` - Baseline inference
- `tmr_inference.py` - TMR inference
- `run_single_inference.py` - Worker called by NVBitFI 10K times
- `run_evaluations.py` - Run without faults for testing
- `setup_and_run_nvbitfi.sh` - Complete NVBitFI setup and execution
- `compare_results.py` - Analyze predictions
- `aggregate_fault_results.py` - Aggregate 10K results

## Development Guidelines
- Keep adding any new things being installed into requirements.txt and also in the venv
- Always update existing files instead of creating new ones
- One setup script maximum - don't create multiple automation scripts
- Don't create documentation files (*.md, README) unless explicitly requested



