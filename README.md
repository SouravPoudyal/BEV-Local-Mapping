# Vehicle-Centric-All-Around-Bird's-Eye-Vew-Local-Mapping
This work involves creating a Bird's Eye Vew (BEV) local map where the position of the vehicle with respect to the road can be viewed from a prespective transformed top view. BEV is commonly used for environmental perception as it clearly presents the location and scale of objects (lanes and the vehicle in this case). The focus is getting a higher FPS or cycle time on the BEV view that allows faster tracking of the dynamically changing trajectories. This is also a crucial factor for time-critical applications like autonomous driving and collision avoidance.
The objectives of this work are listed below:
  1. Creating BEV from four fish-eye cameras.
  2. 2D Vehicle positioning and heading information.
  3. Both tasks must be performed in simulation, i.e., Unity and Real hardware (NVIDIA Jetson Xavier).
  4. Output to be 15-30 FPS.

BEV and Vehicle Position Pipeline:

The Pipeline is divided broadly into three components based on the workflow and division.
  1. Calibration and Homography (Run once in the PC).
  2. Seamless BEV (Runs continuously in Jetson).
  3. Extracting Lane Information or Vehicle 2D Position (Runs continuously in Jetson).

<img align="center" src="assets/Images/Pipeline.png" width="750">

Requirements:
  1. Unity (2021.3.15.f1 or later)
  2. Computing Platform( e.g. NVDIA Jetson Xavier)
  3. Four Cameras (e.g. Fish Eye Camera)
  4. Auxiliary hardwares (Display, I/O devices, USB Interfaces)
  5. Python 3.3.8
  6. OpenCV 4.6.0

Results:
[![Watch the Video](https://github.com/SouravPoudyal/BEV-Local-Mapping/blob/main/assets/Images/hardware.png?raw=true)](https://github.com/SouravPoudyal/BEV-Local-Mapping/raw/main/assets/Videos/demo_hardware.mp4)
[![Watch the Video](https://github.com/SouravPoudyal/BEV-Local-Mapping/blob/main/assets/Images/unity.png?raw=true)](https://github.com/SouravPoudyal/BEV-Local-Mapping/raw/main/assets/Videos/demo_simulation.mp4)

