# cv5551-s26-MMVT

## project Title
- EchoFind
## Team Members
| Name | Email | Role |
|-----|-----|-----|
| Mahmoud Alhihi | Mahmoudalhihi7@gmail.com | Vision, target detection, scene understanding, presentation preparation |
| Viktor Radev | viktorpradev@gmail.com  | Voxel map, belief visualization, probability updates, failure recovery |
| Matthew Giambrone | mattbr0n3@gmail.com | Audio testing, audio filtering, background noise handling, presentation preparation |
| Tarun | sehga060@umn.edu |  Robot motion planning, manipulation primitives, action pipeline, failure recovery |


---

## Project Overview

EchoFind is a robotic search system designed to help a robot locate a hidden or partially occluded target object using both audio and vision. When a robot cannot directly see the target object, it usually has to search through every possible object in the environment. This can waste time and energy, especially in cluttered scenes.

Our solution adds audio sensing to guide the robot toward the most likely location of the target. The robot listens for the sound source, updates a belief map over the workspace, uses visual scene understanding to reason about objects and obstacles, and then performs manipulation actions such as pushing, grasping, and rotating objects until the target is found.

---

## Problem Statement

When a robot cannot directly sense its goal object, such as when the object is hidden or not visually detectable, it must systematically search possible objects in the environment. This exhaustive search can be slow and inefficient.

EchoFind addresses this problem by combining:

- Microphone-based audio localization
- Camera-based vision input
- Voxel-based belief mapping
- Robot manipulation actions
- Iterative belief updates after each observation or action

The goal is to reduce the search area and improve the efficiency of finding the target object.

---

## System Inputs and Outputs

### Inputs

- Camera video stream
- Microphone audio
- Audio recording of the target sound to filter from background noise
- Object and scene information from vision

### Outputs

- Voxel visualization of the environment
- Heatmap/belief map of the target’s probable location
- Robot manipulation actions to search the environment
- Updated belief after audio, vision, and manipulation feedback

---

## Method

EchoFind uses a discrete belief-based search pipeline.

The workspace is represented as a voxel map. Each voxel represents a possible target location. The robot updates the probability of each voxel using audio and vision observations.

The belief update follows the idea of a discrete Bayes filter:

- The state is a discrete voxel location.
- The actions are discrete robot actions such as grasp, rotate, and push.
- The observations come from audio and vision.
- The belief can be multimodal, meaning several locations may be likely at the same time.

The robot repeatedly:

1. Captures audio from the microphone.
2. Filters the target sound from background noise.
3. Updates the voxel belief map.
4. Uses vision to understand the scene and detect possible target regions.
5. Selects a high-probability region or object to interact with.
6. Performs manipulation such as push, grasp, or rotate.
7. Updates the belief again based on the result.
8. Stops when the target is found and graspable.

---

## Main Files

| File/Folder | Description |
|---|---|
| `ArduinoCode/` | Arduino microphone/audio sampling code, including the 20 kHz audio collection setup |
| `audio_localize.py` | Processes audio input and estimates sound direction or location cues |
| `read_mic.py` | Reads microphone data for audio sensing |
| `sweep.py` | Performs audio sweep/search behavior |
| `voxel.py` | Builds and manages the voxel representation of the environment |
| `integratedBeliefV7.py` | Final integrated belief-map pipeline combining audio, vision, and robot interaction |
| `push_cube.py` | Robot pushing behavior for object manipulation |
| `grasp_and_rotate.py` | Robot grasping and rotating behavior |
| `checkpoint0.py` | Initial checkpoint/setup code used for the project pipeline |

---

## Roles and Responsibilities

### Mahmoud Alhihi

- Repurposed AprilTag pose estimation for detecting objects in the environment.
- Worked on 3D pose estimation and occlusion reasoning.
- Contributed to scene state output and looped re-perception.
- Worked on classification and target detection using SAM 3.
- Helped prepare the final project presentation.

### Matthew Giambrone

- Developed and tested speaker/audio setup.
- Worked on audio peaking and search algorithms.
- Implemented audio filtering and matching.
- Improved performance in noisy background environments.
- Helped prepare the final project presentation.

### Viktor Radev

- Created the voxel map of the environment using vision.
- Built belief visualization using Gaussian-style probability distributions.
- Updated voxel probabilities using audio and vision information.
- Worked on live-stream-based belief updates.
- Contributed to failure detection and recovery strategies.

### Tarun Sehgal

- Implemented basic robot motion planning.
- Built manipulation primitives such as push, pick, and place.
- Developed the action pipeline for selecting objects, planning motion, executing actions, and updating the scene.
- Used the belief map to guide obstacle removal and prioritize high-probability regions.
- Contributed to failure detection and recovery strategies.

---

## Evaluation

The system was designed to compare search efficiency with and without audio guidance.

The planned evaluation involved:

- Running trials with audio enabled.
- Running trials with audio disabled.
- Comparing average search time across trials.
- Measuring whether audio reduced the search space and improved localization efficiency.

The main hypothesis is that adding audio cues allows the robot to focus on high-probability regions instead of searching every object in the environment.

---

## Technologies Used

- Python
- Arduino
- Open3D
- NumPy
- SAM 3
- UFactory Lite6 robot arm
- Microphone-based audio sensing
- Camera-based visual perception
- Voxel-based belief mapping

---

## Final Project Branch

This cleaned version of the repository contains the main files needed for the final EchoFind project pipeline.


---

## Troubleshooting Note

If there are any issues with the cleaned `main` branch version, please use `integratedBeliefV7.py` from the `Final-Project-V1` branch as the reference final version.