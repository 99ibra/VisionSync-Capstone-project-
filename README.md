# VisionSync-Capstone-project-


## Overview

This project focuses on developing a wearable assistive device designed to help visually impaired individuals better understand and navigate their surroundings. The device provides real-time environmental descriptions and object distance detection using stereo vision, delivering feedback through audio output to enhance independence and safety.

The system is built using a Raspberry Pi 5, dual camera modules, and a Bluetooth speaker, enabling real-time image capture, processing, and auditory feedback without relying on dedicated depth sensors.


## Features

- Real-time environmental description using image captioning

- Stereo vision–based depth estimation using two cameras

- Obstacle detection and distance alerts through audio feedback

- Hands-free wearable design for ease of use

- Bluetooth audio output for clear and accessible feedback


## How It Works

1. Image Capture
    Dual cameras capture images of the surrounding environment.

2. Stereo Vision Processing
    Stereo vision techniques extract depth information by comparing the two camera views, inspired by human binocular vision.

3. Object Detection & Distance Estimation
    Objects are detected and their distances estimated using computer vision models.

4. Environmental Description
    Captured images are processed to generate descriptive audio captions.

5. Auditory Feedback
    The processed information is converted into speech and played through a Bluetooth speaker, alerting users about their        surroundings and nearby obstacles.


## Technologies Used

- Hardware: Raspberry Pi 5, Stereo Camera Modules, Bluetooth Speaker

- Programming Language: Python

- Computer Vision: OpenCV

- Stereo Vision Techniques: Camera calibration, rectification, block matching

- Audio Output: Text-to-Speech (TTS)


## Project Goals

- Improve independence and situational awareness for visually impaired users

- Reduce reliance on external assistance

- Provide a cost-effective alternative to depth sensors using stereo vision

- Enhance navigation safety through real-time alerts


## Motivation

This project aims to demonstrate how computer vision and embedded systems can be combined to create meaningful, real-world assistive technology that improves quality of life.
