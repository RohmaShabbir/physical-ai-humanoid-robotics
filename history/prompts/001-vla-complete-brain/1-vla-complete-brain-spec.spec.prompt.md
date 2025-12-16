---
id: 1
title: "vla-complete-brain-spec"
stage: "spec"
date: "2025-12-12"
surface: "agent"
model: "Claude Sonnet 4.5"
feature: "001-vla-complete-brain"
branch: "001-vla-complete-brain"
user: "user"
command: "/sp.specify MODULE 4 – Vision-Language-Action – The Complete Brain"
labels: ["specification", "vla", "robotics", "vision-language-action"]
links:
  spec: "null"
  ticket: "null"
  adr: "null"
  pr: "null"
files:
  - "specs/001-vla-complete-brain/spec.md"
  - "specs/001-vla-complete-brain/checklists/requirements.md"
tests: []
---

# PHR: vla-complete-brain-spec

## Prompt Text

```
MODULE 4 – Vision-Language-Action – The Complete Brain
Weeks 11–13 · ~100 pages · 5 chapters + Capstone
Chapter 4.1 – VLA Models You Can Actually Run on Jetson in 2025

Open-VLA-7B (Prismatic), RT-2-X, Octo-Small, Gemini-Robotics-1.5
Quantized GGUF versions that fit in 16 GB VRAM
Performance table: tokens/sec vs success rate on common tasks

Chapter 4.2 – Voice → Action End-to-End Pipeline

Faster-Whisper (large-v3) running locally at 8× real-time on Orin
LLM prompt template that turns "Put the apple in the microwave" into structured ROS 2 goals
Skill library + chaining (grasp → navigate → place)

Chapter 4.3 – Making LLMs Reliable for Robotics

Chain-of-thought prompting, self-correction, tool calling (ReAct style)
When to fall back to classical planning or imitation

Chapter 4.4 – Running 7B Models on Jetson with TensorRT-LLM

Converting .safetensors → TensorRT engine step-by-step
Streaming tokens while robot is moving (no blocking)

Chapter 4.5 – Safety, Watchdogs & Real-Robot Deployment

Heartbeat monitor, emergency stop action server
Gradual autonomy levels (shadow mode → shared control → full autonomy)

Capstone Project – "Hey Robot, Tidy the Apartment"
Exact task (same as real 2025 robotics competitions):

Robot starts on charging dock
User says: "The red cup is on the table. Bring it to the kitchen sink."
Robot must perceive, plan, walk, grasp (or scoop), navigate back, place cup in sink
Success if completed in ≤3 minutes in simulation and ≤5 minutes on real hardware (Unitree Go2/G1 or proxy).

Deliverable repo contains:

Single ros2 launch file launch_capstone.py that works both in Isaac Sim and on real Jetson + robot
3 demo videos: simulation, Jetson-only dry run, real robot run
```

## Response Text

Created a comprehensive specification for the Vision-Language-Action (VLA) Complete Brain module. The specification includes:

- User stories for voice command processing, VLA processing, LLM reliability, and safe deployment
- Functional requirements covering VLA models, speech recognition, skill chaining, safety mechanisms
- Key entities like VLA Models, Speech Recognition System, ROS 2 Goals, Skill Library
- Success criteria with measurable outcomes for task completion time, performance, and reliability

The specification is ready for the planning phase and includes a quality checklist to ensure all requirements are met.

## Outcome

Specification created successfully in specs/001-vla-complete-brain/spec.md with quality validation checklist.