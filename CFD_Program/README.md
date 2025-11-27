Airflow Sandbox — Interactive 2D Flow Visualizer

A simple, educational “CFD-lite” tool for visualizing how particles move around user-drawn obstacles.

Overview

Airflow Sandbox is an interactive Python program that lets users draw obstacles directly on the canvas and immediately see how airflow reacts around them. Although not a full CFD solver, it aims to provide an intuitive, real-time visualization of streamlines, arrows, and particle motion for beginners learning fluid flow concepts.

The tool updates flow fields dynamically whenever the user draws or modifies the mask, making it act like a lightweight CFD sandbox for classrooms, hobbyists, and online demos.

Features
Interactive Drawing (In-App Mask Creation)

Draw obstacles directly with the mouse using a brush tool.

Brush radius adjustable through UI controls.

When drawing ends, the program automatically recalculates the flow field as if a new mask/geometry was imported.

Flow Visualization Modes

Arrows (Velocity field)

Streamlines

Colormap (Magnitude visualization)
You can toggle any combination through on-screen checkboxes.

Particle Simulation

Small particles advect along the local velocity field.

They automatically flow around the obstacles based on the user-drawn geometry.

Real-Time Updates

As soon as the user finishes drawing, streamlines, arrows, and particle motion all restart and recompute using the new geometry.

Requirements:
Python 3.9+
numpy
matplotlib
Pillow (PIL)

pip install numpy matplotlib pillow
python airflow_sandbox.py

How It Works

This project uses a simplified conceptual flow model inspired by real CFD but optimized for:

real-time feedback

smooth interactive behavior

visual intuition rather than numerical accuracy

The main components include:

A 2D velocity field generated over a rectangular domain.

Obstacles represented as a binary mask drawn by the user.

Streamlines computed using matplotlib’s built-in streamline integrator.

Particles advected frame-by-frame using the local velocity vectors.

A redraw-on-update architecture to keep everything synced.