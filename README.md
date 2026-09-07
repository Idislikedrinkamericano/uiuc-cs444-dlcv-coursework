# UIUC CS 444 — Deep Learning for Computer Vision (Fall 2025)

> **Private coursework archive. Do not redistribute or make public.**
>
> The original course repositories explicitly instructed students to work from private clones and to prevent other students from copying their work. This repository is therefore intended only as a personal archival copy of my completed/attempted coursework.

## Overview

This repository contains coursework from **CS 444 / ECE 494: Deep Learning for Computer Vision** at the University of Illinois Urbana-Champaign, Fall 2025.

The assignments move from classical image classification and neural-network fundamentals through modern computer-vision architectures and generative models.

| Assignment | Main topics represented in the files |
| --- | --- |
| `mp1` | Nearest-neighbor classification, linear classifiers, feature extraction, NumPy vectorization |
| `mp2` | Fully connected networks, convolutional neural networks, backpropagation, PyTorch CNNs, transfer learning |
| `mp3` | Self-attention, layer normalization, Vision Transformers, transfer learning, model design |
| `mp4` | ROIAlign, anchor boxes, box regression, NMS, RetinaNet-style object detection |
| `mp5` | Variational autoencoders, score matching, diffusion sampling, classifier-free guidance, visual anagrams |

## Repository structure

```text
.
├── mp1/   # Classical classifiers and feature representations
├── mp2/   # Neural networks and CNNs
├── mp3/   # Vision Transformers and transfer learning
├── mp4/   # Object detection
└── mp5/   # VAEs, score matching, and diffusion models
```

Each assignment directory retains the original course README and supporting files where useful for context. Those instructions and any starter/scaffolding code originated from the course staff. Student work is mixed into the assignment files at the locations designated by the course templates, along with locally generated reports, figures, notebooks, predictions, and experiment outputs.

## Archive cleanup

To keep the repository lightweight and avoid storing machine-specific or reproducible artifacts, this archive excludes:

- local Python virtual environments (`.venv/`, `venv/`)
- Python caches and macOS metadata
- TensorBoard / training-run logs
- large downloaded datasets used by MP1 and MP2
- MP4 autograder/test-data binaries
- the precomputed MP5 prompt-embedding binary

The original assignment instructions describe the expected environment and, where applicable, how course-provided data/assets were obtained.

## Notes

- This is an **archived course repository**, not a standalone software package.
- Some files include course-provided scaffolding alongside student implementations; this repository should not be interpreted as claiming authorship of the entire codebase.
- No license is provided for redistribution of course materials or starter code.
- The archive may include partially completed or experimental work; it is preserved as coursework rather than presented as a polished production project.
