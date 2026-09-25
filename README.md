
# Invoke - Professional Creative AI Tools for Visual Media


Invoke is a leading creative engine built to empower professionals and enthusiasts alike. Generate and create stunning visual media using the latest AI-driven technologies. Invoke offers an industry leading web-based UI, and serves as the foundation for multiple commercial products.

- Free to use under a commercially-friendly license
- Download and install on compatible hardware
- Generate, refine, iterate on images, and build workflows

![Highlighted Features - Canvas and Workflows](https://github.com/invoke-ai/InvokeAI/assets/31807370/708f7a82-084f-4860-bfbe-e2588c53548d)

### Web Server & UI

Invoke runs a locally hosted web server & React UI with an industry-leading user experience.

### Unified Canvas

The Unified Canvas is a fully integrated canvas implementation with support for all core generation capabilities, in/out-painting, brush tools, and more. This creative tool unlocks the capability for artists to create with AI as a creative collaborator, and can be used to augment AI-generated imagery, sketches, photography, renders, and more.

### Workflows & Nodes

Invoke offers a fully featured workflow management solution, enabling users to combine the power of node-based workflows with the ease of a UI. This allows for customizable generation pipelines to be developed and shared by users looking to create specific workflows to support their production use-cases.

### Board & Gallery Management

Invoke features an organized gallery system for easily storing, accessing, and remixing your content in the Invoke workspace. Images can be dragged/dropped onto any Image-base UI element in the application, and rich metadata within the Image allows for easy recall of key prompts or settings used in your workflow.

### Model Support
- Z-Image Turbo
- Z-Image Base
- Krea 2 Turbo
- Krea 2 Raw
- Anima
- Qwen Image 2.1
- Ideogram 4
- ERNIE-Image
- ERNIE-Image-Turbo
- Wan 2.2 (5B / 12B)

> **Note:** This fork focuses on maintained model families. SD1.x, SD2.x, SDXL, FLUX.1, FLUX.2, SD3.5, and related variants are being removed from this codebase. If you need those architectures, use the upstream `invoke-ai/InvokeAI` release line.

### Platform Support
- Windows 11+
- Ubuntu 24.04+
- Arch Linux (rolling)
- Python 3.13 (stable), 3.14 / 3.15 (WIP - Qt6 refactor may change requirements)
- CUDA 13.4
- ROCm 10
- Intel XPU

### Other features

- Support for ckpt, diffusers, and some gguf models
- Upscaling Tools
- Embedding Manager & Support
- Model Manager & Support
- Workflow creation & management
- Node-Based Architecture
- Object Segmentation & Selection Models (SAM / SAM2)

## Custom Nodes

Copy your node packs to the `invokeai/app/invocations/custom_nodes/` directory.

When nodes are added or changed, you must restart the app to see the changes.

### Directory Structure

For a node pack to be loaded, it must be placed in a directory alongside this file. Here's an example structure:

```
.
├── __init__.py # Invoke-managed custom node loader
│
├── cool_node
│   ├── __init__.py # see example below
│   └── cool_node.py
│
└── my_node_pack
    ├── __init__.py # see example below
    ├── tasty_node.py
    ├── bodacious_node.py
    ├── utils.py
    └── extra_nodes
        └── fancy_node.py
```

### Node Pack `__init__.py`

Each node pack must have an `__init__.py` file that imports its nodes.

The structure of each node or node pack is otherwise not important.

Here are examples, based on the example directory structure.

#### `cool_node/__init__.py`

```py
from .cool_node import CoolInvocation
```

#### `my_node_pack/__init__.py`

```py
from .tasty_node import TastyInvocation
from .bodacious_node import BodaciousInvocation
from .extra_nodes.fancy_node import FancyInvocation
```

Only nodes imported in the `__init__.py` file are loaded.

