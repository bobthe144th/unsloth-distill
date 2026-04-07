# “””
DESIGN REQUIREMENTS SPECIFICATION

Distillation-Mode Docker Container with Jupyter Flag Control

Document Version: 1.0
Date: 2026-04-07
Author: CTO/Founder
Status: Requirements Definition

============================================================

1. OVERVIEW
   ============================================================

Purpose:
Provide a Docker container that switches between standard model serving
and frozen-layer distillation training mode based on a runtime flag set
in a Jupyter notebook.

Trigger:
DISTILLATION environment variable or Jupyter cell flag = “on”

Scope:

- Only affects training/fine-tuning workflows
- Standard inference paths unaffected
- Minimal overhead when DISTILLATION = “off”

# ============================================================
2. FUNCTIONAL REQUIREMENTS

2.1 Jupyter Flag Control

Req-2.1.1: Flag Interface

- Flag: DISTILLATION (environment variable or notebook magic)
- Values: “on”, “off” (case-insensitive)
- Default: “off” (standard mode)
- Settable via: Jupyter magic command, env var, or notebook cell variable

Req-2.1.2: Magic Command Support
Example usage in Jupyter cell:
%set_distillation_mode on

Or direct assignment:
DISTILLATION = “on”

Should not require kernel restart

Req-2.1.3: Persistence Across Cells

- Flag persists across notebook cells until explicitly changed
- Can be toggled mid-session without restarting kernel
- Affects all subsequent training operations in same kernel session

2.2 Distillation Mode Behavior

Req-2.2.1: Import Behavior
When DISTILLATION = “on”:
- Import frozen_layer_distillation.py modules
- Import slow_drift_frozen_layers.py
- Import compatibility monitoring (if enabled)
- Initialize SlowDriftTrainer by default
- Load base model in inference mode (required for divergence penalty)

When DISTILLATION = “off”:
- Standard transformers training imports only
- No frozen-layer overhead loaded
- Models train normally (no constraints)

Req-2.2.2: Model Loading
When DISTILLATION = “on”:
- Load fine-tuning model with gradient tracking enabled
- Load base model (frozen, inference-only)
- Initialize divergence penalty computation
- Initialize slow-drift regularizer

When DISTILLATION = “off”:
- Load model normally (standard training)

Req-2.2.3: Training Loop Activation
When DISTILLATION = “on”:
- Replace standard training loop with SlowDriftTrainer
- Inject frozen-layer constraints into loss
- Add drift penalty to scalar loss
- Add post-epoch restoration routine

When DISTILLATION = “off”:
- Use standard training loop (Hugging Face Trainer, torch.optim, etc.)

Req-2.2.4: Logging & Diagnostics
When DISTILLATION = “on”:
- Log frozen-layer drift magnitudes after each epoch
- Log gradient flow health
- Log compatibility metrics (if monitoring enabled)
- Print SlowDriftTrainer initialization summary

When DISTILLATION = “off”:
- Standard training logs only

2.3 Configuration Management

Req-2.3.1: Config Priority (highest to lowest)

1. Jupyter notebook magic / cell variable (DISTILLATION = “on”)
1. Environment variable at container startup (DISTILLATION=on)
1. Config file in notebook directory (.distillation_config.yaml)
1. Default: “off”

Req-2.3.2: Config File Format
If .distillation_config.yaml exists in notebook directory:
distillation_mode: on|off
drift_weight: 0.1
restoration_factor: 0.99
divergence_threshold: 0.15
divergence_weight: 0.05

Req-2.3.3: Runtime Flag Validation

- Must be “on” or “off” (case-insensitive)
- Log warning if invalid value provided
- Default to “off” if invalid
- Provide helpful error message

2.4 Backward Compatibility

Req-2.4.1: No Breaking Changes

- Existing Jupyter notebooks work unchanged when DISTILLATION = “off”
- No modifications to Hugging Face Trainer API
- Standard transformers imports unaffected

Req-2.4.2: Graceful Degradation

- If frozen-layer modules fail to import (missing files), fall back to “off”
- Log warning but continue training in standard mode
- Do not crash the kernel

# ============================================================
3. TECHNICAL REQUIREMENTS

3.1 Docker Image Modifications

Req-3.1.1: Base Image

- Base: pytorch/pytorch:2.1.0-cuda12.1-runtime-ubuntu22.04
- Include: transformers, torch, accelerate, jupyter

Req-3.1.2: Environment Setup

- Set DISTILLATION=off by default in Dockerfile ENV
- Add /frozen_layer_modules to PYTHONPATH
- Pre-install frozen-layer dependencies

Req-3.1.3: Jupyter Extensions

- Install custom Jupyter extension for %set_distillation_mode magic
- Register magic in kernel startup config

3.2 Jupyter Integration

Req-3.2.1: Kernel Startup Script
File: /etc/jupyter/kernel_startup.py

Functionality:
- Check for DISTILLATION env var at kernel startup
- Register %set_distillation_mode magic command
- Register DISTILLATION global variable
- Load DistillationModeController into kernel namespace

Runs: Automatically when Jupyter kernel starts

Req-3.2.2: Magic Command Implementation
Command: %set_distillation_mode

Syntax:
%set_distillation_mode on
%set_distillation_mode off

Behavior:
- Sets global DISTILLATION variable
- Prints confirmation: “Distillation mode: ON/OFF”
- Optionally prints active configuration

Error Handling:
- Reject invalid values with helpful message
- Suggest valid options

Req-3.2.3: Query Command
Command: %distillation_status

Output:
Distillation Mode: ON/OFF
Drift Weight: 0.1
Restoration Factor: 0.99
Base Model Loaded: Yes/No
Fine-tuning Model Loaded: Yes/No

3.3 Module Architecture

Req-3.3.1: Import Guard
File: frozen_layer_modules/**init**.py

Logic:
```python
import os
DISTILLATION_MODE = os.getenv(“DISTILLATION”, “off”).lower() == “on”

```
if DISTILLATION_MODE:
    from .slow_drift_frozen_layers import SlowDriftTrainer
    from .frozen_layer_distillation import AlternatingLayerFreezer
    # ... other imports
else:
    # Lazy imports only on demand
    pass
```
```

Req-3.3.2: Conditional Imports

- All frozen-layer modules lazy-imported (not pre-loaded)
- Only instantiated if DISTILLATION = “on”
- Standard mode never loads these modules

Req-3.3.3: Training Loop Integration
File: training_utils.py

Function: get_trainer()
Input: config dict (including distillation flag)
Output: SlowDriftTrainer (if DISTILLATION=“on”) or standard Trainer

Logic:
`python def get_trainer(model, base_model, config): if config.get("distillation_mode") == "on": from frozen_layer_modules import SlowDriftTrainer return SlowDriftTrainer(model, base_model, config) else: from transformers import Trainer return Trainer(model, config) `

3.4 Performance Requirements

Req-3.4.1: Memory Overhead (Distillation=off)

- No additional memory when DISTILLATION=“off”
- Standard training memory footprint unchanged

Req-3.4.2: Memory Overhead (Distillation=on)

- Base model in memory (inference-only): ~8-9GB for 4B model
- Fine-tuning model with gradients: ~12-14GB for 4B model
- Total: ~20-23GB for both models simultaneously
- Acceptable on single A100 (80GB) or 2x H100 (141GB each)

Req-3.4.3: Compute Overhead (Distillation=on)

- Forward pass through base model: ~10-15% additional per batch
- Divergence penalty computation: <5% additional
- Drift penalty computation: <5% additional
- Total training time overhead: ~15-20% (acceptable for method validation)

Req-3.4.4: Import Time
When DISTILLATION=“off”: <100ms additional startup time
When DISTILLATION=“on”: <500ms additional import time (base model not loaded yet)

# ============================================================
4. INTERFACE REQUIREMENTS

4.1 Jupyter Notebook API

Req-4.1.1: Basic Example

```python
# Cell 1: Import and enable distillation
from frozen_layer_modules import SlowDriftTrainer
%set_distillation_mode on

# Cell 2: Load models
from transformers import AutoModelForCausalLM, AutoTokenizer
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-4B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-4B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-4B")

# Cell 3: Prepare data
dataset = load_dataset(...)
dataloader = DataLoader(dataset, batch_size=16)

# Cell 4: Train with frozen layers
config = {
  "drift_weight": 0.1,
  "restoration_factor": 0.99,
  "divergence_threshold": 0.15,
  "distillation_mode": "on"
}
trainer = SlowDriftTrainer(model, base_model, config)
metrics = trainer.train_epoch(dataloader)
print(metrics)

# Cell 5: Disable distillation for next experiment
%set_distillation_mode off
```

Req-4.1.2: Context Manager (Optional)
Alternative API using context manager:

```python
with DistillationMode(on=True):
    trainer = SlowDriftTrainer(model, base_model, config)
    trainer.train_epoch(dataloader)
# Distillation mode automatically off after context exits
```

4.2 Environment Configuration

Req-4.2.1: Docker Run Command
Standard mode (default):
docker run -it my-distillation-container jupyter lab

Distillation mode (pre-enabled):
docker run -e DISTILLATION=on -it my-distillation-container jupyter lab

Req-4.2.2: Config File Override
Create .distillation_config.yaml in notebook directory:
distillation_mode: on
drift_weight: 0.1
restoration_factor: 0.99

Load in notebook:
from frozen_layer_modules.config import load_config
config = load_config()  # Reads .distillation_config.yaml

# ============================================================
5. ERROR HANDLING & VALIDATION

Req-5.1: Invalid Flag Values
Scenario: DISTILLATION=“maybe”
Action: Log warning, default to “off”
Message: “Invalid DISTILLATION value ‘maybe’. Valid values: ‘on’, ‘off’. Defaulting to ‘off’.”

Req-5.2: Missing Base Model
Scenario: DISTILLATION=“on” but base_model not provided
Action: Raise ValueError with helpful message
Message: “Distillation mode requires base_model parameter. Provide it to SlowDriftTrainer().”

Req-5.3: Module Import Failure
Scenario: frozen_layer_modules import fails (corrupted file, etc.)
Action: Log error, fall back to standard training, continue
Message: “Failed to import frozen_layer_modules. Training in standard mode. Error: {details}”

Req-5.4: Incompatible Model Architecture
Scenario: Trying to use frozen layers on non-transformer model
Action: Raise ValueError with architecture details
Message: “Model must be a Transformer with 32 layers. Got: {architecture}. Distillation mode disabled.”

Req-5.5: Insufficient GPU Memory
Scenario: Loading base model exceeds available VRAM
Action: Check memory before loading, raise informative error
Message: “Distillation mode requires ~20GB VRAM (base + fine-tune models). Available: {available}GB.”

# ============================================================
6. TESTING REQUIREMENTS

6.1 Unit Tests

Test-6.1.1: Flag Parsing

- Test DISTILLATION=“on” → True
- Test DISTILLATION=“off” → False
- Test DISTILLATION=“ON” (uppercase) → True
- Test DISTILLATION=“invalid” → False (with warning)

Test-6.1.2: Config Loading

- Test .distillation_config.yaml exists → loaded
- Test .distillation_config.yaml missing → defaults used
- Test env var overrides config file

Test-6.1.3: Module Imports

- Test frozen-layer modules importable when DISTILLATION=“on”
- Test frozen-layer modules not imported when DISTILLATION=“off”
- Test graceful fallback on import failure

6.2 Integration Tests

Test-6.2.1: Standard Training (DISTILLATION=“off”)

- Run full training loop
- Verify output matches standard Hugging Face Trainer
- Check no frozen-layer overhead

Test-6.2.2: Distillation Training (DISTILLATION=“on”)

- Run full training loop with frozen layers
- Verify frozen-layer drift penalties applied
- Verify post-epoch restoration works
- Check metrics match expected patterns

Test-6.2.3: Mode Switching

- Start with DISTILLATION=“off”
- Switch to DISTILLATION=“on” mid-session
- Verify models load correctly
- Switch back to “off”
- Verify training reverts to standard

6.3 Performance Tests

Test-6.3.1: Memory Usage

- Baseline (DISTILLATION=“off”): ~14GB for 4B model
- Distillation (DISTILLATION=“on”): ~20-23GB for 4B + base model
- Verify acceptable overhead

Test-6.3.2: Training Speed

- Measure steps/second with DISTILLATION=“off”
- Measure steps/second with DISTILLATION=“on”
- Verify overhead <20%

# ============================================================
7. DEPLOYMENT & DELIVERY

7.1 Docker Image Build

Dockerfile structure:
FROM pytorch/pytorch:2.1.0-cuda12.1-runtime-ubuntu22.04

# Install dependencies

RUN pip install transformers torch accelerate jupyter

# Add frozen-layer modules

COPY frozen_layer_modules /frozen_layer_modules
ENV PYTHONPATH=/frozen_layer_modules:$PYTHONPATH

# Add Jupyter startup script

COPY jupyter_startup.py /etc/jupyter/kernel_startup.py

# Set default DISTILLATION mode

ENV DISTILLATION=off

# Expose Jupyter port

EXPOSE 8888

# Launch Jupyter

CMD [“jupyter”, “lab”, “–ip=0.0.0.0”, “–no-browser”, “–allow-root”]

7.2 Files to Include

Required files:

- Dockerfile
- jupyter_startup.py (magic command registration)
- frozen_layer_modules/**init**.py (distillation mode check)
- frozen_layer_modules/slow_drift_frozen_layers.py
- frozen_layer_modules/frozen_layer_distillation.py
- frozen_layer_modules/config.py (config loading)
- training_utils.py (trainer selection logic)
- .dockerignore (exclude test files, **pycache**)

7.3 Documentation

Documentation to provide:

- README.md: Quick-start guide
- docs/distillation_mode.md: Detailed distillation mode reference
- docs/jupyter_api.md: Jupyter notebook examples
- docs/troubleshooting.md: Common issues and solutions

# ============================================================
8. ACCEPTANCE CRITERIA

The Docker container is considered complete when:

[ ] Flag Interface

- DISTILLATION env variable works in Jupyter
- %set_distillation_mode magic command works
- Toggling mode mid-session doesn’t crash kernel

[ ] Distillation Mode Activation

- When DISTILLATION=“on”, frozen-layer modules load
- When DISTILLATION=“off”, standard training path used
- No overhead when distillation mode off

[ ] Training Loop Integration

- SlowDriftTrainer called automatically when DISTILLATION=“on”
- Standard Trainer used when DISTILLATION=“off”
- Loss includes drift penalty when distillation mode on

[ ] Backward Compatibility

- Existing notebooks work unchanged with DISTILLATION=“off”
- No API changes required for standard training
- Graceful fallback on module import failure

[ ] Testing

- All unit tests pass (flag parsing, config loading, imports)
- All integration tests pass (standard and distillation training)
- Memory and performance overhead within spec (<20%)

[ ] Documentation

- README with quick-start example
- API documentation for DistillationModeController
- Jupyter notebook examples provided
- Troubleshooting guide complete

# ============================================================
9. FUTURE EXTENSIONS

Potential enhancements (not in scope for v1.0):

- GPU memory optimization (model sharding if both models >available VRAM)
- Multi-GPU synchronization (distributed training with frozen layers)
- Adaptive drift penalties (auto-tune based on training dynamics)
- Web UI for toggling distillation mode (instead of magic command)
- Integration with Weights & Biases for drift metric logging

============================================================
“””