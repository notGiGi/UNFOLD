# K-Conditioned Decomposition (KCD)

A PyTorch implementation of K-Conditioned Decomposition for unsupervised image layer decomposition using slot attention.

## Overview

K-Conditioned Decomposition (KCD) is a research-grade implementation for decomposing images into K distinct layers or objects in an unsupervised manner. The model learns to:

- **Decompose images** into K semantic layers without supervision
- **Condition on K** during training to handle varying numbers of objects
- **Generalize** to different decomposition granularities at inference time

### Key Components

1. **CNN Encoder**: Extracts spatial features from input images
2. **Slot Attention**: Iteratively binds features to K object slots
3. **Spatial Broadcast Decoder**: Reconstructs images from slot representations

The K-conditioning approach allows the model to be trained with variable numbers of slots (K_min to K_max), enabling better generalization and flexibility at test time.

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- CUDA (optional, for GPU acceleration)

### Install from source

```bash
# Clone the repository
git clone https://github.com/kcd-research/kcd.git
cd kcd

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Kaggle Compatibility

This codebase is designed to be Kaggle-compatible. To use on Kaggle:

1. Upload the repository as a Kaggle dataset
2. Install dependencies in a notebook:
   ```python
   !pip install torch>=2.0.0 torchvision pyyaml tqdm matplotlib
   ```
3. Set paths to your Kaggle input data
4. Run training or evaluation scripts

## Project Structure

```
kcd/
├── src/kcd/              # Source code
│   ├── data/             # Dataset loading
│   ├── models/           # Neural network models
│   │   ├── encoder.py    # CNN encoder
│   │   ├── slot_attention.py  # Slot attention mechanism
│   │   ├── decoder.py    # Spatial broadcast decoder
│   │   └── kcd_model.py  # Main KCD model
│   ├── losses.py         # Loss functions
│   ├── train.py          # Training script
│   ├── eval.py           # Evaluation script
│   └── utils.py          # Utility functions
├── configs/              # Configuration files
│   ├── model.yaml        # Model architecture config
│   └── train.yaml        # Training config
├── scripts/              # Shell scripts
│   ├── train.sh          # Training wrapper
│   └── eval.sh           # Evaluation wrapper
└── outputs/              # Output directory
    ├── checkpoints/      # Model checkpoints
    ├── logs/             # Training logs
    └── figures/          # Visualizations
```

## Usage

### Configuration

Edit configuration files in `configs/`:

- **model.yaml**: Network architecture (encoder dims, slot dim, decoder dims)
- **train.yaml**: Training hyperparameters (lr, batch size, K_min, K_max)

### Training

#### Using shell script (recommended)

```bash
bash scripts/train.sh /path/to/dataset
```

#### Direct Python invocation

```bash
python -m kcd.train \
    --model-config configs/model.yaml \
    --train-config configs/train.yaml \
    --data-root /path/to/dataset
```

#### Training parameters

Key hyperparameters in `configs/train.yaml`:

- `k_min`, `k_max`: Range of slot numbers to sample during training
- `sampling_strategy`: How to sample K (uniform, weighted, curriculum)
- `lr`: Learning rate (default: 0.0004)
- `batch_size`: Batch size (default: 64)
- `max_steps`: Total training steps (default: 500000)

### Evaluation

#### Using shell script (recommended)

```bash
bash scripts/eval.sh \
    outputs/checkpoints/checkpoint_final.pt \
    /path/to/dataset \
    5
```

#### Direct Python invocation

```bash
python -m kcd.eval \
    --checkpoint outputs/checkpoints/checkpoint_final.pt \
    --data-root /path/to/dataset \
    --num-slots 5 \
    --output-dir outputs/figures
```

This will:
- Load the trained model
- Decompose test images into K slots
- Generate visualizations in `outputs/figures/`
- Report reconstruction metrics

### Custom Dataset

To use your own dataset:

1. Organize images in a directory structure:
   ```
   dataset/
   ├── image1.png
   ├── image2.png
   └── ...
   ```

2. Update `configs/train.yaml`:
   ```yaml
   data:
     dataset_name: "custom"
     image_size: [128, 128]  # Your desired size
   ```

3. Run training with `--data-root /path/to/dataset`

## Kaggle Notebook Example

```python
import sys
sys.path.append('/kaggle/input/kcd-repo/kcd')

from kcd.models.kcd_model import KCDModel
from kcd.utils import load_config

# Load model
model_config = load_config('/kaggle/input/kcd-repo/kcd/configs/model.yaml')
model = KCDModel.from_config(model_config)

# Train or evaluate
# ... (see train.py and eval.py for full examples)
```

## Scientific Background

### K-Conditioned Learning

Traditional object-centric models fix the number of slots K at training time, limiting their ability to handle scenes with varying complexity. KCD introduces K-conditioning:

- **Training**: Sample K uniformly from [K_min, K_max] for each batch
- **Inference**: Choose K based on expected scene complexity
- **Benefit**: Single model handles multiple decomposition granularities

### Architecture Details

**Encoder**: Multi-layer CNN that processes RGB images into spatial feature maps.

**Slot Attention**: Iterative attention mechanism that:
1. Initializes K slot vectors randomly
2. Computes attention between slots and spatial features
3. Updates slots via competition and GRU-based refinement
4. Repeats for fixed iterations

**Decoder**: Spatial broadcast decoder that:
1. Broadcasts each slot to full spatial resolution
2. Adds learnable position encodings
3. Decodes to RGB + alpha mask per slot
4. Composites slots via alpha-weighted sum

### Loss Function

The model is trained with reconstruction loss:

```
L = ||I - Î||²
```

where I is the input image and Î is the reconstruction. Additional regularization terms can be added in `losses.py`.

## Development

### Running tests

```bash
pytest tests/
```

### Code formatting

```bash
black src/
ruff check src/
```

### Type checking

```bash
mypy src/
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{kcd2024,
  title={K-Conditioned Decomposition},
  author={KCD Research Team},
  year={2024},
  url={https://github.com/kcd-research/kcd}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This implementation builds on concepts from:
- Slot Attention (Locatello et al., 2020)
- Object-centric learning literature
- Spatial broadcast decoder architectures

## Contact

For questions or issues, please open a GitHub issue.
