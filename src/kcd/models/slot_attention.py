"""Slot Attention module for K-conditioned iterative feature binding.

This module implements the core slot attention mechanism that enables
unsupervised object decomposition through competitive binding.
"""

import logging
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SlotAttention(nn.Module):
    """Slot Attention for K-conditioned object-centric decomposition.

    Slot Attention is an iterative attention mechanism that learns to bind
    spatial features to a fixed number of K object-centric slots without
    supervision. The key insight is that competitive normalization forces
    slots to specialize on different objects/layers.

    How Competition Induces Layer Separation:
    -----------------------------------------
    The mechanism relies on three key components working together:

    1. **Softmax Competition Across Slots** (dim=1 normalization):
       - For each spatial location, attention weights sum to 1 across slots
       - This creates a "winner-take-all" dynamic where slots compete
       - Each spatial feature can only strongly attend to ONE slot
       - Forces slots to specialize on non-overlapping image regions

    2. **Iterative Refinement** (T iterations):
       - Initial slots are random → assign features roughly
       - Each iteration: slots bid for features → update based on attention
       - Slots gradually specialize on coherent object parts
       - Competition becomes sharper as slots diverge in representation

    3. **GRU-Based Memory**:
       - Slots maintain state across iterations (via GRU)
       - Allows slots to "remember" previous assignments
       - Provides stability during competition
       - Prevents chaotic reassignment of features

    Mathematical Formulation:
    ------------------------
    For iteration t, given slots S^t ∈ R^(B×K×D) and inputs X ∈ R^(B×N×D):

    1. Compute queries, keys, values:
       Q = Linear_q(LayerNorm(S^t))  # (B, K, D)
       K = Linear_k(LayerNorm(X))     # (B, N, D)
       V = Linear_v(LayerNorm(X))     # (B, N, D)

    2. Compute attention logits:
       A_logits = (Q @ K^T) / sqrt(D)  # (B, K, N)

    3. **CRITICAL: Competitive normalization**
       A_norm = softmax(A_logits, dim=-1)              # Over spatial dim N
       A_norm = A_norm / sum(A_norm, dim=1)            # Over slot dim K
       → Each spatial position distributes probability across K slots
       → Each slot's total attention is balanced

    4. Aggregate features for each slot:
       Updates = A_norm @ V  # (B, K, D)

    5. Update slots with GRU + residual MLP:
       S^(t+1) = GRU(Updates, S^t) + MLP(LayerNorm(S^(t+1)))

    Why This Leads to Object Discovery:
    -----------------------------------
    - **Spatial coherence**: Adjacent pixels with similar features tend to
      attend to the same slot (due to similar K vectors)
    - **Distinctiveness**: Slots must differentiate to capture different
      regions (enforced by competition)
    - **Reconstruction pressure**: Decoder must reconstruct image from slots,
      so slots learn semantically meaningful groupings
    - **K-conditioning**: Variable K during training prevents overfitting to
      specific object counts, improves generalization

    The result: slots automatically discover object/layer boundaries without
    any segmentation supervision.
    """

    def __init__(
        self,
        slot_dim: int = 64,
        num_iterations: int = 3,
        mlp_hidden_dim: int = 128,
        epsilon: float = 1e-8,
        use_implicit_diff: bool = False,
    ) -> None:
        """Initialize K-conditioned Slot Attention.

        Args:
            slot_dim: Dimension of each slot (D). Must match input dimension.
            num_iterations: Number of iterative refinement steps (T).
                          Typical: 3-7 iterations. More iterations = sharper
                          slot specialization but slower training.
            mlp_hidden_dim: Hidden dimension for slot update MLP.
            epsilon: Small constant for numerical stability in normalization.
                    Prevents division by zero.
            use_implicit_diff: If True, stop gradients through slot
                             initialization (experimental).

        Note:
            - slot_dim must match the encoder's embedding_dim
            - Increasing num_iterations improves slot separation but adds cost
            - epsilon should be small (1e-8) but non-zero for stability
        """
        super().__init__()

        self.slot_dim = slot_dim
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.use_implicit_diff = use_implicit_diff

        # Input normalization for stable attention computation
        self.norm_inputs = nn.LayerNorm(slot_dim)

        # Slot normalization before query projection
        self.norm_slots = nn.LayerNorm(slot_dim)

        # MLP normalization for residual updates
        self.norm_mlp = nn.LayerNorm(slot_dim)

        # Attention projections (no bias for symmetry)
        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(slot_dim, slot_dim, bias=False)

        # GRU for slot state updates (provides memory across iterations)
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # MLP for slot refinement (residual connection)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, slot_dim),
        )

        # Initialize weights with appropriate scale
        self._init_weights()

        logger.info(
            f"SlotAttention initialized:\n"
            f"  Slot dimension: {slot_dim}\n"
            f"  Iterations: {num_iterations}\n"
            f"  MLP hidden dim: {mlp_hidden_dim}\n"
            f"  Epsilon: {epsilon}"
        )

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        inputs: torch.Tensor,
        num_slots: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with dynamic K-conditioned slot attention.

        This method performs iterative competitive binding of spatial features
        to K object-centric slots. K is specified per forward pass, enabling
        K-conditioned training.

        Args:
            inputs: Spatial feature tokens of shape (B, N, D) where:
                   - B: batch size
                   - N: number of spatial tokens (from encoder)
                   - D: feature dimension (must match slot_dim)
            num_slots: Number of slots K to use for this forward pass.
                      This can vary between batches during training.

        Returns:
            slots: Final slot representations of shape (B, K, D)
                  Each slot represents one discovered object/layer
            attn: Final attention weights of shape (B, K, N)
                 Shows which spatial features each slot captured

        Note:
            The attention weights reveal the discovered segmentation:
            attn[b, k, :] shows the "mask" for slot k in batch b
        """
        batch_size, num_inputs, input_dim = inputs.shape

        # Normalize inputs for stable attention computation
        # This prevents attention from being dominated by feature magnitude
        inputs = self.norm_inputs(inputs)  # (B, N, D)

        # Initialize K slots randomly for this forward pass
        # Each batch can have different K during training (K-conditioning)
        slots = self._initialize_slots(
            batch_size, num_slots, input_dim, inputs.device
        )  # (B, K, D)

        # Optionally stop gradients through initialization (implicit differentiation)
        if self.use_implicit_diff:
            slots = slots.detach()

        # Pre-compute keys and values (same across all iterations)
        # This saves computation since inputs don't change
        k = self.to_k(inputs)  # (B, N, D)
        v = self.to_v(inputs)  # (B, N, D)

        # Iterative slot refinement
        # Each iteration: slots compete for features → update based on attention
        for iteration in range(self.num_iterations):
            slots_prev = slots

            # Normalize slots before computing queries
            # Ensures queries have consistent scale across iterations
            slots_norm = self.norm_slots(slots)  # (B, K, D)

            # Compute queries from current slot states
            q = self.to_q(slots_norm)  # (B, K, D)

            # --- STEP 1: Compute attention logits ---
            # Dot product between slot queries and spatial keys
            # Higher logit = slot is more interested in that spatial location
            attn_logits = torch.einsum('bid,bjd->bij', q, k)  # (B, K, N)

            # Scale by sqrt(d) for stable gradients (standard attention scaling)
            attn_logits = attn_logits / (self.slot_dim ** 0.5)

            # --- STEP 2: COMPETITIVE NORMALIZATION (KEY MECHANISM) ---
            # This is where slot competition happens!

            # First: softmax over spatial dimension (dim=-1)
            # Each slot's attention over all N spatial locations sums to 1
            # Interpretation: "How much does slot k attend to each location?"
            attn = F.softmax(attn_logits, dim=-1)  # (B, K, N)

            # Add epsilon for numerical stability (prevents NaN in normalization)
            attn = attn + self.epsilon

            # Second: normalize over slot dimension (dim=1)
            # At each spatial location, attention across K slots sums to 1
            # Interpretation: "Each location can only be 'owned' by one slot"
            # This creates COMPETITION: slots fight for spatial features
            attn = attn / attn.sum(dim=1, keepdim=True)  # (B, K, N)

            # After this normalization:
            # - sum(attn[:, :, n]) = 1 for each spatial location n
            # - Slots specialize on different regions to avoid competition
            # - This is the core mechanism that induces object separation!

            # --- STEP 3: Aggregate features for each slot ---
            # Weighted sum of values according to attention
            # Each slot collects features from its attended regions
            updates = torch.einsum('bij,bjd->bid', attn, v)  # (B, K, D)

            # --- STEP 4: Update slot states with GRU ---
            # GRU maintains slot memory across iterations
            # Input: new features (updates), State: previous slot
            # This provides stability during competition
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            )
            slots = slots.reshape(batch_size, num_slots, self.slot_dim)
            # (B, K, D)

            # --- STEP 5: Residual MLP for slot refinement ---
            # Additional non-linear transformation of slots
            # Residual connection helps gradient flow
            slots = slots + self.mlp(self.norm_mlp(slots))  # (B, K, D)

            # After T iterations, slots have converged to represent distinct objects

        # Return final slots and attention weights
        # Attention weights can be visualized as object masks
        return slots, attn

    def _initialize_slots(
        self,
        batch_size: int,
        num_slots: int,
        dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Initialize K slots with random Gaussian vectors.

        Random initialization ensures:
        - No built-in bias toward specific objects
        - Different initializations → exploration during training
        - Slot specialization emerges from competition, not initialization

        Args:
            batch_size: Number of examples in batch (B)
            num_slots: Number of slots to initialize (K)
            dim: Dimension of each slot (D)
            device: Device to place tensor on

        Returns:
            Random slot vectors of shape (B, K, D)
            Drawn from N(0, 1) Gaussian distribution

        Note:
            Each batch gets independent random initialization.
            Different K values create different slot configurations.
            This supports K-conditioned training.
        """
        # Sample from standard Gaussian N(0, 1)
        # Scale is normalized by LayerNorm in first iteration
        slots = torch.randn(
            batch_size,
            num_slots,
            dim,
            device=device,
            dtype=torch.float32,
        )

        return slots
