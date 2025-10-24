"""
Flax implementation of the NSA transformer model.
Replaces PyTorch implementation with JAX/Flax for reduced memory footprint.
"""

import math

import jax
import jax.numpy as jnp
import flax.linen as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence data."""

    n_embd: int
    max_len: int = 6500

    @nn.compact
    def __call__(self, x):
        """
        Add sinusoidal positional encodings to input embeddings.

        Args:
            x: Input tensor of shape [batch, seq_len, n_embd]

        Returns:
            Tensor with positional encodings added, shape [batch, seq_len, n_embd]
        """
        batch_size, seq_len, _ = x.shape

        # Generate position indices
        position = jnp.arange(0, self.max_len, dtype=jnp.float32)[:, None]

        # Generate div_term for sin/cos
        div_term = jnp.exp(
            jnp.arange(0, self.n_embd, 2, dtype=jnp.float32) * (-math.log(10000.0) / self.n_embd)
        )

        # Create positional encoding matrix
        pe = jnp.zeros((self.max_len, self.n_embd))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        # Add positional encoding to input (broadcast over batch dimension)
        return x + pe[None, :seq_len, :]


class FlaxTransformerEncoder(nn.Module):
    """Single transformer encoder layer with self-attention and feedforward."""

    n_embd: int = 512
    n_head: int = 8
    n_inner: int = 2048
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, mask: jnp.ndarray | None = None, deterministic: bool = True):
        """
        Apply transformer encoder layer.

        Args:
            x: Input tensor of shape [batch, seq_len, n_embd]
            mask: Optional attention mask of shape [batch, seq_len] (1 for valid, 0 for padding)
            deterministic: If True, disable dropout

        Returns:
            Output tensor of shape [batch, seq_len, n_embd]
        """
        # Convert mask to attention bias format if provided
        # Flax expects mask shape [batch, num_heads, seq_len, seq_len] or broadcastable
        attention_bias = None
        if mask is not None:
            # Convert [batch, seq_len] to [batch, 1, 1, seq_len] for broadcasting
            mask_expanded = mask[:, None, None, :]  # [batch, 1, 1, seq_len]
            # Create attention bias: 0 for valid positions, large negative for padding
            attention_bias = jnp.where(mask_expanded, 0.0, -1e10)

        # Self-attention block
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.n_head,
            qkv_features=self.n_embd,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
        )(x, mask=attention_bias)

        # Add & Norm
        x = nn.LayerNorm()(x + attn_output)

        # Feedforward block
        ff_output = nn.Dense(self.n_inner)(x)
        ff_output = nn.gelu(ff_output)
        ff_output = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(ff_output)
        ff_output = nn.Dense(self.n_embd)(ff_output)
        ff_output = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(ff_output)

        # Add & Norm
        x = nn.LayerNorm()(x + ff_output)

        return x


class FlaxCustomTransformer(nn.Module):
    """
    Main transformer model with 3 classification tokens.
    Matches PyTorch architecture: 25.3M parameters.
    """

    vocab_size: int
    n_embd: int = 512
    n_layer: int = 8
    n_head: int = 8
    dropout_rate: float = 0.0
    num_cls_tokens: int = 3

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
        deterministic: bool = True,
    ):
        """
        Forward pass through the transformer.

        Args:
            input_ids: Token IDs of shape [batch, seq_len]
            attention_mask: Optional mask of shape [batch, seq_len] (1 for valid, 0 for padding)
            deterministic: If True, disable dropout

        Returns:
            Logits of shape [batch, num_cls_tokens, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Embedding layer
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.n_embd)(input_ids)

        # Positional encoding
        x = SinusoidalPositionalEncoding(n_embd=self.n_embd)(x)

        # Dropout
        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)

        # Create and prepend CLS tokens
        cls_tokens = self.param(
            "cls_tokens",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_cls_tokens, self.n_embd),
        )
        # Broadcast CLS tokens to batch size using jnp.tile
        cls_tokens_batch = jnp.tile(cls_tokens, (batch_size, 1, 1))
        x = jnp.concatenate([cls_tokens_batch, x], axis=1)

        # Extend attention mask for CLS tokens if provided
        if attention_mask is not None:
            cls_mask = jnp.ones((batch_size, self.num_cls_tokens), dtype=attention_mask.dtype)
            attention_mask = jnp.concatenate([cls_mask, attention_mask], axis=1)

        # Transformer encoder layers
        for _ in range(self.n_layer):
            x = FlaxTransformerEncoder(
                n_embd=self.n_embd,
                n_head=self.n_head,
                n_inner=4 * self.n_embd,
                dropout_rate=self.dropout_rate,
            )(x, mask=attention_mask, deterministic=deterministic)

        # Extract CLS token outputs
        cls_outputs = x[:, : self.num_cls_tokens, :]

        # Project to vocabulary size
        logits = nn.Dense(self.vocab_size)(cls_outputs)

        return logits


def count_parameters(params):
    """
    Count the total number of parameters in a Flax model.

    Args:
        params: Flax parameter tree

    Returns:
        Total number of parameters
    """
    return sum(x.size for x in jax.tree.leaves(params))


def print_parameter_breakdown(params, prefix=""):
    """
    Print a breakdown of parameters by layer.

    Args:
        params: Flax parameter tree
        prefix: Prefix for nested parameter names
    """
    if isinstance(params, dict):
        for key, value in params.items():
            new_prefix = f"{prefix}/{key}" if prefix else key
            print_parameter_breakdown(value, new_prefix)
    else:
        # Leaf node - print parameter count
        param_count = params.size if hasattr(params, "size") else 0
        print(f"{prefix}: {param_count:,} parameters")
