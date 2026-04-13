import jax
import jax.numpy as jnp


def _broadcast(src: jnp.ndarray, other: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Mimic PyTorch _broadcast more permissively in JAX."""
    if dim < 0:
        dim = other.ndim + dim

    if src.ndim == 1:
        for _ in range(dim):
            src = jnp.expand_dims(src, 0)

    while src.ndim < other.ndim:
        src = jnp.expand_dims(src, -1)

    shape = []
    for s_dim, o_dim in zip(src.shape, other.shape):
        if s_dim == o_dim:
            shape.append(s_dim)
        elif s_dim == 1:
            shape.append(o_dim)
        else:
            raise ValueError(
                f"Incompatible shapes for broadcasting: {src.shape} vs {other.shape}"
            )

    return jnp.broadcast_to(src, shape)


def scatter_sum(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim: int = -1,
    out: jnp.ndarray | None = None,
    dim_size: int | None = None,
    reduce: str = "sum",
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    bucket_size: int | None = None,
    mode: str | None = None,
) -> jnp.ndarray:
    assert reduce == "sum"

    if dim < 0:
        dim = src.ndim + dim

    if dim == 0 and index.ndim == 1 and index.shape[0] == src.shape[0]:
        index = jnp.asarray(index)
        if out is None:
            if dim_size is None:
                dim_size = 0 if index.size == 0 else int(index.max()) + 1
            out = jnp.zeros((dim_size,) + src.shape[1:], dtype=src.dtype)
        else:
            if dim_size is None:
                dim_size = out.shape[0]
        segment = jax.ops.segment_sum(
            src,
            index,
            num_segments=dim_size,
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices,
            bucket_size=bucket_size,
            mode=mode,
        )
        return out.at[:].add(segment)

    index = _broadcast(index, src, dim)

    if out is None:
        size = list(src.shape)
        if dim_size is not None:
            size[dim] = dim_size
        elif index.size == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = jnp.zeros(size, dtype=src.dtype)

    idx_grids = jnp.meshgrid(*[jnp.arange(s) for s in src.shape], indexing="ij")
    idx_grids[dim] = index
    scatter_indices = tuple(idx_grids)

    return out.at[scatter_indices].add(src)
