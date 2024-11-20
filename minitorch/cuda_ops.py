# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Convert jit to device."""
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """Convert function to jit"""
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Calls tensor_zip function."""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Calls tensor_reduce."""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Calls the tensor matrix_multipy function."""
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")

        # skip thread if out of bounds
        if i >= out_size:
            return

        # Check if shapes and strides are aligned without numpy library
        shape_same = True
        for d in range(len(out_shape)):
            if out_shape[d] != in_shape[d]:
                shape_same = False
                break

        strides_same = True
        for d in range(len(out_strides)):
            if out_strides[d] != in_strides[d]:
                strides_same = False
                break

        # stride aligned
        if shape_same and strides_same:
            out[i] = fn(in_storage[i])
        else:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            o = index_to_position(out_index, out_strides)
            j = index_to_position(in_index, in_strides)
            out[o] = fn(in_storage[j])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")

        # skip thread if out of bounds
        if i >= out_size:
            return

        # check if shape is the same
        shape_same = True
        for d in range(len(out_shape)):
            if out_shape[d] != a_shape[d] or a_shape[d] != b_shape[d]:
                shape_same = False
                break

        # check if shape is the same
        strides_same = True
        for d in range(len(out_strides)):
            if out_strides[d] != a_strides[d] or a_strides[d] != b_strides[d]:
                strides_same = False
                break

        if shape_same and strides_same:  # Check if out and in are stride-aligned
            out[i] = fn(a_storage[i], b_storage[i])
        else:
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            j = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            k = index_to_position(b_index, b_strides)
            out[o] = fn(a_storage[j], b_storage[k])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""A practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # TODO: Implement for Task 3.3.
    # raise NotImplementedError("Need to implement for Task 3.3")

    # load in cache if i is not out of bounds
    if i < size:
        cache[pos] = a[i]
    else:  # pad with zeros if i is out of bounds
        cache[pos] = 0.0

    # Synchronize threads within the block to ensure all data is loaded
    cuda.syncthreads()

    # Perform parallel reduction
    # pairs are computed two at a time
    stride = 1
    while stride < BLOCK_DIM:
        if pos % (2 * stride) == 0 and pos + stride < BLOCK_DIM:
            cache[pos] = cache[pos] + cache[pos + stride]
        stride *= 2
        cuda.syncthreads()

    # Write the reduced result to global memory
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Run the _sum_practice stencil code."""
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024  # Number of threads per block
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)  # Shared memory
        out_index = cuda.local.array(MAX_DIMS, numba.int32)  # Local array for indexing
        out_pos = (
            cuda.blockIdx.x
        )  # Index of the current block which maps to the output position
        pos = cuda.threadIdx.x  # Index of the current thread within a block

        # map the out_pos which is the block id to a position
        to_index(out_pos, out_shape, out_index)
        o = index_to_position(out_index, out_strides)

        # Reduction size (number of elements along the reduce_dim)
        reduce_size = a_shape[reduce_dim]

        # Load elements into shared memory
        reduce_index = pos
        if reduce_index < reduce_size:
            # Compute the position in the input storage for this reduction element
            a_index = (
                index_to_position(out_index, a_strides)
                + reduce_index * a_strides[reduce_dim]
            )
            cache[pos] = a_storage[a_index]
        else:
            cache[pos] = reduce_value  # Initialize excess threads with neutral value

        # Synchronize threads after loading data into shared memory
        cuda.syncthreads()

        # Perform parallel reduction
        # pairs are computed two at a time
        stride = 1
        while stride < BLOCK_DIM:
            if pos % (2 * stride) == 0 and pos + stride < BLOCK_DIM:
                cache[pos] = fn(cache[pos], cache[pos + stride])
            stride *= 2
            cuda.syncthreads()

        # Write the reduced result to global memory
        if pos == 0:
            out[o] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    # raise NotImplementedError("Need to implement for Task 3.3")

    a_cache = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_cache = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if i < size and j < size:  # guard for threads
        a_cache[i, j] = a[i * size + j]  # load a into cache
        b_cache[i, j] = b[i * size + j]  # load b into cache

        cuda.syncthreads()

        out_val = 0.0
        for k in range(size):  # loop through k to build up output val
            out_val += a_cache[i, k] * b_cache[k, j]

        out[i * size + j] = out_val  # assign output val


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Calls the _mm_practice function."""
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.
    # raise NotImplementedError("Need to implement for Task 3.4")

    c_val = 0.0  # accumulate result of dot product

    # Loop over blocks of the shared dimension
    shared_dim = a_shape[-1]  # This equals b_shape[-2]
    num_blocks = (
        shared_dim + BLOCK_DIM - 1
    ) // BLOCK_DIM  # number of blocks needed to cover shared dimension

    # loop over blocks of shared dimension
    for block_idx in range(num_blocks):
        a_shared[pi, pj] = 0.0  # Initialize memory element in the local block a
        b_shared[pi, pj] = 0.0  # Initialize memory element in the local block a

        # Load block a into shared memory
        # Ensure thread is within the bounds of A's rows and columns for the current block
        if i < a_shape[-2] and (block_idx * BLOCK_DIM + pj) < a_shape[-1]:
            # Calculate the global memory index for a's current element
            # Run this calculation using strides to convert index to position
            a_idx = (
                batch * a_batch_stride
                + i * a_strides[-2]
                + (block_idx * BLOCK_DIM + pj) * a_strides[-1]
            )
            # Load global value into shared memory
            a_shared[pi, pj] = a_storage[a_idx]

        # Load B block into shared memory
        if j < b_shape[-1] and (block_idx * BLOCK_DIM + pi) < b_shape[-2]:
            # Calculate the global memory index for b's current element
            # Run this calculation using strides to convert index to position
            b_idx = (
                batch * b_batch_stride
                + (block_idx * BLOCK_DIM + pi) * b_strides[-2]
                + j * b_strides[-1]
            )
            # Load global value into shared memory
            b_shared[pi, pj] = b_storage[b_idx]

        # Synchronize threads to ensure all shared memory loads are complete
        cuda.syncthreads()

        # Compute partial dot product for c[i, j] using shared blocks
        for k in range(BLOCK_DIM):
            c_val += a_shared[pi, k] * b_shared[k, pj]

        # Synchronize threads to prepare for the next block load
        cuda.syncthreads()

    # Write the computed value to the global memory
    if i < out_shape[-2] and j < out_shape[-1]:  # Guard for threads
        # Calculate global position in out suing strides and batch
        out_idx = batch * out_strides[0] + i * out_strides[-2] + j * out_strides[-1]
        out[out_idx] = c_val  # one write to global memory per kernel


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
