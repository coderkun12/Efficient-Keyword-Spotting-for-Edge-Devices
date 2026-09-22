"""
Pure-Python INT8 reference model for the weight-stationary MAC array.

Golden vectors come from here, never from the RTL. Every RTL test compares
against these functions, so a bug has to appear in both independently written
implementations to slip through.

The arithmetic is deliberately plain Python integers: INT8 operands into an
INT32 accumulator cannot overflow for any tile we generate (worst case is
conv4 with K = 1152 taps: 1152 * 127 * 128 = 1.87e7, comfortably inside
INT32's 2.1e9), so the model asserts that rather than emulating wraparound.
"""

import random

INT8_MIN, INT8_MAX = -128, 127
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


def matvec(weights, activations):
    """One array output vector.

    weights     : M x K nested list, weights[m][k] = W[m][k]
    activations : length-K list, the aligned input vector X[:, n]
    returns     : length-M list, Y[m] = sum_k W[m][k] * X[k]
    """
    m_count = len(weights)
    k_count = len(activations)
    out = []
    for m in range(m_count):
        assert len(weights[m]) == k_count, (
            f"weight row {m} has {len(weights[m])} taps, expected {k_count}"
        )
        acc = 0
        for k in range(k_count):
            acc += weights[m][k] * activations[k]
        assert INT32_MIN <= acc <= INT32_MAX, f"accumulator overflow: {acc}"
        out.append(acc)
    return out


def matmul(weights, act_vectors):
    """Stream of output vectors, one per input vector."""
    return [matvec(weights, vec) for vec in act_vectors]


def pe_step(w, a, psum_in):
    """Single PE: the value it will register on the next clock edge."""
    return psum_in + w * a


def random_weights(m_count, k_count, rng=None, lo=INT8_MIN, hi=INT8_MAX):
    rng = rng or random
    return [[rng.randint(lo, hi) for _ in range(k_count)] for _ in range(m_count)]


def random_activations(k_count, n_count, rng=None, lo=INT8_MIN, hi=INT8_MAX):
    rng = rng or random
    return [[rng.randint(lo, hi) for _ in range(k_count)] for _ in range(n_count)]


def weight_shift_order(weights, k_count):
    """Per-cycle column values for the weight shift chain.

    Weights shift DOWN one PE per enabled cycle, so the first value pushed ends
    up furthest down the column. To leave PE(k, m) holding W[m][k] after
    k_count shifts, push W[m][K-1] first and W[m][0] last.

    Returns a list of k_count entries; entry i is the list of per-column bytes
    to drive on shift cycle i.
    """
    return [[weights[m][k_count - 1 - i] for m in range(len(weights))]
            for i in range(k_count)]


# --- helpers shared by the testbenches -------------------------------------

def to_signed(value, width):
    """Interpret a raw unsigned RTL read as a two's-complement signed int."""
    value = int(value)
    return value - (1 << width) if value & (1 << (width - 1)) else value


def pack(values, width):
    """Pack a list into one integer, element 0 in the least significant field."""
    packed = 0
    mask = (1 << width) - 1
    for i, v in enumerate(values):
        packed |= (int(v) & mask) << (i * width)
    return packed


def unpack(packed, width, count):
    """Inverse of pack(), returning signed elements."""
    mask = (1 << width) - 1
    return [to_signed((int(packed) >> (i * width)) & mask, width)
            for i in range(count)]


# ---------------------------------------------------------------------------
# Fused write-back: requantise (BatchNorm folded in), ReLU, 2x2 max-pool.
# ---------------------------------------------------------------------------

def requantize(acc, bias, mult, shift, relu=True):
    """One channel's requantisation, exactly as writeback.sv computes it.

    BatchNorm at inference is affine and its constants are known after
    training, so folding it into the convolution turns it into precisely this
    per-channel scale and offset. It therefore costs no hardware beyond the
    requantiser INT8 inference needs anyway.
    """
    product = (acc + bias) * mult
    rnd = (1 << (shift - 1)) if shift > 0 else 0
    shifted = (product + rnd) >> shift          # Python >> on ints is arithmetic
    lo = 0 if relu else -128
    return max(lo, min(127, shifted))


def fused_writeback(acc_vectors, biases, mults, shifts, height, width,
                    relu=True, pool=True):
    """Golden model for writeback.sv.

    acc_vectors is the row-major stream of INT32 accumulator vectors, one per
    pre-pool output pixel, each of length M. Returns the INT8 output stream.

    Pooling floors like torch.nn.MaxPool2d(2): a trailing odd column or row is
    dropped, which is what turns conv2's 40x101 into 20x50.
    """
    m_count = len(acc_vectors[0])
    assert len(acc_vectors) == height * width, (
        f"expected {height * width} vectors, got {len(acc_vectors)}"
    )

    quantised = [
        [requantize(vec[m], biases[m], mults[m], shifts[m], relu)
         for m in range(m_count)]
        for vec in acc_vectors
    ]
    grid = [[quantised[y * width + x] for x in range(width)]
            for y in range(height)]

    if not pool:
        return [grid[y][x] for y in range(height) for x in range(width)]

    out = []
    for py in range(height // 2):
        for px in range(width // 2):
            out.append([
                max(grid[2 * py + dy][2 * px + dx][m]
                    for dy in (0, 1) for dx in (0, 1))
                for m in range(m_count)
            ])
    return out
