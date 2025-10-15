# MIT License
#
# Copyright (c) 2020 Mehran Maghoumi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import torch
import torch.cuda
from numba import jit, prange, float64
from torch.autograd import Function
from numba import cuda
import math

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_cuda(D, gamma, bandwidth, max_i, max_j, n_passes, R):
    """
    :param seq_len: The length of the sequence (both inputs are assumed to be of the same size)
    :param n_passes: 2 * seq_len - 1 (The number of anti-diagonals)
    """
    # Each block processes one pair of examples
    b = cuda.blockIdx.x
    # We have as many threads as seq_len, because the most number of threads we need
    # is equal to the number of elements on the largest anti-diagonal
    tid = cuda.threadIdx.x

    # Compute I, J, the indices from [0, seq_len)

    # The row index is always the same as tid
    I = tid

    # Use float64 for precision
    gamma_f64 = float64(gamma)
    inv_gamma_f64 = 1.0 / gamma_f64

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(n_passes):

        # The index is actually 'p - tid' but need to force it in-bounds
        J = max(0, min(p - tid, max_j - 1))

        # For simplicity, we define i, j which start from 1 (offset from I, J)
        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == p and (I < max_i and J < max_j):
            # Don't compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                r0 = -float64(R[b, i - 1, j - 1]) * inv_gamma_f64
                r1 = -float64(R[b, i - 1, j]) * inv_gamma_f64
                r2 = -float64(R[b, i, j - 1]) * inv_gamma_f64
                rmax = max(max(r0, r1), r2)
                rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                softmin = -gamma_f64 * (math.log(rsum) + rmax)
                R[b, i, j] = float64(D[b, i - 1, j - 1]) + softmin

        # Wait for other threads in this block
        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_backward_cuda(D, R, inv_gamma, bandwidth, max_i, max_j, n_passes, E):
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    # Indexing logic is the same as above, however, the anti-diagonal needs to
    # progress backwards
    I = tid

    for p in range(n_passes):
        # Reverse the order to make the loop go backward
        rev_p = n_passes - p - 1

        # convert tid to I, J, then i, j
        J = max(0, min(rev_p - tid, max_j - 1))

        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == rev_p and (I < max_i and J < max_j):

            if math.isinf(R[k, i, j]):
                R[k, i, j] = -math.inf

            # Don't compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                # Use float64 for precision
                inv_gamma_f64 = float64(inv_gamma)
                
                R_ij = float64(R[k, i, j])
                
                a = math.exp((float64(R[k, i + 1, j]) - R_ij - float64(D[k, i + 1, j])) * inv_gamma_f64)
                b = math.exp((float64(R[k, i, j + 1]) - R_ij - float64(D[k, i, j + 1])) * inv_gamma_f64)
                c = math.exp((float64(R[k, i + 1, j + 1]) - R_ij - float64(D[k, i + 1, j + 1])) * inv_gamma_f64)
                
                E[k, i, j] = float64(E[k, i + 1, j]) * a + float64(E[k, i, j + 1]) * b + float64(E[k, i + 1, j + 1]) * c

        # Wait for other threads in this block
        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTWCUDA(Function):
    """
    CUDA implementation is inspired by the diagonal one proposed in https://ieeexplore.ieee.org/document/8400444:
    "Developing a pattern discovery method in time series data and its GPU acceleration"
    """

    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.cuda.FloatTensor([gamma])
        bandwidth = torch.cuda.FloatTensor([bandwidth])

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        # Prepare the output array
        R = torch.ones((B, N + 2, M + 2), device=dev, dtype=dtype) * math.inf
        R[:, 0, 0] = 0

        # Run the CUDA kernel.
        # Set CUDA's grid size to be equal to the batch size (every CUDA block processes one sample pair)
        # Set the CUDA block size to be equal to the length of the longer sequence (equal to the size of the largest diagonal)
        compute_softdtw_cuda[B, threads_per_block](cuda.as_cuda_array(D.detach()),
                                                   gamma.item(), bandwidth.item(), N, M, n_passes,
                                                   cuda.as_cuda_array(R))
        ctx.save_for_backward(D, R.clone(), gamma, bandwidth)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        D_ = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        D_[:, 1:N + 1, 1:M + 1] = D

        R[:, :, -1] = -math.inf
        R[:, -1, :] = -math.inf
        R[:, -1, -1] = R[:, -2, -2]

        # Use float64 for accumulation
        E = torch.zeros((B, N + 2, M + 2), dtype=torch.float64, device=dev)
        E[:, -1, -1] = 1

        # Grid and block sizes are set same as done above for the forward() call
        compute_softdtw_backward_cuda[B, threads_per_block](cuda.as_cuda_array(D_),
                                                            cuda.as_cuda_array(R),
                                                            1.0 / gamma.item(), bandwidth.item(), N, M, n_passes,
                                                            cuda.as_cuda_array(E))
        E = E[:, 1:N + 1, 1:M + 1]
        # Cast back to original dtype
        E = E.to(dtype)
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None


# ----------------------------------------------------------------------------------------------------------------------
#
# The following is the CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
# Credit goes to Kanru Hua.
# I've added support for batching and pruning.
#
# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def compute_softdtw(D, gamma, bandwidth):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0
    for b in prange(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):

                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                r0 = -R[b, i - 1, j - 1] / gamma
                r1 = -R[b, i - 1, j] / gamma
                r2 = -R[b, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = - gamma * (np.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin
    return R

# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def compute_softdtw_backward(D_, R, gamma, bandwidth):
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1:N + 1, 1:M + 1] = D_
    E[:, -1, -1] = 1
    R[:, :, -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]
    for k in prange(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):

                if np.isinf(R[k, i, j]):
                    R[k, i, j] = -np.inf

                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
    return E[:, 1:N + 1, 1:M + 1]

# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTW(Function):
    """
    CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
    """

    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype)  # dtype fixed
        bandwidth = torch.Tensor([bandwidth]).to(dev).type(dtype)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        R = torch.Tensor(compute_softdtw(D_, g_, b_)).to(dev).type(dtype)
        ctx.save_for_backward(D, R, gamma, bandwidth)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        E = torch.Tensor(compute_softdtw_backward(D_, R_, g_, b_)).to(dev).type(dtype)
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None

# ----------------------------------------------------------------------------------------------------------------------
def manhattan_bce_dist_func(x, y, gamma=1.0, eps=1e-8):
    """
    x: [B, N, D]
    y: [B, M, D]
    Returns: [B, N, M] cost matrix
    - Manhattan (L1) distance for dims 0..D-2
    - Binary cross-entropy for dim D-1 (last dim), assuming probability input
    """
    B, N, D = x.shape
    M = y.shape[1]
    # Expand for pairwise comparison
    x_exp = x.unsqueeze(2).expand(B, N, M, D)
    y_exp = y.unsqueeze(1).expand(B, N, M, D)
    
    # Manhattan distance for all dims except last
    manhattan = torch.abs(x_exp[..., :-1] - y_exp[..., :-1]).sum(-1)
    
    # Binary cross entropy for last dim; assumes values in [0, 1]
    x_prob = x_exp[..., -1].clamp(min=eps, max=1-eps)
    y_prob = y_exp[..., -1].clamp(min=eps, max=1-eps)
    bce = - (y_prob * x_prob.log() + (1 - y_prob) * (1 - x_prob).log())
    # shape: [B, N, M]

    # Combine
    return manhattan + gamma * bce


def manhattan_focal_dist_func(x, y, dist_gamma=1.0, focal_gamma=2.0, focal_alpha=0.25, eps=1e-8):
    """
    x: [B, N, D] predicted sequence
    y: [B, M, D] ground truth sequence
    Returns: [B, N, M] cost matrix
    - Manhattan (L1) distance for dims 0..D-2
    - Focal Loss for dim D-1 (last dim), assuming probability input for x and binary for y
    """
    B, N, D = x.shape
    M = y.shape[1]
    # Expand for pairwise comparison
    x_exp = x.unsqueeze(2).expand(B, N, M, D)
    y_exp = y.unsqueeze(1).expand(B, N, M, D)

    # Manhattan distance for all dims except last
    manhattan = torch.abs(x_exp[..., :-1] - y_exp[..., :-1]).sum(-1)

    # --- Focal Loss for last dim ---
    # p is the predicted probability, y_true is the ground truth label (0 or 1)
    p = x_exp[..., -1].clamp(min=eps, max=1-eps)
    y_true = y_exp[..., -1]

    # Calculate p_t, the model's estimated probability for the ground truth class
    p_t = p * y_true + (1 - p) * (1 - y_true)

    # Calculate focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)
    focal_term = - (1 - p_t).pow(focal_gamma) * p_t.log()

    # Apply alpha weighting
    alpha_t = focal_alpha * y_true + (1 - focal_alpha) * (1 - y_true)
    focal_loss = alpha_t * focal_term
    # shape: [B, N, M]

    # Combine
    return manhattan + dist_gamma * focal_loss


def focal_manhattan_focal_dist_func(x, y, dist_gamma=1.0, focal_gamma=2.0, focal_alpha=0.25, eps=1e-8):
    """
    x: [B, N, D] predicted sequence
    y: [B, M, D] ground truth sequence
    Returns: [B, N, M] cost matrix
    - Manhattan (L1) distance for dims 0..D-2
    - Focal Loss for dim D-1 (last dim), assuming probability input for x and binary for y
    """
    B, N, D = x.shape
    M = y.shape[1]
    # Expand for pairwise comparison
    x_exp = x.unsqueeze(2).expand(B, N, M, D)
    y_exp = y.unsqueeze(1).expand(B, N, M, D)

    # Manhattan distance for all dims except last
    manhattan = torch.abs(x_exp[..., :-1] - y_exp[..., :-1]).sum(-1)

    # --- Focal Loss for last dim ---
    # p is the predicted probability, y_true is the ground truth label (0 or 1)
    p = x_exp[..., -1].clamp(min=eps, max=1-eps)
    y_true = y_exp[..., -1]

    # Calculate p_t, the model's estimated probability for the ground truth class
    p_t = p * y_true + (1 - p) * (1 - y_true)

    # Calculate focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)
    focal_term = - (1 - p_t).pow(focal_gamma) * p_t.log()

    # Apply alpha weighting
    alpha_t = focal_alpha * y_true + (1 - focal_alpha) * (1 - y_true)
    focal_loss = alpha_t * focal_term
    # shape: [B, N, M]
    manhattan = alpha_t.detach() * manhattan
    
    # Combine
    return manhattan + dist_gamma * focal_loss

def manhattan_only_dist_func(x, y, **kwargs):
    """
    Ultra-simple distance function for debugging.
    - Manhattan (L1) distance for dims 0..D-2
    - Ignores the last dimension (touch).
    """
    B, N, D = x.shape
    M = y.shape[1]
    x_exp = x.unsqueeze(2).expand(B, N, M, D)
    y_exp = y.unsqueeze(1).expand(B, N, M, D)
    
    manhattan = torch.abs(x_exp[..., :-1] - y_exp[..., :-1]).sum(-1)
    return manhattan

class SoftDTW(torch.nn.Module):
    """
    The soft DTW implementation that optionally supports CUDA
    """

    def __init__(self, use_cuda, gamma=1.0, normalize=False, bandwidth=None, dist_func=None):
        """
        Initializes a new instance using the supplied parameters
        :param use_cuda: Flag indicating whether the CUDA implementation should be used
        :param gamma: sDTW's gamma parameter
        :param normalize: Flag indicating whether to perform normalization
                          (as discussed in https://github.com/mblondel/soft-dtw/issues/10#issuecomment-383564790)
        :param bandwidth: Sakoe-Chiba bandwidth for pruning. Passing 'None' will disable pruning.
        :param dist_func: Optional point-wise distance function to use. If 'None', then a default Euclidean distance function will be used.
        """
        super(SoftDTW, self).__init__()
        self.normalize = normalize
        self.gamma = gamma
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = use_cuda

        # Set the distance function
        if dist_func is not None:
            self.dist_func = dist_func
        else:
            self.dist_func = SoftDTW._euclidean_dist_func

    def _get_func_dtw(self, x, y):
        """
        Checks the inputs and selects the proper implementation to use.
        """
        bx, lx, dx = x.shape
        by, ly, dy = y.shape
        # Make sure the dimensions match
        assert bx == by  # Equal batch sizes
        assert dx == dy  # Equal feature dimensions

        use_cuda = self.use_cuda

        if use_cuda and (lx > 1024 or ly > 1024):  # We should be able to spawn enough threads in CUDA
                print("SoftDTW: Cannot use CUDA because the sequence length > 1024 (the maximum block size supported by CUDA)")
                use_cuda = False

        # Finally, return the correct function
        return _SoftDTWCUDA.apply if use_cuda else _SoftDTW.apply

    @staticmethod
    def _euclidean_dist_func(x, y):
        """
        Calculates the Euclidean distance between each element in x and y per timestep
        """
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        return torch.pow(x - y, 2).sum(3)

    def forward(self, X, Y):
        """
        Compute the soft-DTW value between X and Y
        :param X: One batch of examples, batch_size x seq_len x dims
        :param Y: The other batch of examples, batch_size x seq_len x dims
        :return: The computed results
        """

        # Check the inputs and get the correct implementation
        func_dtw = self._get_func_dtw(X, Y)

        # Clamp max distance to avoid numerical instability in sdtw, which can
        # happen with high gamma values.
        max_dist = 1e4

        if self.normalize:
            # Stack everything up and run
            x = torch.cat([X, X, Y])
            y = torch.cat([Y, X, Y])
            D = self.dist_func(x, y)
            D = torch.clamp(D, max=max_dist)
            out = func_dtw(D, self.gamma, self.bandwidth)
            out_xy, out_xx, out_yy = torch.split(out, X.shape[0])
            return out_xy - 1 / 2 * (out_xx + out_yy)
        else:
            D_xy = self.dist_func(X, Y)
            D_xy = torch.clamp(D_xy, max=max_dist)
            return func_dtw(D_xy, self.gamma, self.bandwidth)

# ----------------------------------------------------------------------------------------------------------------------
def timed_run(a, b, sdtw):
    """
    Runs a and b through sdtw, and times the forward and backward passes.
    Assumes that a requires gradients.
    :return: timing, forward result, backward result
    """
    from timeit import default_timer as timer

    # Forward pass
    start = timer()
    forward = sdtw(a, b)
    end = timer()
    t = end - start

    grad_outputs = torch.ones_like(forward)

    # Backward
    start = timer()
    grads = torch.autograd.grad(forward, a, grad_outputs=grad_outputs)[0]
    end = timer()

    # Total time
    t += end - start

    return t, forward, grads

# ----------------------------------------------------------------------------------------------------------------------
def profile(batch_size, seq_len_a, seq_len_b, dims, tol_backward, dist_func, bandwith=None):
    sdtw = SoftDTW(False, gamma=1.0, normalize=False, dist_func=dist_func, bandwidth=bandwith)
    sdtw_cuda = SoftDTW(True, gamma=1.0, normalize=False, dist_func=dist_func, bandwidth=bandwith)
    n_iters = 6

    print("Profiling forward() + backward() times for batch_size={}, seq_len_a={}, seq_len_b={}, dims={}...".format(batch_size, seq_len_a, seq_len_b, dims))

    times_cpu = []
    times_gpu = []

    for i in range(n_iters):
        # Create dummy data from standard normal distribution
        a_cpu_val = torch.randn((batch_size, seq_len_a, dims))
        b_cpu_val = torch.randn((batch_size, seq_len_b, dims))

        # Convert the last dimension to probabilities using sigmoid
        # a_cpu_val[..., -1] = torch.sigmoid(a_cpu_val[..., -1])
        # b_cpu_val[..., -1] = torch.sigmoid(b_cpu_val[..., -1])
        a_cpu_val = torch.sigmoid(a_cpu_val)
        b_cpu_val = torch.sigmoid(b_cpu_val)
        
        # Clone and set requires_grad for the tensor to be tested
        a_cpu = a_cpu_val.clone().detach().requires_grad_(True)
        b_cpu = b_cpu_val.clone().detach() # No grad needed for b

        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()

        # GPU
        t_gpu, forward_gpu, backward_gpu = timed_run(a_gpu, b_gpu, sdtw_cuda)

        # CPU
        t_cpu, forward_cpu, backward_cpu = timed_run(a_cpu, b_cpu, sdtw)

        # Verify the results
        assert torch.allclose(forward_cpu, forward_gpu.cpu())

        # Check backward pass
        are_close = torch.allclose(backward_cpu, backward_gpu.cpu(), atol=tol_backward, rtol=1e-5)
        if not are_close:
            print(f"Backward pass results are not close for batch_size={batch_size}, seq_len_a={seq_len_a}, seq_len_b={seq_len_b}, dims={dims}, tol_backward={tol_backward}")
            abs_diff = torch.abs(backward_cpu - backward_gpu.cpu())
            rel_diff = abs_diff / torch.abs(backward_cpu)
            print(f"  Max absolute difference: {torch.max(abs_diff)}")
            # To avoid inf/nan from division by zero
            rel_diff = rel_diff[torch.isfinite(rel_diff)]
            if rel_diff.numel() > 0:
                print(f"  Max relative difference: {torch.max(rel_diff)}")
            
            # Find atol required for it to pass
            required_atol = torch.max(torch.abs(backward_cpu - backward_gpu.cpu()))
            print(f"  Required atol for torch.allclose to pass: {required_atol.item()}")

        assert are_close, "Backward pass results are not close."

        if i > 0:  # Ignore the first time we run, in case this is a cold start (because timings are off at a cold start of the script)
            times_cpu += [t_cpu]
            times_gpu += [t_gpu]

    # Average and log
    avg_cpu = np.mean(times_cpu)
    avg_gpu = np.mean(times_gpu)
    print("  CPU:     ", avg_cpu)
    print("  GPU:     ", avg_gpu)
    print("  Speedup: ", avg_cpu / avg_gpu)
    print()

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    from timeit import default_timer as timer

    torch.manual_seed(1234)

    # Note: The first profile call with non-square inputs is more prone to numerical errors.
    # profile(128, 17, 15, 2, tol_backward=5e-4, dist_func=lambda x, y: manhattan_focal_dist_func(x, y, dist_gamma=1.0))
    
    # Profile with different settings, now using focal loss
    # profile(512, 64, 64, 2, tol_backward=5e-4, dist_func=lambda x, y: manhattan_focal_dist_func(x, y, dist_gamma=1.0))
    # profile(512, 256, 256, 2, tol_backward=5e-4, dist_func=lambda x, y: manhattan_focal_dist_func(x, y, dist_gamma=1.0))
    
    # Test different focal loss gammas
    profile(512, 250, 250, 3, tol_backward=5e-4, dist_func=lambda x, y: manhattan_focal_dist_func(x, y, dist_gamma=1.0, focal_gamma=2.0))
    profile(512, 250, 250, 3, tol_backward=5e-4, dist_func=lambda x, y: manhattan_focal_dist_func(x, y, dist_gamma=0.5, focal_gamma=2.0))
    profile(512, 250, 250, 3, tol_backward=5e-4, dist_func=lambda x, y: manhattan_focal_dist_func(x, y, dist_gamma=0.25, focal_gamma=2.0))
    
    profile(512, 250, 250, 3, tol_backward=5e-4, dist_func=lambda x, y: manhattan_focal_dist_func(x, y, dist_gamma=1.0, focal_gamma=2.0), bandwith=100)
    profile(512, 250, 250, 3, tol_backward=5e-4, dist_func=lambda x, y: manhattan_focal_dist_func(x, y, dist_gamma=0.5, focal_gamma=2.0), bandwith=100)
    profile(512, 250, 250, 3, tol_backward=5e-4, dist_func=lambda x, y: manhattan_focal_dist_func(x, y, dist_gamma=0.25, focal_gamma=2.0), bandwith=100)