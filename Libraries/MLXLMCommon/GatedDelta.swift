//
//  GatedDelta.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gated_delta.py
//

import Foundation
import MLX
import MLXNN

// MARK: - Compute G

func computeGatedDeltaG(_ aLog: MLXArray, _ a: MLXArray, _ dtBias: MLXArray) -> MLXArray {
    let decay = exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
    return decay
}

// MARK: - Metal Kernel

private func makeGatedDeltaKernel(hasMask: Bool) -> MLXFast.MLXFastKernel? {
    let maskSource = hasMask ? "mask[b_idx * T + t]" : "true"

    let source = """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            // q, k: [B, T, Hk, Dk]
            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            // v, y: [B, T, Hv, Dv]
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            // g: [B, T, Hv]
            auto g_ = g + b_idx * T * Hv;
            auto beta_ = beta + b_idx * T * Hv;

            // state_in, state_out: [B, Hv, Dv, Dk]
            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            for (int t = 0; t < T; ++t) {
              if (\(maskSource)) {
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] * g_[hv_idx];
                  kv_mem += state[i] * k_[s_idx];
                }
                kv_mem = simd_sum(kv_mem);

                auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] + k_[s_idx] * delta;
                  out += state[i] * q_[s_idx];
                }
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {
                  y[dv_idx] = static_cast<InT>(out);
                }
              } else {
                y[dv_idx] = static_cast<InT>(0);
              }
              // Increment data pointers to next time step
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += Hv;
              beta_ += Hv;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<StT>(state[i]);
            }
        """

    var inputNames = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if hasMask {
        inputNames.append("mask")
    }

    let suffix = hasMask ? "_mask" : ""

    return MLXFast.metalKernel(
        name: "gated_delta_step\(suffix)",
        inputNames: inputNames,
        outputNames: ["y", "state_out"],
        source: source
    )
}

private final class GatedDeltaKernelManager: Sendable {
    static let shared = GatedDeltaKernelManager()

    let kernel: MLXFast.MLXFastKernel?
    let kernelMasked: MLXFast.MLXFastKernel?

    private init() {
        kernel = makeGatedDeltaKernel(hasMask: false)
        kernelMasked = makeGatedDeltaKernel(hasMask: true)
    }
}

// MARK: - Kernel Dispatch

func gatedDeltaKernel(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = k.dim(0)
    let T = k.dim(1)
    let Hk = k.dim(2)
    let Dk = k.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)
    let inputType = q.dtype
    let stateType = state.dtype

    let selectedKernel: MLXFast.MLXFastKernel?
    var inputs: [MLXArray] = [q, k, v, g, beta, state, MLXArray(T)]
    if let mask {
        selectedKernel = GatedDeltaKernelManager.shared.kernelMasked
        inputs.append(mask)
    } else {
        selectedKernel = GatedDeltaKernelManager.shared.kernel
    }

    guard let kernel = selectedKernel else {
        fatalError("Gated delta kernel not available")
    }

    let outputs = kernel(
        inputs,
        template: [
            ("InT", inputType),
            ("StT", stateType),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid: (32, Dv, B * Hv),
        threadGroup: (32, 4, 1),
        outputShapes: [[B, T, Hv, Dv], state.shape],
        outputDTypes: [inputType, stateType]
    )

    return (outputs[0], outputs[1])
}

// MARK: - Ops Fallback

private func gatedDeltaStepOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let oldState = state
    let decay: MLXArray
    if g.ndim == 2 {
        decay = expandedDimensions(g, axes: [2, 3])
    } else if g.ndim == 3 {
        decay = expandedDimensions(g, axis: -2)
    } else {
        fatalError("Unsupported gating shape \(g.shape)")
    }

    var state = state * decay
    let kvMem = (state * expandedDimensions(k, axis: -2)).sum(axis: -1)
    let delta = (v - kvMem) * expandedDimensions(beta, axis: -1)
    state = state + expandedDimensions(k, axis: -2) * expandedDimensions(delta, axis: -1)
    let y = (state * expandedDimensions(q, axis: -2)).sum(axis: -1)

    if let mask {
        let expandedMask: MLXArray
        if mask.ndim == 1 {
            expandedMask = expandedDimensions(mask, axes: [1, 2, 3])
        } else if mask.ndim == 2 {
            expandedMask = expandedDimensions(mask, axes: [2, 3])
        } else if mask.ndim == 3 {
            expandedMask = expandedDimensions(mask, axis: -1)
        } else {
            fatalError("Unsupported mask shape \(mask.shape)")
        }
        state = MLX.where(expandedMask, state, oldState)
    }

    return (y.asType(q.dtype), state)
}

func gatedDeltaOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = q.dim(0)
    let T = q.dim(1)
    let Hk = q.dim(2)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    var q = q
    var k = k

    let repeatFactor = Hv / Hk
    if repeatFactor > 1 {
        q = repeated(q, count: repeatFactor, axis: -2)
        k = repeated(k, count: repeatFactor, axis: -2)
    }

    var state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)

    var ys = [MLXArray]()
    ys.reserveCapacity(T)

    for t in 0 ..< T {
        let qT = q[0..., t]
        let kT = k[0..., t]
        let vT = v[0..., t]
        let gT = g[0..., t]
        let betaT = beta[0..., t]
        let maskT = mask == nil ? nil : mask![0..., t]

        let (y, newState) = gatedDeltaStepOps(
            q: qT,
            k: kT,
            v: vT,
            g: gT,
            beta: betaT,
            state: state,
            mask: maskT
        )
        ys.append(y)
        state = newState
    }

    let y = MLX.stacked(ys, axis: 1)
    return (y, state)
}

// MARK: - Chunked (WY) Prefill

/// Sequence length from which the chunked form beats the per-token loop on the
/// CPU backend. Measured on an Intel i7-10700K (H=16, D=128): T=16 is 2.4x,
/// T=32 4x, T>=64 6-7x faster; below 16 the fixed per-chunk cost dominates.
let gatedDeltaChunkedMinLength = 16

/// Chunk length of the WY decomposition (matmul size inside a chunk).
let gatedDeltaChunkSize = 64

/// Chunked (WY / UT-transform) evaluation of the gated delta rule. Mathematically
/// identical to `gatedDeltaOps` (relative error ~1e-7 in fp32), but replaces the
/// T sequential rank-1 state updates with ceil(T/C) chunk steps built from
/// matmuls, which is what makes prefill fast on the CPU backend.
///
/// Shapes: q, k `[B, T, H, Dk]` (already repeated to Hv heads), v `[B, T, H, Dv]`,
/// logG and beta `[B, T, H]`, state `[B, H, Dv, Dk]`.
private func gatedDeltaChunked(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    logG: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    chunkSize C: Int
) -> (MLXArray, MLXArray) {
    let B = q.dim(0)
    let T = q.dim(1)
    let H = q.dim(2)
    let N = (T + C - 1) / C
    let pad = N * C - T

    // [B, T, H, ...] -> [B, H, N, C, ...] in fp32. Padded steps carry k = v = beta = 0
    // and logG = 0, so they neither decay nor write the state.
    func toChunks(_ x: MLXArray) -> MLXArray {
        var x = x.asType(.float32)
        x = x.ndim == 4 ? x.transposed(0, 2, 1, 3) : x.transposed(0, 2, 1)
        if pad > 0 {
            var widths = Array(repeating: IntOrPair(0), count: x.ndim)
            widths[2] = IntOrPair((0, pad))
            x = padded(x, widths: widths)
        }
        return x.ndim == 4 ? x.reshaped(B, H, N, C, x.dim(3)) : x.reshaped(B, H, N, C)
    }

    let qc = toChunks(q)
    let kc = toChunks(k)
    let vc = toChunks(v)
    let bc = toChunks(beta)
    let lg = cumsum(toChunks(logG), axis: -1)  // [B, H, N, C] cumulative log-decay

    let lower = tril(MLXArray.ones([C, C], dtype: .bool))
    let strictLower = tril(MLXArray.ones([C, C], dtype: .bool), k: -1)
    let lgDiff = expandedDimensions(lg, axis: -1) - expandedDimensions(lg, axis: -2)
    let decay = exp(MLX.where(lower, lgDiff, MLXArray(-Float.infinity)))  // [B, H, N, C, C]

    let kBeta = kc * expandedDimensions(bc, axis: -1)
    let vBeta = vc * expandedDimensions(bc, axis: -1)
    let a = MLX.where(
        strictLower, matmul(kBeta, kc.swappedAxes(-1, -2)) * decay, MLXArray(Float(0)))
    let tInv = MLXLinalg.triInv(a + MLXArray.eye(C), upper: false, stream: .cpu)
    let u = matmul(tInv, vBeta)  // [B, H, N, C, Dv]
    let w = matmul(tInv, kBeta * expandedDimensions(exp(lg), axis: -1))  // [B, H, N, C, Dk]
    let qk = MLX.where(
        lower, matmul(qc, kc.swappedAxes(-1, -2)) * decay, MLXArray(Float(0)))

    var s = state.asType(.float32).swappedAxes(-1, -2)  // [B, H, Dk, Dv]
    var outputs = [MLXArray]()
    outputs.reserveCapacity(N)
    for n in 0 ..< N {
        let lgN = lg[0..., 0..., n]  // [B, H, C]
        let lgLast = lgN[0..., 0..., (C - 1) ..< C]  // [B, H, 1]
        let vNew = u[0..., 0..., n] - matmul(w[0..., 0..., n], s)
        let inter = matmul(qc[0..., 0..., n] * expandedDimensions(exp(lgN), axis: -1), s)
        outputs.append(inter + matmul(qk[0..., 0..., n], vNew))
        let kDecayed = kc[0..., 0..., n] * expandedDimensions(exp(lgLast - lgN), axis: -1)
        s = s * expandedDimensions(exp(lgLast), axis: -1)
            + matmul(kDecayed.swappedAxes(-1, -2), vNew)
    }

    var y = concatenated(outputs, axis: 2)  // [B, H, N*C, Dv]
    if pad > 0 {
        y = y[0..., 0..., 0 ..< T]
    }
    return (y.transposed(0, 2, 1, 3).asType(q.dtype), s.swappedAxes(-1, -2))
}

// MARK: - Public API

public func gatedDeltaUpdate(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    a: MLXArray,
    b: MLXArray,
    aLog: MLXArray,
    dtBias: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let beta = sigmoid(b).asType(.float32)
    let g = computeGatedDeltaG(aLog, a, dtBias)

    let B = q.dim(0)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    // State kept in fp32 to match Python mlx-lm. Using q.dtype (bf16) loses
    // precision across T-step recurrence, compounding rounding error.
    var state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)
    if state.dtype != .float32 {
        state = state.asType(.float32)
    }

    // Custom Metal kernels only run on the GPU stream. On machines where the
    // default device is the CPU (e.g. the Intel/AMD port), fall back to the ops
    // implementation instead of crashing in mlx-c.
    if Device.defaultDevice().deviceType == .gpu, GatedDeltaKernelManager.shared.kernel != nil {
        return gatedDeltaKernel(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
    }

    // Prefill on the CPU backend: the chunked form turns T sequential state
    // updates into matmuls. Decode (T = 1) and masked batches keep the loop.
    let T = q.dim(1)
    if T >= gatedDeltaChunkedMinLength, mask == nil, a.ndim == 3 {
        var qh = q
        var kh = k
        let repeatFactor = Hv / q.dim(2)
        if repeatFactor > 1 {
            qh = repeated(qh, count: repeatFactor, axis: -2)
            kh = repeated(kh, count: repeatFactor, axis: -2)
        }
        // log of computeGatedDeltaG, computed directly to avoid log(exp(x)) rounding.
        let logG = -exp(aLog.asType(.float32)) * softplus(a + dtBias)
        return gatedDeltaChunked(
            q: qh, k: kh, v: v, logG: logG, beta: beta, state: state,
            chunkSize: gatedDeltaChunkSize)
    }

    return gatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
}
