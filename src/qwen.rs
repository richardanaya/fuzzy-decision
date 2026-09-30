//! Qwen3-0.6B backbone and the Kev pointer head, executed on Burn's WGPU backend.
//!
//! Weights come from `Qwen/Qwen3-0.6B-Base` with the `jaredpalmer/kev-0.6b`
//! LoRA merged in, plus that checkpoint's pointer head.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::{Tensor, TensorData};

const HIDDEN: usize = 1024;
const LAYERS: usize = 28;
const HEADS: usize = 16;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const INTERMEDIATE: usize = 3072;
const RMS_EPS: f32 = 1e-6;
const ROPE_THETA: f32 = 1_000_000.0;
const POINTER_DIM: usize = 256;
const MAX_POS: usize = 8192;
const GROUP: usize = HEADS / KV_HEADS;

pub struct Qwen3Kev {
    embed: Tensor<Wgpu, 2>,
    layers: Vec<Layer>,
    norm: Tensor<Wgpu, 1>,
    pointer_q: Tensor<Wgpu, 2>,
    pointer_k: Tensor<Wgpu, 2>,
    pointer_q_bias: Tensor<Wgpu, 1>,
    pointer_k_bias: Tensor<Wgpu, 1>,
    head_temperature: f32,
    rope_cos: Tensor<Wgpu, 2>,
    rope_sin: Tensor<Wgpu, 2>,
    device: WgpuDevice,
}

struct Layer {
    q: Tensor<Wgpu, 2>,
    k: Tensor<Wgpu, 2>,
    v: Tensor<Wgpu, 2>,
    o: Tensor<Wgpu, 2>,
    q_norm: Tensor<Wgpu, 1>,
    k_norm: Tensor<Wgpu, 1>,
    gate: Tensor<Wgpu, 2>,
    up: Tensor<Wgpu, 2>,
    down: Tensor<Wgpu, 2>,
    input_norm: Tensor<Wgpu, 1>,
    post_norm: Tensor<Wgpu, 1>,
}

pub struct Packed {
    pub ids: Vec<i32>,
    pub pos: Vec<i32>,
    pub seg: Vec<i32>,
    /// Per question: index of `<decide>`, then the `</opt>` index of each option.
    pub readouts: Vec<(usize, Vec<usize>)>,
}

impl Qwen3Kev {
    pub fn from_parts(
        embed: Tensor<Wgpu, 2>,
        layers: Vec<LayerParts>,
        norm: Tensor<Wgpu, 1>,
        pointer_q: Tensor<Wgpu, 2>,
        pointer_k: Tensor<Wgpu, 2>,
        pointer_q_bias: Tensor<Wgpu, 1>,
        pointer_k_bias: Tensor<Wgpu, 1>,
        head_temperature: f32,
        device: WgpuDevice,
    ) -> Self {
        let (cos, sin) = rope_tables();
        Self {
            embed,
            layers: Self::stash_layers(layers),
            norm,
            pointer_q,
            pointer_k,
            pointer_q_bias,
            pointer_k_bias,
            head_temperature,
            rope_cos: Tensor::from_data(TensorData::new(cos, [MAX_POS, HEAD_DIM]), &device),
            rope_sin: Tensor::from_data(TensorData::new(sin, [MAX_POS, HEAD_DIM]), &device),
            device,
        }
    }

    fn stash_layers(layers: Vec<LayerParts>) -> Vec<Layer> {
        layers.into_iter().map(LayerParts::into_layer).collect()
    }

    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// One logit per option, grouped by question.
    pub fn score(&self, packed: &Packed, user_temperature: f32) -> Vec<Vec<f32>> {
        let hidden = self.backbone(packed);
        let temperature = self.head_temperature * user_temperature.max(1e-6);
        let scale = 1.0 / (POINTER_DIM as f32).sqrt();
        packed
            .readouts
            .iter()
            .map(|(decide, opts)| {
                let h_dec = hidden.clone().narrow(0, *decide, 1);
                let q = linear(&h_dec, &self.pointer_q, Some(&self.pointer_q_bias));
                opts.iter()
                    .map(|opt| {
                        let h_opt = hidden.clone().narrow(0, *opt, 1);
                        let k = linear(&h_opt, &self.pointer_k, Some(&self.pointer_k_bias));
                        let logit = k.mul(q.clone()).sum() * scale / temperature;
                        logit.into_data().as_slice::<f32>().unwrap()[0]
                    })
                    .collect()
            })
            .collect()
    }

    fn backbone(&self, packed: &Packed) -> Tensor<Wgpu, 2> {
        let seq = packed.ids.len();
        let ids = Tensor::<Wgpu, 1, burn::tensor::Int>::from_data(
            TensorData::new(packed.ids.clone(), [seq]),
            &self.device,
        );
        let mut hidden = self.embed.clone().select(0, ids);
        let pos = Tensor::<Wgpu, 1, burn::tensor::Int>::from_data(
            TensorData::new(packed.pos.clone(), [seq]),
            &self.device,
        );
        let cos = self.rope_cos.clone().select(0, pos.clone());
        let sin = self.rope_sin.clone().select(0, pos);
        let mask = branch_mask(&packed.seg, &self.device);

        for layer in &self.layers {
            let normed = rms_norm_2d(hidden.clone(), &layer.input_norm);
            let attn = attention(&normed, layer, &cos, &sin, &mask);
            hidden = hidden + attn;
            let normed = rms_norm_2d(hidden.clone(), &layer.post_norm);
            let ff = mlp(&normed, layer);
            hidden = hidden + ff;
        }
        rms_norm_2d(hidden, &self.norm)
    }
}

pub use self::layer_ctor::LayerParts;

mod layer_ctor {
    use super::*;
    pub struct LayerParts {
        pub q: Tensor<Wgpu, 2>,
        pub k: Tensor<Wgpu, 2>,
        pub v: Tensor<Wgpu, 2>,
        pub o: Tensor<Wgpu, 2>,
        pub q_norm: Tensor<Wgpu, 1>,
        pub k_norm: Tensor<Wgpu, 1>,
        pub gate: Tensor<Wgpu, 2>,
        pub up: Tensor<Wgpu, 2>,
        pub down: Tensor<Wgpu, 2>,
        pub input_norm: Tensor<Wgpu, 1>,
        pub post_norm: Tensor<Wgpu, 1>,
    }
    impl LayerParts {
        pub fn into_layer(self) -> Layer {
            Layer {
                q: self.q,
                k: self.k,
                v: self.v,
                o: self.o,
                q_norm: self.q_norm,
                k_norm: self.k_norm,
                gate: self.gate,
                up: self.up,
                down: self.down,
                input_norm: self.input_norm,
                post_norm: self.post_norm,
            }
        }
    }
}

fn linear(
    x: &Tensor<Wgpu, 2>,
    weight: &Tensor<Wgpu, 2>,
    bias: Option<&Tensor<Wgpu, 1>>,
) -> Tensor<Wgpu, 2> {
    let y = x.clone().matmul(weight.clone().transpose());
    match bias {
        Some(bias) => y + bias.clone().unsqueeze_dim::<2>(0),
        None => y,
    }
}

fn rms_norm_2d(x: Tensor<Wgpu, 2>, weight: &Tensor<Wgpu, 1>) -> Tensor<Wgpu, 2> {
    let variance = x.clone().powf_scalar(2.0).mean_dim(1);
    let normed = x / (variance + RMS_EPS).sqrt();
    normed * weight.clone().unsqueeze_dim::<2>(0)
}

fn rms_norm_last(x: Tensor<Wgpu, 3>, weight: &Tensor<Wgpu, 1>) -> Tensor<Wgpu, 3> {
    let variance = x.clone().powf_scalar(2.0).mean_dim(2);
    let normed = x / (variance + RMS_EPS).sqrt();
    normed * weight.clone().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0)
}

fn mlp(x: &Tensor<Wgpu, 2>, layer: &Layer) -> Tensor<Wgpu, 2> {
    let gate = linear(x, &layer.gate, None);
    let up = linear(x, &layer.up, None);
    let activated = silu(gate) * up;
    linear(&activated, &layer.down, None)
}

fn silu(x: Tensor<Wgpu, 2>) -> Tensor<Wgpu, 2> {
    x.clone() * softmax_sigmoid(x)
}

fn softmax_sigmoid(x: Tensor<Wgpu, 2>) -> Tensor<Wgpu, 2> {
    sigmoid(x)
}

fn attention(
    x: &Tensor<Wgpu, 2>,
    layer: &Layer,
    cos: &Tensor<Wgpu, 2>,
    sin: &Tensor<Wgpu, 2>,
    mask: &Tensor<Wgpu, 2>,
) -> Tensor<Wgpu, 2> {
    let seq = x.dims()[0];
    let q = project_heads(&linear(x, &layer.q, None), seq, HEADS);
    let k = project_heads(&linear(x, &layer.k, None), seq, KV_HEADS);
    let v = project_heads(&linear(x, &layer.v, None), seq, KV_HEADS);
    let q = apply_rope(rms_norm_last(q, &layer.q_norm), cos, sin);
    let k = apply_rope(rms_norm_last(k, &layer.k_norm), cos, sin);
    let k = repeat_kv(k);
    let v = repeat_kv(v);

    let scores = q.clone().matmul(k.swap_dims(1, 2)) / (HEAD_DIM as f32).sqrt();
    let scores = scores + mask.clone().unsqueeze_dim::<3>(0);
    let probs = softmax(scores, 2);
    let mixed = probs.matmul(v);
    let flat = mixed
        .swap_dims(0, 1)
        .reshape([seq, HEADS * HEAD_DIM]);
    linear(&flat, &layer.o, None)
}

fn project_heads(x: &Tensor<Wgpu, 2>, seq: usize, heads: usize) -> Tensor<Wgpu, 3> {
    x.clone().reshape([seq, heads, HEAD_DIM]).swap_dims(0, 1)
}

fn repeat_kv(x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
    let seq = x.dims()[1];
    x.unsqueeze_dim::<4>(2)
        .expand([KV_HEADS, seq, GROUP, HEAD_DIM])
        .swap_dims(1, 2)
        .reshape([HEADS, seq, HEAD_DIM])
}

fn apply_rope(x: Tensor<Wgpu, 3>, cos: &Tensor<Wgpu, 2>, sin: &Tensor<Wgpu, 2>) -> Tensor<Wgpu, 3> {
    let half = HEAD_DIM / 2;
    let x1 = x.clone().narrow(2, 0, half);
    let x2 = x.clone().narrow(2, half, half);
    let rotated = Tensor::cat(vec![-x2, x1], 2);
    let cos = cos.clone().unsqueeze_dim::<3>(0);
    let sin = sin.clone().unsqueeze_dim::<3>(0);
    x * cos + rotated * sin
}

fn branch_mask(seg: &[i32], device: &WgpuDevice) -> Tensor<Wgpu, 2> {
    let n = seg.len();
    let mut mask = vec![0f32; n * n];
    let blocked = f32::NEG_INFINITY;
    for i in 0..n {
        for j in 0..n {
            let allowed = j <= i && (seg[j] == 0 || seg[j] == seg[i]);
            if !allowed {
                mask[i * n + j] = blocked;
            }
        }
    }
    Tensor::from_data(TensorData::new(mask, [n, n]), device)
}

fn rope_tables() -> (Vec<f32>, Vec<f32>) {
    let half = HEAD_DIM / 2;
    let mut cos = vec![0f32; MAX_POS * HEAD_DIM];
    let mut sin = vec![0f32; MAX_POS * HEAD_DIM];
    for pos in 0..MAX_POS {
        for i in 0..half {
            let freq = 1.0 / ROPE_THETA.powf((2 * i) as f32 / HEAD_DIM as f32);
            let angle = pos as f32 * freq;
            let c = angle.cos();
            let s = angle.sin();
            cos[pos * HEAD_DIM + i] = c;
            cos[pos * HEAD_DIM + half + i] = c;
            sin[pos * HEAD_DIM + i] = s;
            sin[pos * HEAD_DIM + half + i] = s;
        }
    }
    (cos, sin)
}


