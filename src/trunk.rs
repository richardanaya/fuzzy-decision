//! LFM2.5 encoder trunk: short convolutions, grouped-query attention, SwiGLU.

use burn::tensor::{Device, Tensor, TensorData};

use crate::nn::{
    attend_prefix, load_tensor, repeat_kv, rotate_half, silu, tensor2, Linear, RmsNorm,
};
use crate::weights::{LayerKind, TensorSource, TrunkSpec};

pub struct Trunk {
    layers: Vec<Block>,
    embedding_norm: RmsNorm,
    head_dim: usize,
    rope_theta: f32,
    device: Device,
}

impl Trunk {
    pub fn load(
        source: &impl TensorSource,
        spec: &TrunkSpec,
        device: &Device,
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(spec.layers.len());
        for (index, kind) in spec.layers.iter().enumerate() {
            layers.push(Block::load(
                source,
                &format!("encoder.layers.{index}"),
                spec,
                *kind,
                device,
            )?);
        }
        Ok(Self {
            layers,
            embedding_norm: RmsNorm::load(source, "encoder.embedding_norm", spec.eps, device)?,
            head_dim: spec.hidden / spec.heads,
            rope_theta: spec.rope_theta,
            device: device.clone(),
        })
    }

    /// `hidden` is `(seq, dim)`. `prefix` is the media length; text-only calls pass 0.
    pub fn forward(&self, hidden: Tensor<2>, prefix: usize) -> Tensor<2> {
        let seq = hidden.dims()[0];
        let (cos, sin) = rope_tables(seq, self.head_dim, self.rope_theta, &self.device);
        let mut hidden = hidden;
        for layer in &self.layers {
            hidden = layer.forward(hidden, &cos, &sin, prefix);
        }
        self.embedding_norm.forward_2d(hidden)
    }
}

struct Block {
    mixer: Mixer,
    ffn: Mlp,
    op_norm: RmsNorm,
    ffn_norm: RmsNorm,
}

impl Block {
    fn load(
        source: &impl TensorSource,
        prefix: &str,
        spec: &TrunkSpec,
        kind: LayerKind,
        device: &Device,
    ) -> Result<Self, String> {
        let mixer = match kind {
            LayerKind::Conv => Mixer::Conv(ShortConv::load(
                source,
                &format!("{prefix}.conv"),
                spec.hidden,
                device,
            )?),
            LayerKind::Attention => Mixer::Attention(Attention::load(
                source,
                &format!("{prefix}.self_attn"),
                spec,
                device,
            )?),
        };
        Ok(Self {
            mixer,
            ffn: Mlp::load(
                source,
                &format!("{prefix}.feed_forward"),
                spec.ffn_dim(),
                device,
            )?,
            op_norm: RmsNorm::load(source, &format!("{prefix}.operator_norm"), spec.eps, device)?,
            ffn_norm: RmsNorm::load(source, &format!("{prefix}.ffn_norm"), spec.eps, device)?,
        })
    }

    fn forward(
        &self,
        hidden: Tensor<2>,
        cos: &Tensor<2>,
        sin: &Tensor<2>,
        prefix: usize,
    ) -> Tensor<2> {
        let mixed = match &self.mixer {
            Mixer::Conv(conv) => conv.forward(self.op_norm.forward_2d(hidden.clone()), prefix),
            Mixer::Attention(attn) => {
                attn.forward(self.op_norm.forward_2d(hidden.clone()), cos, sin, prefix)
            }
        };
        let hidden = hidden + mixed;
        hidden.clone() + self.ffn.forward(self.ffn_norm.forward_2d(hidden))
    }
}

enum Mixer {
    Conv(ShortConv),
    Attention(Attention),
}

struct ShortConv {
    in_proj: Linear,
    out_proj: Linear,
    weight: Tensor<3>,
    device: Device,
}

impl ShortConv {
    fn load(
        source: &impl TensorSource,
        prefix: &str,
        hidden: usize,
        device: &Device,
    ) -> Result<Self, String> {
        let _ = hidden;
        Ok(Self {
            in_proj: Linear::load(source, &format!("{prefix}.in_proj"), device, false)?,
            out_proj: Linear::load(source, &format!("{prefix}.out_proj"), device, false)?,
            weight: load_tensor(source, &format!("{prefix}.conv.weight"), device)?,
            device: device.clone(),
        })
    }

    fn forward(&self, x: Tensor<2>, prefix: usize) -> Tensor<2> {
        let seq = x.dims()[0];
        let hidden = x.dims()[1];
        let mixed = self.in_proj.forward(x);
        let b = mixed.clone().narrow(1, 0, hidden);
        let c = mixed.clone().narrow(1, hidden, hidden);
        let u = mixed.narrow(1, 2 * hidden, hidden);
        let bx = (b * u).swap_dims(0, 1);
        let pad = Tensor::<2>::zeros([hidden, 1], &self.device);
        let xp = Tensor::cat(vec![pad.clone(), bx, pad], 1);
        let left = xp.clone().narrow(1, 0, seq);
        let mid = xp.clone().narrow(1, 1, seq);
        let right = xp.narrow(1, 2, seq);
        let taps = self.weight.clone().reshape([hidden, 3]);
        let w0 = taps.clone().narrow(1, 0, 1);
        let w1 = taps.clone().narrow(1, 1, 1);
        let w2 = taps.narrow(1, 2, 1);
        let keep = tensor2(keep_right(seq, prefix), 1, seq, &self.device);
        let y = left * w0 + mid * w1 + right * keep * w2;
        self.out_proj.forward(c * y.swap_dims(0, 1))
    }
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(
        source: &impl TensorSource,
        prefix: &str,
        spec: &TrunkSpec,
        device: &Device,
    ) -> Result<Self, String> {
        Ok(Self {
            q: Linear::load(source, &format!("{prefix}.q_proj"), device, false)?,
            k: Linear::load(source, &format!("{prefix}.k_proj"), device, false)?,
            v: Linear::load(source, &format!("{prefix}.v_proj"), device, false)?,
            out: Linear::load(source, &format!("{prefix}.out_proj"), device, false)?,
            q_norm: RmsNorm::load(source, &format!("{prefix}.q_layernorm"), spec.eps, device)?,
            k_norm: RmsNorm::load(source, &format!("{prefix}.k_layernorm"), spec.eps, device)?,
            heads: spec.heads,
            kv_heads: spec.kv_heads,
            head_dim: spec.hidden / spec.heads,
        })
    }

    fn forward(
        &self,
        x: Tensor<2>,
        cos: &Tensor<2>,
        sin: &Tensor<2>,
        prefix: usize,
    ) -> Tensor<2> {
        let seq = x.dims()[0];
        let q = self
            .q_norm
            .forward_3d(
                self.q
                    .forward(x.clone())
                    .reshape([seq, self.heads, self.head_dim]),
            )
            .swap_dims(0, 1);
        let k = self
            .k_norm
            .forward_3d(
                self.k
                    .forward(x.clone())
                    .reshape([seq, self.kv_heads, self.head_dim]),
            )
            .swap_dims(0, 1);
        let v = self
            .v
            .forward(x)
            .reshape([seq, self.kv_heads, self.head_dim])
            .swap_dims(0, 1);
        let q = apply_rope(q, cos, sin);
        let k = apply_rope(k, cos, sin);
        let groups = self.heads / self.kv_heads;
        let y = attend_prefix(
            q,
            repeat_kv(k, groups),
            repeat_kv(v, groups),
            prefix,
            (self.head_dim as f32).powf(-0.5),
        );
        self.out
            .forward(y.swap_dims(0, 1).reshape([seq, self.heads * self.head_dim]))
    }
}

struct Mlp {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl Mlp {
    fn load(
        source: &impl TensorSource,
        prefix: &str,
        ffn: usize,
        device: &Device,
    ) -> Result<Self, String> {
        let w1 = Linear::load(source, &format!("{prefix}.w1"), device, false)?;
        if w1.weight.dims()[0] != ffn {
            return Err(format!(
                "{prefix}.w1 is {:?}, expected out {ffn}",
                w1.weight.dims()
            ));
        }
        Ok(Self {
            w1,
            w2: Linear::load(source, &format!("{prefix}.w2"), device, false)?,
            w3: Linear::load(source, &format!("{prefix}.w3"), device, false)?,
        })
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.w2
            .forward(silu(self.w1.forward(x.clone())) * self.w3.forward(x))
    }
}

fn apply_rope(x: Tensor<3>, cos: &Tensor<2>, sin: &Tensor<2>) -> Tensor<3> {
    let cos = cos.clone().unsqueeze_dim::<3>(0);
    let sin = sin.clone().unsqueeze_dim::<3>(0);
    x.clone() * cos + rotate_half(x) * sin
}

fn keep_right(seq: usize, prefix: usize) -> Vec<f32> {
    let mut keep = vec![1.0; seq];
    if prefix > 0 && prefix - 1 < seq {
        keep[prefix - 1] = 0.0;
    }
    keep
}

fn rope_tables(
    seq: usize,
    head_dim: usize,
    theta: f32,
    device: &Device,
) -> (Tensor<2>, Tensor<2>) {
    let mut cos = vec![0.0f32; seq * head_dim];
    let mut sin = vec![0.0f32; seq * head_dim];
    let half = head_dim / 2;
    for pos in 0..seq {
        for index in 0..half {
            let exponent = (2 * index) as f32 / head_dim as f32;
            let freq = pos as f32 / theta.powf(exponent);
            let (c, s) = (freq.cos(), freq.sin());
            cos[pos * head_dim + index] = c;
            cos[pos * head_dim + half + index] = c;
            sin[pos * head_dim + index] = s;
            sin[pos * head_dim + half + index] = s;
        }
    }
    (
        Tensor::from_data(TensorData::new(cos, [seq, head_dim]), device),
        Tensor::from_data(TensorData::new(sin, [seq, head_dim]), device),
    )
}
