//! Shared tensor helpers. Shapes follow PyTorch: a linear weight is `(out, in)`.

pub use burn::tensor::activation::{gelu, relu, sigmoid, softmax};
use burn::tensor::module::{conv1d, conv2d};
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Device, Tensor, TensorData};

use crate::weights::TensorSource;

pub fn load_tensor<const D: usize>(
    source: &impl TensorSource,
    name: &str,
    device: &Device,
) -> Result<Tensor<D>, String> {
    let (values, shape) = source.tensor(name)?;
    if shape.len() != D {
        return Err(format!("{name} has rank {}, expected {D}", shape.len()));
    }
    let mut dims = [0usize; D];
    dims.copy_from_slice(&shape);
    Ok(Tensor::from_data(TensorData::new(values, dims), device))
}

pub struct Linear {
    pub weight: Tensor<2>,
    pub bias: Option<Tensor<1>>,
}

impl Linear {
    pub fn load(
        source: &impl TensorSource,
        name: &str,
        device: &Device,
        bias: bool,
    ) -> Result<Self, String> {
        let weight = load_tensor(source, &format!("{name}.weight"), device)?;
        let bias = if bias {
            Some(load_tensor(source, &format!("{name}.bias"), device)?)
        } else {
            None
        };
        Ok(Self { weight, bias })
    }

    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let y = x.matmul(self.weight.clone().transpose());
        match &self.bias {
            Some(bias) => y + bias.clone().unsqueeze_dim::<2>(0),
            None => y,
        }
    }
}

pub struct RmsNorm {
    pub weight: Tensor<1>,
    pub eps: f32,
}

impl RmsNorm {
    pub fn load(
        source: &impl TensorSource,
        name: &str,
        eps: f32,
        device: &Device,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: load_tensor(source, &format!("{name}.weight"), device)?,
            eps,
        })
    }

    pub fn forward_2d(&self, x: Tensor<2>) -> Tensor<2> {
        let variance = x.clone().powf_scalar(2.0).mean_dim(1);
        x / (variance + self.eps).sqrt() * self.weight.clone().unsqueeze_dim::<2>(0)
    }

    /// Last-axis RMSNorm for `(seq, heads, dim)`.
    pub fn forward_3d(&self, x: Tensor<3>) -> Tensor<3> {
        let variance = x.clone().powf_scalar(2.0).mean_dim(2);
        x / (variance + self.eps).sqrt()
            * self
                .weight
                .clone()
                .unsqueeze_dim::<2>(0)
                .unsqueeze_dim::<3>(0)
    }
}

pub struct LayerNorm {
    pub weight: Tensor<1>,
    pub bias: Tensor<1>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn load(
        source: &impl TensorSource,
        name: &str,
        eps: f32,
        device: &Device,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: load_tensor(source, &format!("{name}.weight"), device)?,
            bias: load_tensor(source, &format!("{name}.bias"), device)?,
            eps,
        })
    }

    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let mean = x.clone().mean_dim(1);
        let centered = x - mean.clone();
        let variance = centered.clone().powf_scalar(2.0).mean_dim(1);
        centered / (variance + self.eps).sqrt() * self.weight.clone().unsqueeze_dim::<2>(0)
            + self.bias.clone().unsqueeze_dim::<2>(0)
    }
}

pub fn tensor2(values: Vec<f32>, rows: usize, cols: usize, device: &Device) -> Tensor<2> {
    Tensor::from_data(TensorData::new(values, [rows, cols]), device)
}

pub fn to_vec<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor
        .into_data()
        .as_slice::<f32>()
        .expect("f32 tensor")
        .to_vec()
}

pub fn silu<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    let gate = sigmoid(x.clone());
    x * gate
}

pub fn gelu_erf<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    gelu(x)
}

pub fn relu_act<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    relu(x)
}

pub fn softmax_dim<const D: usize>(x: Tensor<D>, dim: usize) -> Tensor<D> {
    softmax(x, dim)
}

/// `gelu_pytorch_tanh` from SigLIP2: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))`.
pub fn gelu_tanh<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    let cube = x.clone().powf_scalar(3.0);
    let inner = (x.clone() + cube * 0.044715) * (2.0 / std::f32::consts::PI).sqrt();
    x * (inner.tanh() + 1.0) * 0.5
}

pub fn rotate_half(x: Tensor<3>) -> Tensor<3> {
    let dim = x.dims()[2];
    let half = dim / 2;
    let x1 = x.clone().narrow(2, 0, half);
    let x2 = x.narrow(2, half, half);
    Tensor::cat(vec![x2.neg(), x1], 2)
}

pub fn attend(q: Tensor<3>, k: Tensor<3>, v: Tensor<3>, scale: f32) -> Tensor<3> {
    let scores = q.matmul(k.swap_dims(1, 2)) * scale;
    softmax_dim(scores, 2).matmul(v)
}

/// Media queries read only the prefix. Text queries read the whole sequence.
/// Equivalent to the reference additive mask when every position is valid.
pub fn attend_prefix(
    q: Tensor<3>,
    k: Tensor<3>,
    v: Tensor<3>,
    prefix: usize,
    scale: f32,
) -> Tensor<3> {
    let seq = q.dims()[1];
    if prefix == 0 || prefix >= seq {
        return attend(q, k, v, scale);
    }
    let media = attend(
        q.clone().narrow(1, 0, prefix),
        k.clone().narrow(1, 0, prefix),
        v.clone().narrow(1, 0, prefix),
        scale,
    );
    let text = attend(q.narrow(1, prefix, seq - prefix), k, v, scale);
    Tensor::cat(vec![media, text], 1)
}

pub fn repeat_kv(x: Tensor<3>, groups: usize) -> Tensor<3> {
    if groups == 1 {
        return x;
    }
    let heads = x.dims()[0];
    let mut parts = Vec::with_capacity(heads * groups);
    for index in 0..heads {
        let row = x.clone().narrow(0, index, 1);
        for _ in 0..groups {
            parts.push(row.clone());
        }
    }
    Tensor::cat(parts, 0)
}

pub fn conv1d_bias(
    x: Tensor<3>,
    weight: Tensor<3>,
    bias: Tensor<1>,
    stride: usize,
    padding: usize,
    groups: usize,
) -> Tensor<3> {
    conv1d(
        x,
        weight,
        Some(bias),
        ConvOptions::new([stride], [padding], [1], groups),
    )
}

pub fn conv2d_bias(
    x: Tensor<4>,
    weight: Tensor<4>,
    bias: Tensor<1>,
    stride: [usize; 2],
    padding: [usize; 2],
    groups: usize,
) -> Tensor<4> {
    conv2d(
        x,
        weight,
        Some(bias),
        ConvOptions::new(stride, padding, [1, 1], groups),
    )
}
