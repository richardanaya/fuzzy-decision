//! FastConformer audio tower: 8x conv subsampling, relative-position layers,
//! then the adapter and residual that map features onto the trunk width.

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use crate::nn::{
    conv1d_bias, conv2d_bias, gelu_erf, load_tensor, relu_act, sigmoid, silu, softmax_dim, tensor2,
    LayerNorm, Linear,
};
use crate::weights::{AudioSpec, TensorSource};

pub struct AudioTower<B: Backend> {
    stem: [Conv2<B>; 5],
    sub_out: Linear<B>,
    layers: Vec<ConformerLayer<B>>,
    adapter_norm: LayerNorm<B>,
    adapter_1: Linear<B>,
    adapter_2: Linear<B>,
    residual_ln: LayerNorm<B>,
    residual_down: Linear<B>,
    residual_up: Linear<B>,
    d_model: usize,
    heads: usize,
    feat_in: usize,
    device: B::Device,
}

struct Conv2<B: Backend> {
    weight: Tensor<B, 4>,
    bias: Tensor<B, 1>,
    stride: usize,
    groups: usize,
}

impl<B: Backend> Conv2<B> {
    fn load(
        source: &impl TensorSource,
        index: usize,
        stride: usize,
        groups: usize,
        device: &B::Device,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: load_tensor(
                source,
                &format!("audio.encoder.pre_encode.conv.{index}.weight"),
                device,
            )?,
            bias: load_tensor(
                source,
                &format!("audio.encoder.pre_encode.conv.{index}.bias"),
                device,
            )?,
            stride,
            groups,
        })
    }

    fn apply(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let stride = [self.stride, self.stride];
        let padding = if self.stride == 1 && self.weight.dims()[2] == 1 {
            [0, 0]
        } else {
            [1, 1]
        };
        conv2d_bias(
            x,
            self.weight.clone(),
            self.bias.clone(),
            stride,
            padding,
            self.groups,
        )
    }
}

impl<B: Backend> AudioTower<B> {
    pub fn load(
        source: &impl TensorSource,
        spec: &AudioSpec,
        device: &B::Device,
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(spec.layers);
        for index in 0..spec.layers {
            layers.push(ConformerLayer::load(
                source,
                &format!("audio.encoder.layers.{index}"),
                spec,
                device,
            )?);
        }
        Ok(Self {
            stem: [
                Conv2::load(source, 0, 2, 1, device)?,
                Conv2::load(source, 2, 2, spec.channels, device)?,
                Conv2::load(source, 3, 1, 1, device)?,
                Conv2::load(source, 5, 2, spec.channels, device)?,
                Conv2::load(source, 6, 1, 1, device)?,
            ],
            sub_out: Linear::load(source, "audio.encoder.pre_encode.out", device, true)?,
            layers,
            adapter_norm: LayerNorm::load(source, "audio.adapter.norm", 1e-5, device)?,
            adapter_1: Linear::load(source, "audio.adapter.linear_1", device, true)?,
            adapter_2: Linear::load(source, "audio.adapter.linear_2", device, true)?,
            residual_ln: LayerNorm::load(source, "audio.residual.ln", 1e-5, device)?,
            residual_down: {
                let down = Linear::load(source, "audio.residual.down", device, true)?;
                if down.weight.dims()[0] != spec.residual_width {
                    return Err(format!(
                        "audio residual width is {}, expected {}",
                        down.weight.dims()[0],
                        spec.residual_width
                    ));
                }
                down
            },
            residual_up: Linear::load(source, "audio.residual.up", device, true)?,
            d_model: spec.d_model,
            heads: spec.heads,
            feat_in: spec.feat_in,
            device: device.clone(),
        })
    }

    /// `mel` is `(n_mels, frames)` feature-major. `valid` is the unmasked frame count.
    pub fn forward_mel(
        &self,
        mel: &[f32],
        n_mels: usize,
        frames: usize,
        valid: usize,
    ) -> Tensor<B, 2> {
        assert_eq!(n_mels, self.feat_in, "mel bins");
        let mut packed = vec![0.0f32; frames * n_mels];
        for frame in 0..frames {
            for bin in 0..n_mels {
                packed[frame * n_mels + bin] = mel[bin * frames + frame];
            }
        }
        let input = Tensor::<B, 4>::from_data(
            TensorData::new(packed, [1, 1, frames, n_mels]),
            &self.device,
        );
        let (encoded, length) = self.pre_encode(input, valid);
        let hidden = self.conformer(encoded, length);
        let hidden = hidden.narrow(0, 0, length);
        self.residual(self.adapter(hidden))
    }

    fn pre_encode(&self, mut x: Tensor<B, 4>, mut length: usize) -> (Tensor<B, 2>, usize) {
        // Sequential: stride conv, ReLU, depthwise, pointwise, ReLU, depthwise, pointwise, ReLU.
        // The time mask is applied before every one of those modules.
        x = self.stem[0].apply(time_mask(x, length, &self.device));
        length = stride_length(length);
        x = relu_act(time_mask(x, length, &self.device));

        x = self.stem[1].apply(time_mask(x, length, &self.device));
        length = stride_length(length);
        x = self.stem[2].apply(time_mask(x, length, &self.device));
        x = relu_act(time_mask(x, length, &self.device));

        x = self.stem[3].apply(time_mask(x, length, &self.device));
        length = stride_length(length);
        x = self.stem[4].apply(time_mask(x, length, &self.device));
        x = relu_act(time_mask(x, length, &self.device));

        x = time_mask(x, length, &self.device);
        let time = x.dims()[2];
        let freq = x.dims()[3];
        let channels = x.dims()[1];
        let flat = x
            .squeeze_dim::<3>(0)
            .swap_dims(0, 1)
            .reshape([time, channels * freq]);
        (self.sub_out.forward(flat), length)
    }

    fn conformer(&self, mut x: Tensor<B, 2>, length: usize) -> Tensor<B, 2> {
        let time = x.dims()[0];
        let pos = tensor2(
            relative_positions(time, self.d_model),
            time * 2 - 1,
            self.d_model,
            &self.device,
        );
        for layer in &self.layers {
            x = layer.forward(x, &pos, length, self.heads, &self.device);
        }
        x
    }

    fn adapter(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.adapter_2.forward(gelu_erf(
            self.adapter_1.forward(self.adapter_norm.forward(x)),
        ))
    }

    fn residual(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        x.clone()
            + self.residual_up.forward(gelu_erf(
                self.residual_down.forward(self.residual_ln.forward(x)),
            ))
    }
}

fn stride_length(length: usize) -> usize {
    (length + 2 - 3) / 2 + 1
}

fn time_mask<B: Backend>(x: Tensor<B, 4>, length: usize, device: &B::Device) -> Tensor<B, 4> {
    let time = x.dims()[2];
    if length >= time {
        return x;
    }
    let mut values = vec![0.0f32; time];
    for value in values.iter_mut().take(length.min(time)) {
        *value = 1.0;
    }
    let mask =
        Tensor::<B, 1>::from_data(TensorData::new(values, [time]), device).reshape([1, 1, time, 1]);
    x * mask
}

struct ConformerLayer<B: Backend> {
    ff1_norm: LayerNorm<B>,
    ff1_up: Linear<B>,
    ff1_down: Linear<B>,
    attn_norm: LayerNorm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    pos: Linear<B>,
    bias_u: Tensor<B, 2>,
    bias_v: Tensor<B, 2>,
    conv_norm: LayerNorm<B>,
    pointwise1: Tensor<B, 3>,
    pointwise1_bias: Tensor<B, 1>,
    depthwise: Tensor<B, 3>,
    depthwise_bias: Tensor<B, 1>,
    bn_weight: Tensor<B, 1>,
    bn_bias: Tensor<B, 1>,
    bn_mean: Tensor<B, 1>,
    bn_var: Tensor<B, 1>,
    pointwise2: Tensor<B, 3>,
    pointwise2_bias: Tensor<B, 1>,
    kernel: usize,
    ff2_norm: LayerNorm<B>,
    ff2_up: Linear<B>,
    ff2_down: Linear<B>,
    norm_out: LayerNorm<B>,
}

impl<B: Backend> ConformerLayer<B> {
    fn load(
        source: &impl TensorSource,
        prefix: &str,
        spec: &AudioSpec,
        device: &B::Device,
    ) -> Result<Self, String> {
        let ff = |name: &str| -> Result<(Linear<B>, Linear<B>), String> {
            Ok((
                Linear::load(source, &format!("{prefix}.{name}.linear1"), device, true)?,
                Linear::load(source, &format!("{prefix}.{name}.linear2"), device, true)?,
            ))
        };
        let (ff1_up, ff1_down) = ff("feed_forward1")?;
        let expected_ff = spec.d_model * spec.ff_expansion;
        if ff1_up.weight.dims()[0] != expected_ff {
            return Err(format!(
                "{prefix} feed-forward width is {}, expected {expected_ff}",
                ff1_up.weight.dims()[0]
            ));
        }
        let (ff2_up, ff2_down) = ff("feed_forward2")?;
        Ok(Self {
            ff1_norm: LayerNorm::load(
                source,
                &format!("{prefix}.norm_feed_forward1"),
                1e-5,
                device,
            )?,
            ff1_up,
            ff1_down,
            attn_norm: LayerNorm::load(source, &format!("{prefix}.norm_self_att"), 1e-5, device)?,
            q: Linear::load(
                source,
                &format!("{prefix}.self_attn.linear_q"),
                device,
                true,
            )?,
            k: Linear::load(
                source,
                &format!("{prefix}.self_attn.linear_k"),
                device,
                true,
            )?,
            v: Linear::load(
                source,
                &format!("{prefix}.self_attn.linear_v"),
                device,
                true,
            )?,
            out: Linear::load(
                source,
                &format!("{prefix}.self_attn.linear_out"),
                device,
                true,
            )?,
            pos: Linear::load(
                source,
                &format!("{prefix}.self_attn.linear_pos"),
                device,
                false,
            )?,
            bias_u: load_tensor(source, &format!("{prefix}.self_attn.pos_bias_u"), device)?,
            bias_v: load_tensor(source, &format!("{prefix}.self_attn.pos_bias_v"), device)?,
            conv_norm: LayerNorm::load(source, &format!("{prefix}.norm_conv"), 1e-5, device)?,
            pointwise1: load_tensor(
                source,
                &format!("{prefix}.conv.pointwise_conv1.weight"),
                device,
            )?,
            pointwise1_bias: load_tensor(
                source,
                &format!("{prefix}.conv.pointwise_conv1.bias"),
                device,
            )?,
            depthwise: load_tensor(
                source,
                &format!("{prefix}.conv.depthwise_conv.weight"),
                device,
            )?,
            depthwise_bias: load_tensor(
                source,
                &format!("{prefix}.conv.depthwise_conv.bias"),
                device,
            )?,
            bn_weight: load_tensor(source, &format!("{prefix}.conv.batch_norm.weight"), device)?,
            bn_bias: load_tensor(source, &format!("{prefix}.conv.batch_norm.bias"), device)?,
            bn_mean: load_tensor(
                source,
                &format!("{prefix}.conv.batch_norm.running_mean"),
                device,
            )?,
            bn_var: load_tensor(
                source,
                &format!("{prefix}.conv.batch_norm.running_var"),
                device,
            )?,
            pointwise2: load_tensor(
                source,
                &format!("{prefix}.conv.pointwise_conv2.weight"),
                device,
            )?,
            pointwise2_bias: load_tensor(
                source,
                &format!("{prefix}.conv.pointwise_conv2.bias"),
                device,
            )?,
            kernel: spec.kernel,
            ff2_norm: LayerNorm::load(
                source,
                &format!("{prefix}.norm_feed_forward2"),
                1e-5,
                device,
            )?,
            ff2_up,
            ff2_down,
            norm_out: LayerNorm::load(source, &format!("{prefix}.norm_out"), 1e-5, device)?,
        })
    }

    fn forward(
        &self,
        x: Tensor<B, 2>,
        pos: &Tensor<B, 2>,
        length: usize,
        heads: usize,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let x = x.clone()
            + self.feed_forward(
                self.ff1_norm.forward(x.clone()),
                &self.ff1_up,
                &self.ff1_down,
            ) * 0.5;
        let x = x.clone()
            + self.attention(
                self.attn_norm.forward(x.clone()),
                pos,
                length,
                heads,
                device,
            );
        let x = x.clone() + self.conv(self.conv_norm.forward(x.clone()), length, device);
        let x = x.clone()
            + self.feed_forward(
                self.ff2_norm.forward(x.clone()),
                &self.ff2_up,
                &self.ff2_down,
            ) * 0.5;
        self.norm_out.forward(x)
    }

    fn feed_forward(&self, x: Tensor<B, 2>, up: &Linear<B>, down: &Linear<B>) -> Tensor<B, 2> {
        down.forward(silu(up.forward(x)))
    }

    fn attention(
        &self,
        x: Tensor<B, 2>,
        pos: &Tensor<B, 2>,
        length: usize,
        heads: usize,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let time = x.dims()[0];
        let width = x.dims()[1];
        let dk = width / heads;
        let q = self
            .q
            .forward(x.clone())
            .reshape([time, heads, dk])
            .swap_dims(0, 1);
        let k = self
            .k
            .forward(x.clone())
            .reshape([time, heads, dk])
            .swap_dims(0, 1);
        let v = self.v.forward(x).reshape([time, heads, dk]).swap_dims(0, 1);
        let pos_len = pos.dims()[0];
        let p = self
            .pos
            .forward(pos.clone())
            .reshape([pos_len, heads, dk])
            .swap_dims(0, 1);
        let u = self.bias_u.clone().unsqueeze_dim::<3>(1);
        let v_bias = self.bias_v.clone().unsqueeze_dim::<3>(1);
        let ac = (q.clone() + u).matmul(k.swap_dims(1, 2));
        let bd = (q + v_bias).matmul(p.swap_dims(1, 2));
        let bd = rel_shift(bd, device).narrow(2, 0, time);
        let mut scores = (ac + bd) / (dk as f32).sqrt();
        if length < time {
            let mask = invalid_mask(time, length, device);
            scores = scores.clone() * (mask.clone() * -1.0 + 1.0) + mask.clone() * -10000.0;
            let attn = softmax_dim(scores, 2) * (mask * -1.0 + 1.0);
            return self.project(attn.matmul(v), time, width);
        }
        self.project(softmax_dim(scores, 2).matmul(v), time, width)
    }

    fn project(&self, y: Tensor<B, 3>, time: usize, width: usize) -> Tensor<B, 2> {
        self.out.forward(y.swap_dims(0, 1).reshape([time, width]))
    }

    fn conv(&self, x: Tensor<B, 2>, length: usize, device: &B::Device) -> Tensor<B, 2> {
        let time = x.dims()[0];
        let width = x.dims()[1];
        let y = conv1d_bias(
            x.swap_dims(0, 1).unsqueeze_dim::<3>(0),
            self.pointwise1.clone(),
            self.pointwise1_bias.clone(),
            1,
            0,
            1,
        )
        .squeeze_dim::<2>(0);
        let a = y.clone().narrow(0, 0, width);
        let b = y.narrow(0, width, width);
        let mut y = a * sigmoid(b);
        if length < time {
            let zeros = Tensor::<B, 2>::zeros([width, time - length], device);
            y = Tensor::cat(vec![y.narrow(1, 0, length), zeros], 1);
        }
        let pad = (self.kernel - 1) / 2;
        if pad > 0 {
            let zeros = Tensor::<B, 2>::zeros([width, pad], device);
            y = Tensor::cat(vec![zeros.clone(), y, zeros], 1);
        }
        let y = conv1d_bias(
            y.unsqueeze_dim::<3>(0),
            self.depthwise.clone(),
            self.depthwise_bias.clone(),
            1,
            0,
            width,
        )
        .squeeze_dim::<2>(0);
        let y = batch_norm(
            y,
            &self.bn_mean,
            &self.bn_var,
            &self.bn_weight,
            &self.bn_bias,
        );
        let y = conv1d_bias(
            silu(y).unsqueeze_dim::<3>(0),
            self.pointwise2.clone(),
            self.pointwise2_bias.clone(),
            1,
            0,
            1,
        )
        .squeeze_dim::<2>(0);
        y.swap_dims(0, 1)
    }
}

fn batch_norm<B: Backend>(
    x: Tensor<B, 2>,
    mean: &Tensor<B, 1>,
    var: &Tensor<B, 1>,
    weight: &Tensor<B, 1>,
    bias: &Tensor<B, 1>,
) -> Tensor<B, 2> {
    let mean = mean.clone().unsqueeze_dim::<2>(1);
    let var = var.clone().unsqueeze_dim::<2>(1);
    let weight = weight.clone().unsqueeze_dim::<2>(1);
    let bias = bias.clone().unsqueeze_dim::<2>(1);
    (x - mean) / (var + 1e-5).sqrt() * weight + bias
}

fn rel_shift<B: Backend>(bd: Tensor<B, 3>, device: &B::Device) -> Tensor<B, 3> {
    let heads = bd.dims()[0];
    let qlen = bd.dims()[1];
    let pos = bd.dims()[2];
    let zeros = Tensor::<B, 3>::zeros([heads, qlen, 1], device);
    let padded = Tensor::cat(vec![zeros, bd], 2);
    padded
        .reshape([heads, pos + 1, qlen])
        .narrow(1, 1, pos)
        .reshape([heads, qlen, pos])
}

fn invalid_mask<B: Backend>(time: usize, length: usize, device: &B::Device) -> Tensor<B, 3> {
    let mut values = vec![0.0f32; time * time];
    for query in 0..time {
        for key in 0..time {
            if query >= length || key >= length {
                values[query * time + key] = 1.0;
            }
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(values, [time, time]), device).unsqueeze_dim::<3>(0)
}

fn relative_positions(time: usize, d_model: usize) -> Vec<f32> {
    let rows = time * 2 - 1;
    let mut values = vec![0.0f32; rows * d_model];
    let slope = -(10_000.0f32.ln()) / d_model as f32;
    let mut div = Vec::with_capacity(d_model / 2);
    for index in (0..d_model).step_by(2) {
        div.push((index as f32 * slope).exp());
    }
    for (row, position) in ((1 - time as i32)..=(time as i32 - 1)).rev().enumerate() {
        for (bin, scale) in div.iter().enumerate() {
            let angle = position as f32 * scale;
            values[row * d_model + bin * 2] = angle.sin();
            values[row * d_model + bin * 2 + 1] = angle.cos();
        }
    }
    values
}
