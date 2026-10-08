//! Two-layer pre-norm decision head. It adds a question-type embedding, runs
//! the text positions through a ReLU transformer, and scores each marker.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::nn::{attend, gelu_erf, load_tensor, relu_act, to_vec, LayerNorm, Linear};
use crate::weights::TensorSource;

pub struct DecisionHead<B: Backend> {
    type_emb: Tensor<B, 2>,
    layers: Vec<HeadLayer<B>>,
    scorer_norm: LayerNorm<B>,
    scorer_in: Linear<B>,
    scorer_out: Linear<B>,
    heads: usize,
    head_dim: usize,
}

impl<B: Backend> DecisionHead<B> {
    pub fn load(
        source: &impl TensorSource,
        hidden: usize,
        layers: usize,
        device: &B::Device,
    ) -> Result<Self, String> {
        if !hidden.is_multiple_of(64) {
            return Err(format!(
                "decision head width {hidden} is not divisible by 64"
            ));
        }
        let mut stack = Vec::with_capacity(layers);
        for index in 0..layers {
            stack.push(HeadLayer::load(
                source,
                &format!("head.head.layers.{index}"),
                hidden,
                device,
            )?);
        }
        Ok(Self {
            type_emb: load_tensor(source, "head.type_emb.weight", device)?,
            layers: stack,
            scorer_norm: LayerNorm::load(source, "head.scorer.0", 1e-5, device)?,
            scorer_in: Linear::load(source, "head.scorer.1", device, true)?,
            scorer_out: Linear::load(source, "head.scorer.3", device, true)?,
            heads: hidden / 64,
            head_dim: 64,
        })
    }

    pub fn logits(&self, hidden: Tensor<B, 2>, qtype: usize, markers: &[usize]) -> Vec<f32> {
        let kind = self.type_emb.clone().narrow(0, qtype, 1);
        let mut hidden = hidden + kind;
        for layer in &self.layers {
            hidden = layer.forward(hidden, self.heads, self.head_dim);
        }
        let rows: Vec<_> = markers
            .iter()
            .map(|marker| hidden.clone().narrow(0, *marker, 1))
            .collect();
        let gathered = Tensor::cat(rows, 0);
        let scored = self.scorer_out.forward(gelu_erf(
            self.scorer_in.forward(self.scorer_norm.forward(gathered)),
        ));
        to_vec(scored.squeeze_dim::<1>(1))
    }
}

struct HeadLayer<B: Backend> {
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
}

impl<B: Backend> HeadLayer<B> {
    fn load(
        source: &impl TensorSource,
        prefix: &str,
        hidden: usize,
        device: &B::Device,
    ) -> Result<Self, String> {
        let weight: Tensor<B, 2> = load_tensor(
            source,
            &format!("{prefix}.self_attn.in_proj_weight"),
            device,
        )?;
        let bias: Tensor<B, 1> =
            load_tensor(source, &format!("{prefix}.self_attn.in_proj_bias"), device)?;
        let (q, k, v) = split_qkv(weight, bias, hidden);
        Ok(Self {
            q,
            k,
            v,
            out: Linear::load(
                source,
                &format!("{prefix}.self_attn.out_proj"),
                device,
                true,
            )?,
            fc1: Linear::load(source, &format!("{prefix}.linear1"), device, true)?,
            fc2: Linear::load(source, &format!("{prefix}.linear2"), device, true)?,
            norm1: LayerNorm::load(source, &format!("{prefix}.norm1"), 1e-5, device)?,
            norm2: LayerNorm::load(source, &format!("{prefix}.norm2"), 1e-5, device)?,
        })
    }

    fn forward(&self, hidden: Tensor<B, 2>, heads: usize, head_dim: usize) -> Tensor<B, 2> {
        let attended = self.attention(self.norm1.forward(hidden.clone()), heads, head_dim);
        let hidden = hidden + attended;
        let fed = self.fc2.forward(relu_act(
            self.fc1.forward(self.norm2.forward(hidden.clone())),
        ));
        hidden + fed
    }

    fn attention(&self, x: Tensor<B, 2>, heads: usize, head_dim: usize) -> Tensor<B, 2> {
        let seq = x.dims()[0];
        let q = self
            .q
            .forward(x.clone())
            .reshape([seq, heads, head_dim])
            .swap_dims(0, 1);
        let k = self
            .k
            .forward(x.clone())
            .reshape([seq, heads, head_dim])
            .swap_dims(0, 1);
        let v = self
            .v
            .forward(x)
            .reshape([seq, heads, head_dim])
            .swap_dims(0, 1);
        let y = attend(q, k, v, (head_dim as f32).powf(-0.5));
        self.out
            .forward(y.swap_dims(0, 1).reshape([seq, heads * head_dim]))
    }
}

fn split_qkv<B: Backend>(
    weight: Tensor<B, 2>,
    bias: Tensor<B, 1>,
    hidden: usize,
) -> (Linear<B>, Linear<B>, Linear<B>) {
    let parts = [(0, hidden), (hidden, hidden), (2 * hidden, hidden)];
    let linears = parts.map(|(start, len)| Linear {
        weight: weight.clone().narrow(0, start, len),
        bias: Some(bias.clone().narrow(0, start, len)),
    });
    (
        linears[0].clone_linear(),
        linears[1].clone_linear(),
        linears[2].clone_linear(),
    )
}

trait CloneLinear<B: Backend> {
    fn clone_linear(&self) -> Linear<B>;
}

impl<B: Backend> CloneLinear<B> for Linear<B> {
    fn clone_linear(&self) -> Linear<B> {
        Linear {
            weight: self.weight.clone(),
            bias: self.bias.clone(),
        }
    }
}
