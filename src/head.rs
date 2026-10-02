//! The Clef joint schema head.
//!
//! A port of `JointSchemaHead` from `joint_schema_model.py` in
//! `Cloudflare/clef-flash`. The head reads the backbone's final hidden states,
//! routes evidence from the state to each question through cross-attention,
//! and scores every option of every question jointly. It is generic over the
//! Burn backend so the parity test can run it on the CPU.

use burn::tensor::activation::{gelu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_store::SafetensorsStore;

use crate::weights::{dims, json_usize_fields, tensor_data, to_f32};

const NORM_EPS: f64 = 1e-5;

#[derive(Debug, Clone, Copy)]
pub struct HeadConfig {
    pub hidden_size: usize,
    pub width: usize,
    pub routing_layers: usize,
    pub layers: usize,
    pub heads: usize,
    /// Checked by parsing; the feedforward shapes come from the tensors.
    #[allow(dead_code)]
    pub feedforward: usize,
}

impl HeadConfig {
    pub fn parse(text: &str) -> Result<Self, String> {
        let fields = json_usize_fields(text);
        let get = |key: &str| {
            fields
                .get(key)
                .copied()
                .ok_or_else(|| format!("joint_head_config.json is missing {key}"))
        };
        Ok(Self {
            hidden_size: get("hidden_size")?,
            width: get("width")?,
            routing_layers: get("routing_layers")?,
            layers: get("layers")?,
            heads: get("heads")?,
            feedforward: get("feedforward")?,
        })
    }
}

/// One question's spans into the token sequence, plus its type id for the
/// head's type embedding (0 = noul, 1 = choice, 2 = score).
pub struct HeadQuestion {
    pub question_type: usize,
    pub question_span: (usize, usize),
    pub option_spans: Vec<(usize, usize)>,
}

struct LayerNormP<B: Backend> {
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
}

impl<B: Backend> LayerNormP<B> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mean = x.clone().mean_dim(1);
        let centered = x - mean;
        let variance = centered.clone().powf_scalar(2.0).mean_dim(1);
        centered / (variance + NORM_EPS).sqrt() * self.weight.clone().unsqueeze_dim::<2>(0)
            + self.bias.clone().unsqueeze_dim::<2>(0)
    }
}

struct LinearP<B: Backend> {
    weight: Tensor<B, 2>,
    bias: Option<Tensor<B, 1>>,
}

impl<B: Backend> LinearP<B> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let y = x.matmul(self.weight.clone().transpose());
        match &self.bias {
            Some(bias) => y + bias.clone().unsqueeze_dim::<2>(0),
            None => y,
        }
    }
}

/// `torch.nn.MultiheadAttention` with `batch_first=True`, batch size one.
struct Mha<B: Backend> {
    in_weight: Tensor<B, 2>,
    in_bias: Tensor<B, 1>,
    out: LinearP<B>,
    heads: usize,
    width: usize,
}

impl<B: Backend> Mha<B> {
    fn forward(&self, query: Tensor<B, 2>, memory: Tensor<B, 2>) -> Tensor<B, 2> {
        let width = self.width;
        let head_dim = width / self.heads;
        let n = query.dims()[0];
        let project = |x: Tensor<B, 2>, at: usize| {
            let weight = self.in_weight.clone().narrow(0, at, width);
            let bias = self.in_bias.clone().narrow(0, at, width);
            let rows = x.dims()[0];
            (x.matmul(weight.transpose()) + bias.unsqueeze_dim::<2>(0))
                .reshape([rows, self.heads, head_dim])
                .swap_dims(0, 1)
        };
        let q = project(query, 0);
        let k = project(memory.clone(), width);
        let v = project(memory, 2 * width);
        let scores = q.matmul(k.swap_dims(1, 2)) / (head_dim as f32).sqrt();
        let mixed = softmax(scores, 2)
            .matmul(v)
            .swap_dims(0, 1)
            .reshape([n, width]);
        self.out.forward(mixed)
    }
}

/// Pre-norm cross-attention plus a GELU feedforward; the memory is normalized
/// by this layer's own `memory_norm`.
struct RoutingLayer<B: Backend> {
    query_norm: LayerNormP<B>,
    memory_norm: LayerNormP<B>,
    attention: Mha<B>,
    feedforward_norm: LayerNormP<B>,
    ff1: LinearP<B>,
    ff2: LinearP<B>,
}

impl<B: Backend> RoutingLayer<B> {
    fn forward(&self, queries: Tensor<B, 2>, memory: &Tensor<B, 2>) -> Tensor<B, 2> {
        let normalized_memory = self.memory_norm.forward(memory.clone());
        let routed = self
            .attention
            .forward(self.query_norm.forward(queries.clone()), normalized_memory);
        let queries = queries + routed;
        let mid = gelu(
            self.ff1
                .forward(self.feedforward_norm.forward(queries.clone())),
        );
        queries.clone() + self.ff2.forward(mid)
    }
}

/// `torch.nn.TransformerDecoderLayer` with `norm_first=True` and GELU, in
/// eval mode. The memory is attended to as-is.
struct DecoderLayer<B: Backend> {
    self_attn: Mha<B>,
    cross_attn: Mha<B>,
    norm1: LayerNormP<B>,
    norm2: LayerNormP<B>,
    norm3: LayerNormP<B>,
    linear1: LinearP<B>,
    linear2: LinearP<B>,
}

impl<B: Backend> DecoderLayer<B> {
    fn forward(&self, x: Tensor<B, 2>, memory: &Tensor<B, 2>) -> Tensor<B, 2> {
        let normed = self.norm1.forward(x.clone());
        let x = x + self.self_attn.forward(normed.clone(), normed);
        let x = x.clone()
            + self
                .cross_attn
                .forward(self.norm2.forward(x), memory.clone());
        let mid = gelu(self.linear1.forward(self.norm3.forward(x.clone())));
        x + self.linear2.forward(mid)
    }
}

pub struct JointSchemaHead<B: Backend> {
    config: HeadConfig,
    hidden_norm: LayerNormP<B>,
    memory_projection: LinearP<B>,
    question_projection: LinearP<B>,
    option_question_projection: LinearP<B>,
    global_projection: LinearP<B>,
    option_context_projection: LinearP<B>,
    option_lexical_projection: LinearP<B>,
    type_embedding: Tensor<B, 2>,
    evidence_layers: Vec<RoutingLayer<B>>,
    option_summary_norm: LayerNormP<B>,
    layers: Vec<DecoderLayer<B>>,
    field_norm: LayerNormP<B>,
    option_norm: LayerNormP<B>,
    scorer1: LinearP<B>,
    scorer2: LinearP<B>,
    prior_scale: f32,
    joint_scale: f32,
    residual_gate: f32,
}

impl<B: Backend> JointSchemaHead<B> {
    pub fn load(bytes: &[u8], config: HeadConfig, device: &B::Device) -> Result<Self, String> {
        let mut store = SafetensorsStore::from_bytes(Some(bytes.to_vec()));
        let matrix = |store: &mut SafetensorsStore, name: &str| -> Result<Tensor<B, 2>, String> {
            let data = tensor_data(store, name)?;
            let shape = dims(&data);
            if shape.len() != 2 {
                return Err(format!("{name} has shape {shape:?}"));
            }
            Ok(Tensor::from_data(
                TensorData::new(to_f32(data)?, [shape[0], shape[1]]),
                device,
            ))
        };
        let vector = |store: &mut SafetensorsStore, name: &str| -> Result<Tensor<B, 1>, String> {
            let data = tensor_data(store, name)?;
            let shape = dims(&data);
            if shape.len() != 1 {
                return Err(format!("{name} has shape {shape:?}"));
            }
            Ok(Tensor::from_data(
                TensorData::new(to_f32(data)?, [shape[0]]),
                device,
            ))
        };
        let scalar = |store: &mut SafetensorsStore, name: &str| -> Result<f32, String> {
            let data = tensor_data(store, name)?;
            to_f32(data)?
                .first()
                .copied()
                .ok_or_else(|| format!("{name} is empty"))
        };
        let norm = |store: &mut SafetensorsStore, name: &str| -> Result<LayerNormP<B>, String> {
            Ok(LayerNormP {
                weight: vector(store, &format!("{name}.weight"))?,
                bias: vector(store, &format!("{name}.bias"))?,
            })
        };
        let projection = |store: &mut SafetensorsStore, name: &str| -> Result<LinearP<B>, String> {
            Ok(LinearP {
                weight: matrix(store, &format!("{name}.weight"))?,
                bias: None,
            })
        };
        let linear = |store: &mut SafetensorsStore, name: &str| -> Result<LinearP<B>, String> {
            Ok(LinearP {
                weight: matrix(store, &format!("{name}.weight"))?,
                bias: Some(vector(store, &format!("{name}.bias"))?),
            })
        };
        let attention = |store: &mut SafetensorsStore, name: &str| -> Result<Mha<B>, String> {
            Ok(Mha {
                in_weight: matrix(store, &format!("{name}.in_proj_weight"))?,
                in_bias: vector(store, &format!("{name}.in_proj_bias"))?,
                out: linear(store, &format!("{name}.out_proj"))?,
                heads: config.heads,
                width: config.width,
            })
        };

        let mut evidence_layers = Vec::with_capacity(config.routing_layers);
        for index in 0..config.routing_layers {
            let prefix = format!("evidence_layers.{index}");
            evidence_layers.push(RoutingLayer {
                query_norm: norm(&mut store, &format!("{prefix}.query_norm"))?,
                memory_norm: norm(&mut store, &format!("{prefix}.memory_norm"))?,
                attention: attention(&mut store, &format!("{prefix}.attention"))?,
                feedforward_norm: norm(&mut store, &format!("{prefix}.feedforward_norm"))?,
                ff1: linear(&mut store, &format!("{prefix}.feedforward.0"))?,
                ff2: linear(&mut store, &format!("{prefix}.feedforward.3"))?,
            });
        }
        let mut layers = Vec::with_capacity(config.layers);
        for index in 0..config.layers {
            let prefix = format!("layers.{index}");
            layers.push(DecoderLayer {
                self_attn: attention(&mut store, &format!("{prefix}.self_attn"))?,
                cross_attn: attention(&mut store, &format!("{prefix}.multihead_attn"))?,
                norm1: norm(&mut store, &format!("{prefix}.norm1"))?,
                norm2: norm(&mut store, &format!("{prefix}.norm2"))?,
                norm3: norm(&mut store, &format!("{prefix}.norm3"))?,
                linear1: linear(&mut store, &format!("{prefix}.linear1"))?,
                linear2: linear(&mut store, &format!("{prefix}.linear2"))?,
            });
        }

        let max_scale = 100f32.ln();
        Ok(Self {
            config,
            hidden_norm: norm(&mut store, "hidden_norm")?,
            memory_projection: projection(&mut store, "memory_projection")?,
            question_projection: projection(&mut store, "question_projection")?,
            option_question_projection: projection(&mut store, "option_question_projection")?,
            global_projection: projection(&mut store, "global_projection")?,
            option_context_projection: projection(&mut store, "option_context_projection")?,
            option_lexical_projection: projection(&mut store, "option_lexical_projection")?,
            type_embedding: matrix(&mut store, "type_embedding.weight")?,
            evidence_layers,
            option_summary_norm: norm(&mut store, "option_summary_norm")?,
            layers,
            field_norm: norm(&mut store, "field_norm")?,
            option_norm: norm(&mut store, "option_norm")?,
            scorer1: linear(&mut store, "residual_scorer.0")?,
            scorer2: linear(&mut store, "residual_scorer.3")?,
            prior_scale: scalar(&mut store, "prior_logit_scale")?
                .min(max_scale)
                .exp(),
            joint_scale: scalar(&mut store, "joint_logit_scale")?
                .min(max_scale)
                .exp(),
            residual_gate: sigmoid_f32(scalar(&mut store, "residual_gate")?),
        })
    }

    pub fn config(&self) -> HeadConfig {
        self.config
    }

    /// `hidden_norm` over the backbone's final hidden states, per token.
    pub fn normalized(&self, hidden: Tensor<B, 2>) -> Tensor<B, 2> {
        self.hidden_norm.forward(hidden)
    }

    /// `memory_projection` of normalized hidden states, per token. Memory rows
    /// for a cached prefix can be computed once and reused.
    pub fn memory(&self, normalized: Tensor<B, 2>) -> Tensor<B, 2> {
        self.memory_projection.forward(normalized)
    }

    /// One logit per option, grouped by question, in encoded option order.
    ///
    /// `memory` covers the whole sequence. `tail` is the normalized hidden
    /// states from `tail_offset` to the end of the sequence; every span and
    /// the global (last) token must fall inside it. `lexical` has one
    /// `[options, hidden]` tensor per question: the mean output-embedding row
    /// of each option span.
    pub fn score(
        &self,
        memory: Tensor<B, 2>,
        tail: Tensor<B, 2>,
        tail_offset: usize,
        questions: &[HeadQuestion],
        lexical: Vec<Tensor<B, 2>>,
    ) -> Vec<Vec<f32>> {
        if questions.is_empty() {
            return Vec::new();
        }
        let device = memory.device();
        let width = self.config.width;
        let tail_len = tail.dims()[0];
        let span_mean = |span: (usize, usize)| -> Tensor<B, 2> {
            tail.clone()
                .narrow(0, span.0 - tail_offset, span.1 - span.0)
                .mean_dim(0)
        };

        let global_vector = tail.clone().narrow(0, tail_len - 1, 1);
        let question_vectors = Tensor::cat(
            questions
                .iter()
                .map(|question| span_mean(question.question_span))
                .collect(),
            0,
        );

        let mut option_queries = Vec::with_capacity(questions.len());
        let mut option_counts = Vec::with_capacity(questions.len());
        for (index, question) in questions.iter().enumerate() {
            let contexts = Tensor::cat(
                question
                    .option_spans
                    .iter()
                    .map(|span| span_mean(*span))
                    .collect(),
                0,
            );
            let question_vector = question_vectors.clone().narrow(0, index, 1);
            option_queries.push(
                self.option_context_projection.forward(contexts)
                    + self
                        .option_lexical_projection
                        .forward(lexical[index].clone())
                    + self.option_question_projection.forward(question_vector),
            );
            option_counts.push(question.option_spans.len());
        }
        let mut routed = Tensor::cat(option_queries, 0);
        for layer in &self.evidence_layers {
            routed = layer.forward(routed, &memory);
        }

        let base_fields = self.question_projection.forward(question_vectors.clone());
        let mut summaries = Vec::with_capacity(questions.len());
        let mut split_options = Vec::with_capacity(questions.len());
        let mut cursor = 0;
        for (index, count) in option_counts.iter().enumerate() {
            let options = routed.clone().narrow(0, cursor, *count);
            cursor += count;
            let field = base_fields.clone().narrow(0, index, 1);
            let weights = softmax(
                options.clone().matmul(field.transpose()) / (width as f32).sqrt(),
                0,
            );
            summaries.push((options.clone() * weights).sum_dim(0));
            split_options.push(options);
        }
        let type_ids: Vec<i32> = questions
            .iter()
            .map(|question| question.question_type as i32)
            .collect();
        let type_ids =
            Tensor::<B, 1, Int>::from_data(TensorData::new(type_ids, [questions.len()]), &device);
        let mut fields = base_fields
            + self.option_summary_norm.forward(Tensor::cat(summaries, 0))
            + self.global_projection.forward(global_vector.clone())
            + self.type_embedding.clone().select(0, type_ids);
        for layer in &self.layers {
            fields = layer.forward(fields, &memory);
        }
        let fields = self.field_norm.forward(fields);

        let mut results = Vec::with_capacity(questions.len());
        for (index, options) in split_options.into_iter().enumerate() {
            let count = option_counts[index];
            let anchor = normalize_rows(
                question_vectors.clone().narrow(0, index, 1) + global_vector.clone(),
                1e-12,
            );
            let lexical_anchor = normalize_rows(lexical[index].clone(), 1e-12);
            let prior = lexical_anchor.matmul(anchor.transpose()) * self.prior_scale;
            let options = self.option_norm.forward(options);
            let field = fields.clone().narrow(0, index, 1);
            let repeated = field.expand([count, width]);
            let cosine = cosine_rows(repeated.clone(), options.clone());
            let features = Tensor::cat(
                vec![
                    repeated.clone(),
                    options.clone(),
                    repeated.clone() * options.clone(),
                    (repeated - options).abs(),
                ],
                1,
            );
            let residual = self.scorer2.forward(gelu(self.scorer1.forward(features)));
            let joint = cosine * self.joint_scale + residual;
            let logits = prior + joint * self.residual_gate;
            let data = logits.into_data();
            results.push(data.as_slice::<f32>().unwrap().to_vec());
        }
        results
    }
}

/// `torch.nn.functional.normalize(x, dim=-1)`.
fn normalize_rows<B: Backend>(x: Tensor<B, 2>, eps: f64) -> Tensor<B, 2> {
    let norm = x.clone().powf_scalar(2.0).sum_dim(1).sqrt().clamp_min(eps);
    x / norm
}

/// `torch.nn.functional.cosine_similarity(a, b, dim=-1)`, row-wise,
/// as an `[n, 1]` tensor.
fn cosine_rows<B: Backend>(a: Tensor<B, 2>, b: Tensor<B, 2>) -> Tensor<B, 2> {
    let eps = 1e-8;
    let dot = (a.clone() * b.clone()).sum_dim(1);
    let na = a.powf_scalar(2.0).sum_dim(1).sqrt().clamp_min(eps);
    let nb = b.powf_scalar(2.0).sum_dim(1).sqrt().clamp_min(eps);
    dot / (na * nb)
}

fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
