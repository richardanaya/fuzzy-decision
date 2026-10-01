//! The Clef-Flash backbone: Qwen3.5-9B with its vision encoder, on Burn's
//! WGPU device.
//!
//! The language model interleaves gated-delta linear-attention layers with a
//! full-attention layer every fourth block. The recurrence is one GPU kernel
//! per layer. Projections, full attention, and MLPs stay on the WGPU device.
//! Weights are read from the sharded `model-*.safetensors` files of a local
//! `Cloudflare/clef-flash` snapshot.

use std::path::Path;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::activation::{gelu, sigmoid, softmax};
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::FloatDType;
use burn::tensor::{Tensor, TensorData};

use crate::weights::{Bf16Table, Store};

pub const VOCAB: usize = 248_320;
pub const TEXT_HIDDEN: usize = 4096;
const TEXT_LAYERS: usize = 32;
const TEXT_HEADS: usize = 16;
const TEXT_KV: usize = 4;
const TEXT_HEAD_DIM: usize = 256;
const TEXT_GROUP: usize = TEXT_HEADS / TEXT_KV;
const ROTARY: usize = 64;
const K_HEADS: usize = 16;
pub(crate) const V_HEADS: usize = 32;
pub(crate) const LIN_DIM: usize = 128;
const KEY_DIM: usize = K_HEADS * LIN_DIM;
const VALUE_DIM: usize = V_HEADS * LIN_DIM;
const CONV_DIM: usize = KEY_DIM * 2 + VALUE_DIM;
const CONV_K: usize = 4;
const ROPE_THETA: f32 = 10_000_000.0;
const MROPE: [usize; 3] = [11, 11, 10];

const VISION_LAYERS: usize = 27;
const VISION_HIDDEN: usize = 1152;
const VISION_HEADS: usize = 16;
const VISION_HEAD_DIM: usize = VISION_HIDDEN / VISION_HEADS;
const POS_SIDE: usize = 48;
pub const PATCH: usize = 16;
pub const MERGE: usize = 2;

enum Block {
    Linear(LinearBlock),
    Full(FullBlock),
}

struct LinearBlock {
    /// `qkv`, `z`, `a`, and `b` stacked on the output axis so one matmul reads the input.
    in_proj: Tensor<Wgpu, 2>,
    conv: Tensor<Wgpu, 2>,
    a_log: Tensor<Wgpu, 1>,
    dt_bias: Tensor<Wgpu, 1>,
    norm: Tensor<Wgpu, 1>,
    out: Tensor<Wgpu, 2>,
    input_norm: Tensor<Wgpu, 1>,
    post_norm: Tensor<Wgpu, 1>,
    gate: Tensor<Wgpu, 2>,
    up: Tensor<Wgpu, 2>,
    down: Tensor<Wgpu, 2>,
}

struct FullBlock {
    q: Tensor<Wgpu, 2>,
    k: Tensor<Wgpu, 2>,
    v: Tensor<Wgpu, 2>,
    o: Tensor<Wgpu, 2>,
    q_norm: Tensor<Wgpu, 1>,
    k_norm: Tensor<Wgpu, 1>,
    input_norm: Tensor<Wgpu, 1>,
    post_norm: Tensor<Wgpu, 1>,
    gate: Tensor<Wgpu, 2>,
    up: Tensor<Wgpu, 2>,
    down: Tensor<Wgpu, 2>,
}

struct VisionLayer {
    norm1_w: Tensor<Wgpu, 1>,
    norm1_b: Tensor<Wgpu, 1>,
    norm2_w: Tensor<Wgpu, 1>,
    norm2_b: Tensor<Wgpu, 1>,
    qkv_w: Tensor<Wgpu, 2>,
    qkv_b: Tensor<Wgpu, 1>,
    proj_w: Tensor<Wgpu, 2>,
    proj_b: Tensor<Wgpu, 1>,
    fc1_w: Tensor<Wgpu, 2>,
    fc1_b: Tensor<Wgpu, 1>,
    fc2_w: Tensor<Wgpu, 2>,
    fc2_b: Tensor<Wgpu, 1>,
}

struct Merger {
    norm_w: Tensor<Wgpu, 1>,
    norm_b: Tensor<Wgpu, 1>,
    fc1_w: Tensor<Wgpu, 2>,
    fc1_b: Tensor<Wgpu, 1>,
    fc2_w: Tensor<Wgpu, 2>,
    fc2_b: Tensor<Wgpu, 1>,
}

struct Tower {
    patch_weight: Tensor<Wgpu, 2>,
    patch_bias: Tensor<Wgpu, 1>,
    pos_embed: Vec<f32>,
    layers: Vec<VisionLayer>,
    merger: Merger,
}

/// Per-layer state after a prefix, so a suffix can continue the pass.
pub enum Carry {
    Linear {
        mixed_tail: Tensor<Wgpu, 2>,
        state: Tensor<Wgpu, 3>,
    },
    Full { k: Tensor<Wgpu, 3>, v: Tensor<Wgpu, 3> },
}

impl Clone for Carry {
    fn clone(&self) -> Self {
        match self {
            Carry::Linear { mixed_tail, state } => Carry::Linear {
                mixed_tail: mixed_tail.clone(),
                state: state.clone(),
            },
            Carry::Full { k, v } => Carry::Full {
                k: k.clone(),
                v: v.clone(),
            },
        }
    }
}

/// Where an image sits in the token sequence, for rope positions.
#[derive(Debug, Clone, Copy)]
pub struct ImageSpan {
    /// Index of the first image token.
    pub at: usize,
    /// Merged grid height and width (image tokens are `llm_h * llm_w`).
    pub llm_h: usize,
    pub llm_w: usize,
}

pub struct Backbone {
    device: WgpuDevice,
    embed: Bf16Table,
    lm_head: Bf16Table,
    text_norm: Tensor<Wgpu, 1>,
    blocks: Vec<Block>,
    vision: Option<Tower>,
}

impl Backbone {
    pub fn load(dir: &Path, with_vision: bool, device: &WgpuDevice) -> Result<Self, String> {
        let index = std::fs::read_to_string(dir.join("model.safetensors.index.json"))
            .map_err(|err| format!("read model.safetensors.index.json: {err}"))?;
        let mut store = Store::parse(dir, &index)?;
        if !with_vision {
            store.skip_prefix("model.visual.");
        }
        let embed = store.table("model.language_model.embed_tokens.weight")?;
        if embed.rows != VOCAB || embed.dim != TEXT_HIDDEN {
            return Err(format!(
                "token table has shape [{}, {}]",
                embed.rows, embed.dim
            ));
        }
        let lm_head = store.table("lm_head.weight")?;
        if lm_head.rows != VOCAB || lm_head.dim != TEXT_HIDDEN {
            return Err(format!(
                "output table has shape [{}, {}]",
                lm_head.rows, lm_head.dim
            ));
        }
        let text_norm = vector(&mut store, "model.language_model.norm.weight", device)?;
        let mut blocks = Vec::with_capacity(TEXT_LAYERS);
        for index in 0..TEXT_LAYERS {
            let prefix = format!("model.language_model.layers.{index}");
            let input_norm = vector(&mut store, &format!("{prefix}.input_layernorm.weight"), device)?;
            let post_norm = vector(
                &mut store,
                &format!("{prefix}.post_attention_layernorm.weight"),
                device,
            )?;
            let gate = matrix(&mut store, &format!("{prefix}.mlp.gate_proj.weight"), device)?;
            let up = matrix(&mut store, &format!("{prefix}.mlp.up_proj.weight"), device)?;
            let down = matrix(&mut store, &format!("{prefix}.mlp.down_proj.weight"), device)?;
            if index % 4 == 3 {
                blocks.push(Block::Full(FullBlock {
                    q: matrix(&mut store, &format!("{prefix}.self_attn.q_proj.weight"), device)?,
                    k: matrix(&mut store, &format!("{prefix}.self_attn.k_proj.weight"), device)?,
                    v: matrix(&mut store, &format!("{prefix}.self_attn.v_proj.weight"), device)?,
                    o: matrix(&mut store, &format!("{prefix}.self_attn.o_proj.weight"), device)?,
                    q_norm: vector(&mut store, &format!("{prefix}.self_attn.q_norm.weight"), device)?,
                    k_norm: vector(&mut store, &format!("{prefix}.self_attn.k_norm.weight"), device)?,
                    input_norm,
                    post_norm,
                    gate,
                    up,
                    down,
                }));
            } else {
                let (conv_values, conv_shape) = store.read(&format!("{prefix}.linear_attn.conv1d.weight"))?;
                let kernel = conv_shape.last().copied().unwrap_or(CONV_K);
                let mut conv = vec![0f32; CONV_DIM * CONV_K];
                for channel in 0..CONV_DIM {
                    for tap in 0..CONV_K.min(kernel) {
                        conv[channel * CONV_K + tap] = conv_values[channel * kernel + tap];
                    }
                }
                let conv = Tensor::from_data(TensorData::new(conv, [CONV_DIM, CONV_K]), device);
                let (a_log, a_shape) = store.read(&format!("{prefix}.linear_attn.A_log"))?;
                let (dt_bias, dt_shape) = store.read(&format!("{prefix}.linear_attn.dt_bias"))?;
                let (norm, norm_shape) = store.read(&format!("{prefix}.linear_attn.norm.weight"))?;
                let qkv = matrix(&mut store, &format!("{prefix}.linear_attn.in_proj_qkv.weight"), device)?;
                let z = matrix(&mut store, &format!("{prefix}.linear_attn.in_proj_z.weight"), device)?;
                let a = matrix(&mut store, &format!("{prefix}.linear_attn.in_proj_a.weight"), device)?;
                let b = matrix(&mut store, &format!("{prefix}.linear_attn.in_proj_b.weight"), device)?;
                blocks.push(Block::Linear(LinearBlock {
                    in_proj: Tensor::cat(vec![qkv, z, a, b], 0),
                    conv,
                    a_log: Tensor::from_data(TensorData::new(a_log, [a_shape[0]]), device),
                    dt_bias: Tensor::from_data(TensorData::new(dt_bias, [dt_shape[0]]), device),
                    norm: Tensor::from_data(TensorData::new(norm, [norm_shape[0]]), device),
                    out: matrix(&mut store, &format!("{prefix}.linear_attn.out_proj.weight"), device)?,
                    input_norm,
                    post_norm,
                    gate,
                    up,
                    down,
                }));
            }
        }
        let vision = if with_vision {
            let (patch_values, patch_shape) = store.read("model.visual.patch_embed.proj.weight")?;
            let patch_in: usize = patch_shape[1..].iter().product();
            let patch_weight = Tensor::from_data(
                TensorData::new(patch_values, [patch_shape[0], patch_in]),
                device,
            );
            let patch_bias = vector(&mut store, "model.visual.patch_embed.proj.bias", device)?;
            let (pos_embed, pos_shape) = store.read("model.visual.pos_embed.weight")?;
            if pos_shape != [POS_SIDE * POS_SIDE, VISION_HIDDEN] {
                return Err(format!("vision position table has shape {pos_shape:?}"));
            }
            let mut layers = Vec::with_capacity(VISION_LAYERS);
            for index in 0..VISION_LAYERS {
                let prefix = format!("model.visual.blocks.{index}");
                layers.push(VisionLayer {
                    norm1_w: vector(&mut store, &format!("{prefix}.norm1.weight"), device)?,
                    norm1_b: vector(&mut store, &format!("{prefix}.norm1.bias"), device)?,
                    norm2_w: vector(&mut store, &format!("{prefix}.norm2.weight"), device)?,
                    norm2_b: vector(&mut store, &format!("{prefix}.norm2.bias"), device)?,
                    qkv_w: matrix_f32(&mut store, &format!("{prefix}.attn.qkv.weight"), device)?,
                    qkv_b: vector(&mut store, &format!("{prefix}.attn.qkv.bias"), device)?,
                    proj_w: matrix_f32(&mut store, &format!("{prefix}.attn.proj.weight"), device)?,
                    proj_b: vector(&mut store, &format!("{prefix}.attn.proj.bias"), device)?,
                    fc1_w: matrix_f32(&mut store, &format!("{prefix}.mlp.linear_fc1.weight"), device)?,
                    fc1_b: vector(&mut store, &format!("{prefix}.mlp.linear_fc1.bias"), device)?,
                    fc2_w: matrix_f32(&mut store, &format!("{prefix}.mlp.linear_fc2.weight"), device)?,
                    fc2_b: vector(&mut store, &format!("{prefix}.mlp.linear_fc2.bias"), device)?,
                });
            }
            let merger = Merger {
                norm_w: vector(&mut store, "model.visual.merger.norm.weight", device)?,
                norm_b: vector(&mut store, "model.visual.merger.norm.bias", device)?,
                fc1_w: matrix_f32(&mut store, "model.visual.merger.linear_fc1.weight", device)?,
                fc1_b: vector(&mut store, "model.visual.merger.linear_fc1.bias", device)?,
                fc2_w: matrix_f32(&mut store, "model.visual.merger.linear_fc2.weight", device)?,
                fc2_b: vector(&mut store, "model.visual.merger.linear_fc2.bias", device)?,
            };
            Some(Tower {
                patch_weight,
                patch_bias,
                pos_embed,
                layers,
                merger,
            })
        } else {
            None
        };
        Ok(Self {
            device: device.clone(),
            embed,
            lm_head,
            text_norm,
            blocks,
            vision,
        })
    }

    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }

    pub fn embed_ids(&self, ids: &[u32]) -> Tensor<Wgpu, 2> {
        let mut gathered = vec![0f32; ids.len() * TEXT_HIDDEN];
        for (index, id) in ids.iter().enumerate() {
            self.embed.write_row(
                *id as usize,
                &mut gathered[index * TEXT_HIDDEN..(index + 1) * TEXT_HIDDEN],
            );
        }
        Tensor::from_data(TensorData::new(gathered, [ids.len(), TEXT_HIDDEN]), &self.device)
    }

    /// The mean output-embedding row of the given token ids, for the joint
    /// head's lexical prior.
    pub fn lexical_mean(&self, ids: &[u32]) -> Vec<f32> {
        self.lm_head.mean_rows(ids)
    }

    /// Runs the vision tower: packed patches to merged rows in the language
    /// model's hidden size.
    pub fn see(&self, pixels: &[f32], grid_h: usize, grid_w: usize) -> Result<Tensor<Wgpu, 2>, String> {
        let tower = self
            .vision
            .as_ref()
            .ok_or_else(|| "this model was loaded without its vision tower".to_string())?;
        let n = pixels.len() / (3 * 2 * PATCH * PATCH);
        let flat = Tensor::<Wgpu, 2>::from_data(
            TensorData::new(pixels.to_vec(), [n, 3 * 2 * PATCH * PATCH]),
            &self.device,
        );
        let mut hidden = linear2(&flat, &tower.patch_weight, Some(&tower.patch_bias));
        hidden = hidden + self.position_embed(tower, grid_h, grid_w);
        let (cos, sin) = vision_rope(grid_h, grid_w, &self.device);
        for layer in &tower.layers {
            hidden = vision_block(hidden, layer, &cos, &sin);
        }
        Ok(merge(&hidden, &tower.merger))
    }

    fn position_embed(&self, tower: &Tower, grid_h: usize, grid_w: usize) -> Tensor<Wgpu, 2> {
        let side = POS_SIDE;
        let n = grid_h * grid_w;
        let mut gathered = vec![0f32; n * VISION_HIDDEN];
        let table = &tower.pos_embed;
        for (index, (row, col)) in block_coords(grid_h, grid_w).into_iter().enumerate() {
            let src_y = row as f32 * (side - 1) as f32 / (grid_h - 1).max(1) as f32;
            let src_x = col as f32 * (side - 1) as f32 / (grid_w - 1).max(1) as f32;
            let y0 = src_y.floor() as usize;
            let x0 = src_x.floor() as usize;
            let y1 = (y0 + 1).min(side - 1);
            let x1 = (x0 + 1).min(side - 1);
            let wy = src_y - y0 as f32;
            let wx = src_x - x0 as f32;
            let taps = [
                (y0, x0, (1.0 - wy) * (1.0 - wx)),
                (y0, x1, (1.0 - wy) * wx),
                (y1, x0, wy * (1.0 - wx)),
                (y1, x1, wy * wx),
            ];
            for dim in 0..VISION_HIDDEN {
                let mut value = 0.0;
                for (y, x, weight) in taps {
                    value += weight * table[(y * side + x) * VISION_HIDDEN + dim];
                }
                gathered[index * VISION_HIDDEN + dim] = value;
            }
        }
        Tensor::from_data(TensorData::new(gathered, [n, VISION_HIDDEN]), &self.device)
    }

    /// Runs the language blocks over `hidden`, continuing from `past` when
    /// given. Returns the hidden states before the final norm, plus the
    /// per-layer carries after the last token.
    pub fn run(
        &self,
        mut hidden: Tensor<Wgpu, 2>,
        cos: &Tensor<Wgpu, 2>,
        sin: &Tensor<Wgpu, 2>,
        past: Option<&[Carry]>,
    ) -> (Tensor<Wgpu, 2>, Vec<Carry>) {
        let mut carries = Vec::with_capacity(self.blocks.len());
        for (index, block) in self.blocks.iter().enumerate() {
            let (next, carry) = step_block(block, hidden, cos, sin, past.map(|item| &item[index]));
            hidden = next;
            carries.push(carry);
        }
        (hidden, carries)
    }

    pub fn final_norm(&self, hidden: Tensor<Wgpu, 2>) -> Tensor<Wgpu, 2> {
        rms35(hidden, &self.text_norm)
    }
}

fn matrix(store: &mut Store, name: &str, device: &WgpuDevice) -> Result<Tensor<Wgpu, 2>, String> {
    Ok(matrix_f32(store, name, device)?.cast(FloatDType::F16))
}

fn matrix_f32(store: &mut Store, name: &str, device: &WgpuDevice) -> Result<Tensor<Wgpu, 2>, String> {
    let (values, shape) = store.read(name)?;
    if shape.len() != 2 {
        return Err(format!("{name} has shape {shape:?}"));
    }
    Ok(Tensor::from_data(
        TensorData::new(values, [shape[0], shape[1]]),
        device,
    ))
}

fn vector(store: &mut Store, name: &str, device: &WgpuDevice) -> Result<Tensor<Wgpu, 1>, String> {
    let (values, shape) = store.read(name)?;
    if shape.len() != 1 {
        return Err(format!("{name} has shape {shape:?}"));
    }
    Ok(Tensor::from_data(TensorData::new(values, [shape[0]]), device))
}

fn step_block(
    block: &Block,
    hidden: Tensor<Wgpu, 2>,
    cos: &Tensor<Wgpu, 2>,
    sin: &Tensor<Wgpu, 2>,
    past: Option<&Carry>,
) -> (Tensor<Wgpu, 2>, Carry) {
    match (block, past) {
        (Block::Linear(layer), Some(Carry::Linear { mixed_tail, state })) => {
            linear_block(hidden, layer, Some(mixed_tail), state.clone())
        }
        (Block::Linear(layer), None) => {
            let state = Tensor::<Wgpu, 3>::zeros([V_HEADS, LIN_DIM, LIN_DIM], &hidden.device());
            linear_block(hidden, layer, None, state)
        }
        (Block::Full(layer), Some(Carry::Full { k, v })) => full_block(hidden, layer, cos, sin, Some((k, v))),
        (Block::Full(layer), None) => full_block(hidden, layer, cos, sin, None),
        _ => panic!("language layer cache does not match the block"),
    }
}

fn linear_block(
    hidden: Tensor<Wgpu, 2>,
    layer: &LinearBlock,
    history: Option<&Tensor<Wgpu, 2>>,
    state: Tensor<Wgpu, 3>,
) -> (Tensor<Wgpu, 2>, Carry) {
    let seq = hidden.dims()[0];
    let normed = rms35(hidden.clone(), &layer.input_norm);
    let projected = linear2(&normed, &layer.in_proj, None);
    let mixed = projected.clone().narrow(1, 0, CONV_DIM);
    let z = projected.clone().narrow(1, CONV_DIM, VALUE_DIM);
    let a = projected.clone().narrow(1, CONV_DIM + VALUE_DIM, V_HEADS);
    let b = projected.narrow(1, CONV_DIM + VALUE_DIM + V_HEADS, V_HEADS);
    let conv = causal_conv_gpu(mixed.clone(), history, &layer.conv);
    let (q, k, v) = l2_qkv(conv);
    let (g, beta) = decay_beta(a, b, &layer.a_log, &layer.dt_bias);
    let (core, state) = gated_delta_gpu(q, k, v, g, beta, state);
    let gated = gate_norm_gpu(core, z, &layer.norm);
    let hidden = hidden + linear2(&gated, &layer.out, None);
    let normed = rms35(hidden.clone(), &layer.post_norm);
    let mixed_tail = conv_tail(mixed, history, seq);
    (
        hidden + mlp(&normed, &layer.gate, &layer.up, &layer.down),
        Carry::Linear { mixed_tail, state },
    )
}

fn conv_tail(mixed: Tensor<Wgpu, 2>, history: Option<&Tensor<Wgpu, 2>>, seq: usize) -> Tensor<Wgpu, 2> {
    let keep = (CONV_K - 1).min(seq);
    if seq >= CONV_K - 1 {
        return mixed.narrow(0, seq - keep, keep);
    }
    let need = CONV_K - 1 - seq;
    let older = match history {
        Some(history) if history.dims()[0] >= need => {
            let rows = history.dims()[0];
            history.clone().narrow(0, rows - need, need)
        }
        Some(history) => {
            let rows = history.dims()[0];
            let pad = Tensor::<Wgpu, 2>::zeros([need - rows, CONV_DIM], &mixed.device());
            Tensor::cat(vec![pad, history.clone()], 0)
        }
        None => Tensor::<Wgpu, 2>::zeros([need, CONV_DIM], &mixed.device()),
    };
    Tensor::cat(vec![older, mixed], 0)
}

fn causal_conv_gpu(
    mixed: Tensor<Wgpu, 2>,
    history: Option<&Tensor<Wgpu, 2>>,
    weight: &Tensor<Wgpu, 2>,
) -> Tensor<Wgpu, 2> {
    let seq = mixed.dims()[0];
    let hist = history.map(|past| past.dims()[0]).unwrap_or(0);
    let full = match history {
        Some(past) if hist > 0 => Tensor::cat(vec![past.clone(), mixed], 0),
        _ => mixed,
    };
    let mut acc = Tensor::<Wgpu, 2>::zeros([seq, CONV_DIM], &full.device());
    for tap in 0..CONV_K {
        let lag = CONV_K - 1 - tap;
        let start = hist as isize - lag as isize;
        let shifted = if start >= 0 {
            full.clone().narrow(0, start as usize, seq)
        } else {
            let missing = (-start) as usize;
            let zeros = Tensor::<Wgpu, 2>::zeros([missing, CONV_DIM], &full.device());
            let take = seq.saturating_sub(missing);
            if take == 0 {
                zeros
            } else {
                Tensor::cat(vec![zeros, full.clone().narrow(0, 0, take)], 0)
            }
        };
        let tap_weight = weight
            .clone()
            .narrow(1, tap, 1)
            .squeeze_dim::<1>(1)
            .unsqueeze_dim::<2>(0);
        acc = acc + shifted * tap_weight;
    }
    acc.clone() * sigmoid(acc)
}

fn l2_qkv(conv: Tensor<Wgpu, 2>) -> (Tensor<Wgpu, 3>, Tensor<Wgpu, 3>, Tensor<Wgpu, 3>) {
    let seq = conv.dims()[0];
    let q = l2_heads(conv.clone().narrow(1, 0, KEY_DIM), seq, true);
    let k = l2_heads(conv.clone().narrow(1, KEY_DIM, KEY_DIM), seq, false);
    let v = conv.narrow(1, 2 * KEY_DIM, VALUE_DIM).reshape([seq, V_HEADS, LIN_DIM]);
    (q, k, v)
}

fn l2_heads(src: Tensor<Wgpu, 2>, seq: usize, scale_query: bool) -> Tensor<Wgpu, 3> {
    let heads = src.reshape([seq, K_HEADS, LIN_DIM]);
    let mut inv = (heads.clone().powf_scalar(2.0).sum_dim(2) + 1e-6).sqrt().recip();
    if scale_query {
        inv = inv * (LIN_DIM as f32).powf(-0.5);
    }
    heads
        .mul(inv)
        .unsqueeze_dim::<4>(2)
        .repeat_dim(2, V_HEADS / K_HEADS)
        .reshape([seq, V_HEADS, LIN_DIM])
}

fn decay_beta(
    a: Tensor<Wgpu, 2>,
    b: Tensor<Wgpu, 2>,
    a_log: &Tensor<Wgpu, 1>,
    dt_bias: &Tensor<Wgpu, 1>,
) -> (Tensor<Wgpu, 2>, Tensor<Wgpu, 2>) {
    let raw = a + dt_bias.clone().unsqueeze_dim(0);
    let softplus = (raw.clone().exp() + 1.0).log();
    let g = raw.clone().greater_elem(20.0).float().mul(raw.clone())
        + raw.lower_equal_elem(20.0).float().mul(softplus);
    let g = g * a_log.clone().exp().neg().unsqueeze_dim(0);
    let beta = sigmoid(b);
    (g, beta)
}

fn gated_delta_gpu(
    q: Tensor<Wgpu, 3>,
    k: Tensor<Wgpu, 3>,
    v: Tensor<Wgpu, 3>,
    g: Tensor<Wgpu, 2>,
    beta: Tensor<Wgpu, 2>,
    state: Tensor<Wgpu, 3>,
) -> (Tensor<Wgpu, 3>, Tensor<Wgpu, 3>) {
    crate::delta::scan(q, k, v, g, beta, state)
}

fn gate_norm_gpu(core: Tensor<Wgpu, 3>, z: Tensor<Wgpu, 2>, weight: &Tensor<Wgpu, 1>) -> Tensor<Wgpu, 2> {
    let seq = core.dims()[0];
    let inv = (core.clone().powf_scalar(2.0).sum_dim(2) / (LIN_DIM as f32) + 1e-6)
        .sqrt()
        .recip();
    let z = z.reshape([seq, V_HEADS, LIN_DIM]);
    let silu_z = z.clone() * sigmoid(z);
    let weight = weight.clone().reshape([1, 1, LIN_DIM]);
    (core * inv * silu_z * weight).reshape([seq, VALUE_DIM])
}

fn full_block(
    hidden: Tensor<Wgpu, 2>,
    layer: &FullBlock,
    cos: &Tensor<Wgpu, 2>,
    sin: &Tensor<Wgpu, 2>,
    past: Option<(&Tensor<Wgpu, 3>, &Tensor<Wgpu, 3>)>,
) -> (Tensor<Wgpu, 2>, Carry) {
    let seq = hidden.dims()[0];
    let normed = rms35(hidden.clone(), &layer.input_norm);
    let projected = linear2(&normed, &layer.q, None).reshape([seq, TEXT_HEADS, TEXT_HEAD_DIM * 2]);
    let q = projected.clone().narrow(2, 0, TEXT_HEAD_DIM).swap_dims(0, 1);
    let gate = sigmoid(
        projected
            .narrow(2, TEXT_HEAD_DIM, TEXT_HEAD_DIM)
            .reshape([seq, TEXT_HEADS * TEXT_HEAD_DIM]),
    );
    let k = heads(&linear2(&normed, &layer.k, None), seq, TEXT_KV);
    let v = heads(&linear2(&normed, &layer.v, None), seq, TEXT_KV);
    let q = rope_partial(rms_heads(q, &layer.q_norm), cos, sin);
    let k = rope_partial(rms_heads(k, &layer.k_norm), cos, sin);
    let k = repeat_kv(k);
    let v = repeat_kv(v);
    let k_all = match past {
        Some((past_k, _)) => Tensor::cat(vec![past_k.clone(), k.clone()], 1),
        None => k.clone(),
    };
    let v_all = match past {
        Some((_, past_v)) => Tensor::cat(vec![past_v.clone(), v.clone()], 1),
        None => v.clone(),
    };
    let mixed = attention(
        q.unsqueeze_dim::<4>(0),
        k_all.clone().unsqueeze_dim::<4>(0),
        v_all.clone().unsqueeze_dim::<4>(0),
        None,
        None,
        AttentionModuleOptions {
            is_causal: true,
            ..AttentionModuleOptions::default()
        },
    )
    .squeeze_dim::<3>(0)
    .swap_dims(0, 1)
    .reshape([seq, TEXT_HEADS * TEXT_HEAD_DIM])
        * gate;
    let hidden = hidden + linear2(&mixed, &layer.o, None);
    let normed = rms35(hidden.clone(), &layer.post_norm);
    (
        hidden + mlp(&normed, &layer.gate, &layer.up, &layer.down),
        Carry::Full { k: k_all, v: v_all },
    )
}

fn mlp(
    x: &Tensor<Wgpu, 2>,
    gate: &Tensor<Wgpu, 2>,
    up: &Tensor<Wgpu, 2>,
    down: &Tensor<Wgpu, 2>,
) -> Tensor<Wgpu, 2> {
    let gated = silu(linear2(x, gate, None));
    let up = linear2(x, up, None);
    linear2(&(gated * up), down, None)
}

pub fn block_coords(grid_h: usize, grid_w: usize) -> Vec<(usize, usize)> {
    let mut coords = Vec::with_capacity(grid_h * grid_w);
    for block_row in 0..(grid_h / MERGE) {
        for block_col in 0..(grid_w / MERGE) {
            for in_row in 0..MERGE {
                for in_col in 0..MERGE {
                    coords.push((block_row * MERGE + in_row, block_col * MERGE + in_col));
                }
            }
        }
    }
    coords
}

fn vision_block(
    mut hidden: Tensor<Wgpu, 2>,
    layer: &VisionLayer,
    cos: &Tensor<Wgpu, 2>,
    sin: &Tensor<Wgpu, 2>,
) -> Tensor<Wgpu, 2> {
    let normed = layer_norm(hidden.clone(), &layer.norm1_w, &layer.norm1_b);
    hidden = hidden + vision_attn(&normed, layer, cos, sin);
    let normed = layer_norm(hidden.clone(), &layer.norm2_w, &layer.norm2_b);
    let mid = gelu_tanh(linear2(&normed, &layer.fc1_w, Some(&layer.fc1_b)));
    hidden + linear2(&mid, &layer.fc2_w, Some(&layer.fc2_b))
}

fn vision_attn(
    x: &Tensor<Wgpu, 2>,
    layer: &VisionLayer,
    cos: &Tensor<Wgpu, 2>,
    sin: &Tensor<Wgpu, 2>,
) -> Tensor<Wgpu, 2> {
    let seq = x.dims()[0];
    let qkv = linear2(x, &layer.qkv_w, Some(&layer.qkv_b));
    let q = rope_vision(take_heads(&qkv, seq, 0), cos, sin);
    let k = rope_vision(take_heads(&qkv, seq, 1), cos, sin);
    let v = take_heads(&qkv, seq, 2);
    let scale = (VISION_HEAD_DIM as f32).sqrt();
    let scores = q.matmul(k.swap_dims(1, 2)) / scale;
    let mixed = softmax(scores, 2)
        .matmul(v)
        .swap_dims(0, 1)
        .reshape([seq, VISION_HIDDEN]);
    linear2(&mixed, &layer.proj_w, Some(&layer.proj_b))
}

fn take_heads(qkv: &Tensor<Wgpu, 2>, seq: usize, which: usize) -> Tensor<Wgpu, 3> {
    qkv.clone()
        .reshape([seq, 3, VISION_HEADS, VISION_HEAD_DIM])
        .narrow(1, which, 1)
        .reshape([seq, VISION_HEADS, VISION_HEAD_DIM])
        .swap_dims(0, 1)
}

fn rope_vision(x: Tensor<Wgpu, 3>, cos: &Tensor<Wgpu, 2>, sin: &Tensor<Wgpu, 2>) -> Tensor<Wgpu, 3> {
    let half = VISION_HEAD_DIM / 2;
    let x1 = x.clone().narrow(2, 0, half);
    let x2 = x.clone().narrow(2, half, half);
    let rotated = Tensor::cat(vec![-x2, x1], 2);
    x * cos.clone().unsqueeze_dim::<3>(0) + rotated * sin.clone().unsqueeze_dim::<3>(0)
}

fn vision_rope(grid_h: usize, grid_w: usize, device: &WgpuDevice) -> (Tensor<Wgpu, 2>, Tensor<Wgpu, 2>) {
    let coords = block_coords(grid_h, grid_w);
    let n = coords.len();
    let mut cos = vec![0f32; n * VISION_HEAD_DIM];
    let mut sin = vec![0f32; n * VISION_HEAD_DIM];
    let spatial = VISION_HEAD_DIM / 2;
    for (index, (row, col)) in coords.into_iter().enumerate() {
        for i in 0..(spatial / 2) {
            let inv = 1.0 / 10000f32.powf((2 * i) as f32 / spatial as f32);
            let (ch, sh) = ((row as f32 * inv).cos(), (row as f32 * inv).sin());
            let (cw, sw) = ((col as f32 * inv).cos(), (col as f32 * inv).sin());
            let base = index * VISION_HEAD_DIM;
            cos[base + i] = ch;
            sin[base + i] = sh;
            cos[base + spatial / 2 + i] = cw;
            sin[base + spatial / 2 + i] = sw;
            cos[base + spatial + i] = ch;
            sin[base + spatial + i] = sh;
            cos[base + spatial + spatial / 2 + i] = cw;
            sin[base + spatial + spatial / 2 + i] = sw;
        }
    }
    (
        Tensor::from_data(TensorData::new(cos, [n, VISION_HEAD_DIM]), device),
        Tensor::from_data(TensorData::new(sin, [n, VISION_HEAD_DIM]), device),
    )
}

fn merge(hidden: &Tensor<Wgpu, 2>, merger: &Merger) -> Tensor<Wgpu, 2> {
    let seq = hidden.dims()[0];
    let groups = seq / (MERGE * MERGE);
    let wide = layer_norm(hidden.clone(), &merger.norm_w, &merger.norm_b)
        .reshape([groups, VISION_HIDDEN * MERGE * MERGE]);
    let mid = gelu(linear2(&wide, &merger.fc1_w, Some(&merger.fc1_b)));
    linear2(&mid, &merger.fc2_w, Some(&merger.fc2_b))
}

fn heads(x: &Tensor<Wgpu, 2>, seq: usize, n: usize) -> Tensor<Wgpu, 3> {
    x.clone().reshape([seq, n, TEXT_HEAD_DIM]).swap_dims(0, 1)
}

fn rms_heads(x: Tensor<Wgpu, 3>, weight: &Tensor<Wgpu, 1>) -> Tensor<Wgpu, 3> {
    let variance = x.clone().powf_scalar(2.0).mean_dim(2);
    let normed = x / (variance + 1e-6).sqrt();
    normed * (weight.clone().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0) + 1.0)
}

fn repeat_kv(x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
    let seq = x.dims()[1];
    x.unsqueeze_dim::<4>(2)
        .expand([TEXT_KV, seq, TEXT_GROUP, TEXT_HEAD_DIM])
        .swap_dims(1, 2)
        .reshape([TEXT_HEADS, seq, TEXT_HEAD_DIM])
}

fn rope_partial(x: Tensor<Wgpu, 3>, cos: &Tensor<Wgpu, 2>, sin: &Tensor<Wgpu, 2>) -> Tensor<Wgpu, 3> {
    let rot = x.clone().narrow(2, 0, ROTARY);
    let pass = x.narrow(2, ROTARY, TEXT_HEAD_DIM - ROTARY);
    let half = ROTARY / 2;
    let x1 = rot.clone().narrow(2, 0, half);
    let x2 = rot.clone().narrow(2, half, half);
    let rotated = Tensor::cat(vec![-x2, x1], 2);
    let spun = rot * cos.clone().unsqueeze_dim::<3>(0) + rotated * sin.clone().unsqueeze_dim::<3>(0);
    Tensor::cat(vec![spun, pass], 2)
}

/// Three-axis rope positions for the whole sequence. Text tokens advance all
/// axes together; an image block spreads over the height and width axes and
/// then advances the cursor by the longer side.
pub fn positions(len: usize, image: Option<ImageSpan>) -> Vec<[i32; 3]> {
    let mut pos = vec![[0i32; 3]; len];
    let mut cursor = 0i32;
    let mut index = 0usize;
    if let Some(span) = image {
        while index < span.at {
            pos[index] = [cursor, cursor, cursor];
            cursor += 1;
            index += 1;
        }
        for h in 0..span.llm_h {
            for w in 0..span.llm_w {
                pos[span.at + h * span.llm_w + w] = [cursor, cursor + h as i32, cursor + w as i32];
            }
        }
        cursor += span.llm_h.max(span.llm_w) as i32;
        index = span.at + span.llm_h * span.llm_w;
    }
    while index < len {
        pos[index] = [cursor, cursor, cursor];
        cursor += 1;
        index += 1;
    }
    pos
}

/// Interleaved multimodal rope tables for the given positions.
pub fn mrope(positions: &[[i32; 3]], device: &WgpuDevice) -> (Tensor<Wgpu, 2>, Tensor<Wgpu, 2>) {
    let pairs = ROTARY / 2;
    let mut cos = vec![0f32; positions.len() * ROTARY];
    let mut sin = vec![0f32; positions.len() * ROTARY];
    for (index, axes) in positions.iter().enumerate() {
        let mut freq = [[0f32; 32]; 3];
        for axis in 0..3 {
            for i in 0..pairs {
                let inv = 1.0 / ROPE_THETA.powf((2 * i) as f32 / ROTARY as f32);
                freq[axis][i] = axes[axis] as f32 * inv;
            }
        }
        let mut mixed = freq[0];
        for (axis, offset) in [(1usize, 1usize), (2, 2)] {
            let length = MROPE[axis] * 3;
            let mut slot = offset;
            while slot < length && slot < pairs {
                mixed[slot] = freq[axis][slot];
                slot += 3;
            }
        }
        let base = index * ROTARY;
        for i in 0..pairs {
            let (c, s) = (mixed[i].cos(), mixed[i].sin());
            cos[base + i] = c;
            sin[base + i] = s;
            cos[base + pairs + i] = c;
            sin[base + pairs + i] = s;
        }
    }
    (
        Tensor::from_data(TensorData::new(cos, [positions.len(), ROTARY]), device),
        Tensor::from_data(TensorData::new(sin, [positions.len(), ROTARY]), device),
    )
}

fn linear2(x: &Tensor<Wgpu, 2>, weight: &Tensor<Wgpu, 2>, bias: Option<&Tensor<Wgpu, 1>>) -> Tensor<Wgpu, 2> {
    let y = if weight.dtype() == FloatDType::F16.into() {
        x.clone()
            .cast(FloatDType::F16)
            .matmul(weight.clone().transpose())
            .cast(FloatDType::F32)
    } else {
        x.clone().matmul(weight.clone().transpose())
    };
    match bias {
        Some(bias) => y + bias.clone().unsqueeze_dim::<2>(0),
        None => y,
    }
}

fn rms35(x: Tensor<Wgpu, 2>, weight: &Tensor<Wgpu, 1>) -> Tensor<Wgpu, 2> {
    let variance = x.clone().powf_scalar(2.0).mean_dim(1);
    let normed = x / (variance + 1e-6).sqrt();
    normed * (weight.clone().unsqueeze_dim::<2>(0) + 1.0)
}

fn layer_norm(x: Tensor<Wgpu, 2>, weight: &Tensor<Wgpu, 1>, bias: &Tensor<Wgpu, 1>) -> Tensor<Wgpu, 2> {
    let mean = x.clone().mean_dim(1);
    let centered = x - mean;
    let variance = centered.clone().powf_scalar(2.0).mean_dim(1);
    centered / (variance + 1e-6).sqrt() * weight.clone().unsqueeze_dim::<2>(0)
        + bias.clone().unsqueeze_dim::<2>(0)
}

fn silu(x: Tensor<Wgpu, 2>) -> Tensor<Wgpu, 2> {
    x.clone() * sigmoid(x)
}

fn gelu_tanh(x: Tensor<Wgpu, 2>) -> Tensor<Wgpu, 2> {
    let cube = x.clone() * x.clone() * x.clone();
    let inner = (x.clone() + cube * 0.044715) * (2.0 / std::f32::consts::PI).sqrt();
    x * (inner.tanh() + 1.0) * 0.5
}

/// 32 rope pairs for `ROTARY` 64; `MROPE` sections fit inside them.
const _: () = assert!(ROTARY / 2 == 32);
