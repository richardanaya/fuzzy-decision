//! The Clef-Flash backbone: Qwen3.5-9B with its vision encoder, on Burn's
//! WGPU device.
//!
//! The language model interleaves gated-delta linear-attention layers with a
//! full-attention layer every fourth block. The linear-attention recurrence
//! runs on the CPU in f32; the projections, full attention, and MLPs run on
//! the GPU. Weights are read from the sharded `model-*.safetensors` files of a
//! local `Cloudflare/clef-flash` snapshot.

use std::path::Path;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::activation::{sigmoid, softmax};
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
const V_HEADS: usize = 32;
const LIN_DIM: usize = 128;
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
    qkv: Tensor<Wgpu, 2>,
    z: Tensor<Wgpu, 2>,
    a: Tensor<Wgpu, 2>,
    b: Tensor<Wgpu, 2>,
    conv: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Vec<f32>,
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
    Linear { mixed_tail: Vec<f32>, state: Vec<f32> },
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
                let (a_log, _) = store.read(&format!("{prefix}.linear_attn.A_log"))?;
                let (dt_bias, _) = store.read(&format!("{prefix}.linear_attn.dt_bias"))?;
                let (norm, _) = store.read(&format!("{prefix}.linear_attn.norm.weight"))?;
                blocks.push(Block::Linear(LinearBlock {
                    qkv: matrix(&mut store, &format!("{prefix}.linear_attn.in_proj_qkv.weight"), device)?,
                    z: matrix(&mut store, &format!("{prefix}.linear_attn.in_proj_z.weight"), device)?,
                    a: matrix(&mut store, &format!("{prefix}.linear_attn.in_proj_a.weight"), device)?,
                    b: matrix(&mut store, &format!("{prefix}.linear_attn.in_proj_b.weight"), device)?,
                    conv,
                    a_log,
                    dt_bias,
                    norm,
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
                    qkv_w: matrix(&mut store, &format!("{prefix}.attn.qkv.weight"), device)?,
                    qkv_b: vector(&mut store, &format!("{prefix}.attn.qkv.bias"), device)?,
                    proj_w: matrix(&mut store, &format!("{prefix}.attn.proj.weight"), device)?,
                    proj_b: vector(&mut store, &format!("{prefix}.attn.proj.bias"), device)?,
                    fc1_w: matrix(&mut store, &format!("{prefix}.mlp.linear_fc1.weight"), device)?,
                    fc1_b: vector(&mut store, &format!("{prefix}.mlp.linear_fc1.bias"), device)?,
                    fc2_w: matrix(&mut store, &format!("{prefix}.mlp.linear_fc2.weight"), device)?,
                    fc2_b: vector(&mut store, &format!("{prefix}.mlp.linear_fc2.bias"), device)?,
                });
            }
            let merger = Merger {
                norm_w: vector(&mut store, "model.visual.merger.norm.weight", device)?,
                norm_b: vector(&mut store, "model.visual.merger.norm.bias", device)?,
                fc1_w: matrix(&mut store, "model.visual.merger.linear_fc1.weight", device)?,
                fc1_b: vector(&mut store, "model.visual.merger.linear_fc1.bias", device)?,
                fc2_w: matrix(&mut store, "model.visual.merger.linear_fc2.weight", device)?,
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
        Ok(merge(&hidden, &tower.merger, &self.device))
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
            let (next, carry) = step_block(
                block,
                hidden,
                cos,
                sin,
                past.map(|item| &item[index]),
                &self.device,
            );
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
    let (values, shape) = store.read(name)?;
    if shape.len() != 2 {
        return Err(format!("{name} has shape {shape:?}"));
    }
    Ok(Tensor::from_data(TensorData::new(values, [shape[0], shape[1]]), device))
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
    device: &WgpuDevice,
) -> (Tensor<Wgpu, 2>, Carry) {
    match (block, past) {
        (Block::Linear(layer), Some(Carry::Linear { mixed_tail, state })) => {
            linear_block(hidden, layer, device, mixed_tail, state)
        }
        (Block::Linear(layer), None) => linear_block(
            hidden,
            layer,
            device,
            &[],
            &vec![0f32; V_HEADS * LIN_DIM * LIN_DIM],
        ),
        (Block::Full(layer), Some(Carry::Full { k, v })) => full_block(hidden, layer, cos, sin, Some((k, v))),
        (Block::Full(layer), None) => full_block(hidden, layer, cos, sin, None),
        _ => panic!("language layer cache does not match the block"),
    }
}

fn linear_block(
    hidden: Tensor<Wgpu, 2>,
    layer: &LinearBlock,
    device: &WgpuDevice,
    history: &[f32],
    state: &[f32],
) -> (Tensor<Wgpu, 2>, Carry) {
    let seq = hidden.dims()[0];
    let normed = rms35(hidden.clone(), &layer.input_norm);
    let mixed = cpu_f32(&linear2(&normed, &layer.qkv, None));
    let conv = causal_conv(&mixed, history, &layer.conv, seq);
    let z = cpu_f32(&linear2(&normed, &layer.z, None));
    let a = cpu_f32(&linear2(&normed, &layer.a, None));
    let b = cpu_f32(&linear2(&normed, &layer.b, None));
    let mut q = vec![0f32; seq * V_HEADS * LIN_DIM];
    let mut k = vec![0f32; seq * V_HEADS * LIN_DIM];
    let mut v = vec![0f32; seq * V_HEADS * LIN_DIM];
    for t in 0..seq {
        let base = t * CONV_DIM;
        l2_repeat(&conv[base..base + KEY_DIM], &mut q[t * V_HEADS * LIN_DIM..], true);
        l2_repeat(&conv[base + KEY_DIM..base + 2 * KEY_DIM], &mut k[t * V_HEADS * LIN_DIM..], false);
        let vsrc = &conv[base + 2 * KEY_DIM..base + CONV_DIM];
        v[t * VALUE_DIM..(t + 1) * VALUE_DIM].copy_from_slice(vsrc);
    }
    let mut g = vec![0f32; seq * V_HEADS];
    let mut beta = vec![0f32; seq * V_HEADS];
    for t in 0..seq {
        for head in 0..V_HEADS {
            let raw = a[t * V_HEADS + head] + layer.dt_bias[head];
            let softplus = if raw > 20.0 { raw } else { (1.0 + raw.exp()).ln() };
            g[t * V_HEADS + head] = -layer.a_log[head].exp() * softplus;
            let bb = b[t * V_HEADS + head];
            beta[t * V_HEADS + head] = 1.0 / (1.0 + (-bb).exp());
        }
    }
    let mut recurrent = state.to_vec();
    let core = gated_delta(&q, &k, &v, &g, &beta, seq, &mut recurrent);
    let gated = gate_norm(&core, &z, &layer.norm);
    let core_t = Tensor::<Wgpu, 2>::from_data(TensorData::new(gated, [seq, VALUE_DIM]), device);
    let projected = linear2(&core_t, &layer.out, None);
    let hidden = hidden + projected;
    let normed = rms35(hidden.clone(), &layer.post_norm);
    let keep = (CONV_K - 1).min(seq);
    let mixed_tail = if seq >= CONV_K - 1 {
        mixed[(seq - keep) * CONV_DIM..].to_vec()
    } else {
        let mut tail = history[history.len().saturating_sub((CONV_K - 1 - seq) * CONV_DIM)..].to_vec();
        tail.extend_from_slice(&mixed);
        tail
    };
    (
        hidden + mlp(&normed, &layer.gate, &layer.up, &layer.down),
        Carry::Linear {
            mixed_tail,
            state: recurrent,
        },
    )
}

fn l2_repeat(src: &[f32], dst: &mut [f32], scale_query: bool) {
    let scale = if scale_query { (LIN_DIM as f32).powf(-0.5) } else { 1.0 };
    for head in 0..K_HEADS {
        let slice = &src[head * LIN_DIM..(head + 1) * LIN_DIM];
        let mut sum = 1e-6f32;
        for value in slice {
            sum += value * value;
        }
        let inv = sum.sqrt().recip() * scale;
        for copy in 0..(V_HEADS / K_HEADS) {
            let at = (head * (V_HEADS / K_HEADS) + copy) * LIN_DIM;
            for dim in 0..LIN_DIM {
                dst[at + dim] = slice[dim] * inv;
            }
        }
    }
}

fn causal_conv(mixed: &[f32], history: &[f32], weight: &[f32], seq: usize) -> Vec<f32> {
    let hist = history.len() / CONV_DIM;
    let mut out = vec![0f32; seq * CONV_DIM];
    for t in 0..seq {
        for channel in 0..CONV_DIM {
            let mut acc = 0.0f32;
            for tap in 0..CONV_K {
                let src = t as isize - (CONV_K as isize - 1 - tap as isize);
                let value = if src >= 0 {
                    mixed[src as usize * CONV_DIM + channel]
                } else {
                    let earlier = hist as isize + src;
                    if earlier >= 0 {
                        history[earlier as usize * CONV_DIM + channel]
                    } else {
                        0.0
                    }
                };
                acc += value * weight[channel * CONV_K + tap];
            }
            out[t * CONV_DIM + channel] = acc / (1.0 + (-acc).exp());
        }
    }
    out
}

fn gated_delta(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    seq: usize,
    state: &mut [f32],
) -> Vec<f32> {
    let mut out = vec![0f32; seq * VALUE_DIM];
    for t in 0..seq {
        for head in 0..V_HEADS {
            let decay = g[t * V_HEADS + head].exp();
            let b = beta[t * V_HEADS + head];
            let base = head * LIN_DIM * LIN_DIM;
            for value in &mut state[base..base + LIN_DIM * LIN_DIM] {
                *value *= decay;
            }
            let token = (t * V_HEADS + head) * LIN_DIM;
            let mut kv = [0f32; LIN_DIM];
            for i in 0..LIN_DIM {
                let ks = k[token + i];
                for j in 0..LIN_DIM {
                    kv[j] += state[base + i * LIN_DIM + j] * ks;
                }
            }
            let mut delta = [0f32; LIN_DIM];
            for j in 0..LIN_DIM {
                delta[j] = (v[token + j] - kv[j]) * b;
            }
            for i in 0..LIN_DIM {
                let ks = k[token + i];
                for j in 0..LIN_DIM {
                    state[base + i * LIN_DIM + j] += ks * delta[j];
                }
            }
            for i in 0..LIN_DIM {
                let qs = q[token + i];
                for j in 0..LIN_DIM {
                    out[token + j] += state[base + i * LIN_DIM + j] * qs;
                }
            }
        }
    }
    out
}

fn gate_norm(core: &[f32], z: &[f32], weight: &[f32]) -> Vec<f32> {
    let rows = core.len() / LIN_DIM;
    let mut out = vec![0f32; core.len()];
    for row in 0..rows {
        let start = row * LIN_DIM;
        let mut var = 0.0f32;
        for dim in 0..LIN_DIM {
            let value = core[start + dim];
            var += value * value;
        }
        let inv = (var / LIN_DIM as f32 + 1e-6).sqrt().recip();
        for dim in 0..LIN_DIM {
            let gate = z[start + dim];
            let silu = gate / (1.0 + (-gate).exp());
            out[start + dim] = core[start + dim] * inv * weight[dim] * silu;
        }
    }
    out
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
    let past_len = past.map(|(key, _)| key.dims()[1]).unwrap_or(0);
    let k_all = match past {
        Some((past_k, _)) => Tensor::cat(vec![past_k.clone(), k.clone()], 1),
        None => k.clone(),
    };
    let v_all = match past {
        Some((_, past_v)) => Tensor::cat(vec![past_v.clone(), v.clone()], 1),
        None => v.clone(),
    };
    let scores = q.clone().matmul(k_all.clone().swap_dims(1, 2)) / (TEXT_HEAD_DIM as f32).sqrt();
    let mask = causal_mask_from(past_len, seq, &hidden.device());
    let probs = softmax(scores + mask.unsqueeze_dim::<3>(0), 2);
    let mixed = probs
        .matmul(v_all.clone())
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
    let scores = q.clone().matmul(k.swap_dims(1, 2)) / (VISION_HEAD_DIM as f32).sqrt();
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

fn merge(hidden: &Tensor<Wgpu, 2>, merger: &Merger, device: &WgpuDevice) -> Tensor<Wgpu, 2> {
    let seq = hidden.dims()[0];
    let groups = seq / (MERGE * MERGE);
    let wide = layer_norm(hidden.clone(), &merger.norm_w, &merger.norm_b)
        .reshape([groups, VISION_HIDDEN * MERGE * MERGE]);
    let mid = gelu_erf(linear2(&wide, &merger.fc1_w, Some(&merger.fc1_b)), device);
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

fn causal_mask_from(past: usize, seq: usize, device: &WgpuDevice) -> Tensor<Wgpu, 2> {
    let total = past + seq;
    let mut mask = vec![0f32; seq * total];
    for i in 0..seq {
        for j in (past + i + 1)..total {
            mask[i * total + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_data(TensorData::new(mask, [seq, total]), device)
}

fn linear2(x: &Tensor<Wgpu, 2>, weight: &Tensor<Wgpu, 2>, bias: Option<&Tensor<Wgpu, 1>>) -> Tensor<Wgpu, 2> {
    let y = x.clone().matmul(weight.clone().transpose());
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

fn gelu_erf(x: Tensor<Wgpu, 2>, device: &WgpuDevice) -> Tensor<Wgpu, 2> {
    let dims = x.dims();
    let data = x.into_data();
    let values = data
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .copied()
        .map(|value| {
            let sign = if value < 0.0 { -1.0 } else { 1.0 };
            let a = value.abs() / std::f32::consts::SQRT_2;
            let t = 1.0 / (1.0 + 0.3275911 * a);
            let poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
                + 0.254829592)
                * t;
            let erf = sign * (1.0 - poly * (-a * a).exp());
            0.5 * value * (1.0 + erf)
        })
        .collect();
    Tensor::from_data(TensorData::new(values, [dims[0], dims[1]]), device)
}

fn cpu_f32(tensor: &Tensor<Wgpu, 2>) -> Vec<f32> {
    tensor.clone().into_data().as_slice::<f32>().unwrap().to_vec()
}

/// 32 rope pairs for `ROTARY` 64; `MROPE` sections fit inside them.
const _: () = assert!(ROTARY / 2 == 32);
