//! Vision mode: `yah01/vjev-vision` on the WGPU device.
//!
//! The caller places the snapshot in a directory. This module does not download it.
//! The trunk is Qwen3.5-4B with its vision tower. `head.pt` scores every option in one pass.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::{Tensor, TensorData};
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;
use tokenizers::Tokenizer;

use crate::answers::{ChoiceAnswer, NoulAnswer, ScoreAnswer};
use crate::Error;

const IMAGE_PAD: u32 = 248056;
const VISION_START: u32 = 248053;
const VISION_END: u32 = 248054;
const PATCH: usize = 16;
const MERGE: usize = 2;
const FACTOR: usize = PATCH * MERGE;
const MAX_SIDE: usize = 512;
const VISION_LAYERS: usize = 24;
const VISION_HIDDEN: usize = 1024;
const VISION_HEADS: usize = 16;
const VISION_HEAD_DIM: usize = VISION_HIDDEN / VISION_HEADS;
const TEXT_LAYERS: usize = 32;
const TEXT_HIDDEN: usize = 2560;
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

pub struct RgbImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

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

struct Store {
    files: HashMap<String, (PathBuf, Option<Vec<u8>>)>,
    map: HashMap<String, String>,
    left: HashMap<String, usize>,
}

pub struct VisionDecision {
    tokenizer: Tokenizer,
    device: WgpuDevice,
    embed: Vec<f32>,
    text_norm: Tensor<Wgpu, 1>,
    blocks: Vec<Block>,
    patch_weight: Tensor<Wgpu, 2>,
    patch_bias: Tensor<Wgpu, 1>,
    pos_embed: Tensor<Wgpu, 2>,
    vision_layers: Vec<VisionLayer>,
    merger: Merger,
    head_w: Vec<f32>,
    head_b: f32,
}

impl VisionDecision {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, Error> {
        let dir = dir.as_ref();
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|err| Error::Weights {
            message: format!("tokenizer.json: {err}"),
        })?;
        let index = std::fs::read_to_string(dir.join("model.safetensors.index.json")).map_err(|_| {
            Error::MissingFile {
                dir: dir.to_path_buf(),
                file: "model.safetensors.index.json",
            }
        })?;
        let (head_w, head_b) = read_head(&dir.join("head.pt"))?;
        let device = WgpuDevice::default();
        let mut store = Store::parse(dir, &index)?;
        let (embed, embed_shape) = store.read("model.language_model.embed_tokens.weight")?;
        if embed_shape != [248320, TEXT_HIDDEN] {
            return Err(Error::Weights {
                message: format!("token table has shape {embed_shape:?}"),
            });
        }
        let text_norm = store.vector("model.language_model.norm.weight", &device)?;
        let mut blocks = Vec::with_capacity(TEXT_LAYERS);
        for index in 0..TEXT_LAYERS {
            let prefix = format!("model.language_model.layers.{index}");
            let input_norm = store.vector(&format!("{prefix}.input_layernorm.weight"), &device)?;
            let post_norm = store.vector(&format!("{prefix}.post_attention_layernorm.weight"), &device)?;
            let gate = store.matrix(&format!("{prefix}.mlp.gate_proj.weight"), &device)?;
            let up = store.matrix(&format!("{prefix}.mlp.up_proj.weight"), &device)?;
            let down = store.matrix(&format!("{prefix}.mlp.down_proj.weight"), &device)?;
            if index % 4 == 3 {
                blocks.push(Block::Full(FullBlock {
                    q: store.matrix(&format!("{prefix}.self_attn.q_proj.weight"), &device)?,
                    k: store.matrix(&format!("{prefix}.self_attn.k_proj.weight"), &device)?,
                    v: store.matrix(&format!("{prefix}.self_attn.v_proj.weight"), &device)?,
                    o: store.matrix(&format!("{prefix}.self_attn.o_proj.weight"), &device)?,
                    q_norm: store.vector(&format!("{prefix}.self_attn.q_norm.weight"), &device)?,
                    k_norm: store.vector(&format!("{prefix}.self_attn.k_norm.weight"), &device)?,
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
                blocks.push(Block::Linear(LinearBlock {
                    qkv: store.matrix(&format!("{prefix}.linear_attn.in_proj_qkv.weight"), &device)?,
                    z: store.matrix(&format!("{prefix}.linear_attn.in_proj_z.weight"), &device)?,
                    a: store.matrix(&format!("{prefix}.linear_attn.in_proj_a.weight"), &device)?,
                    b: store.matrix(&format!("{prefix}.linear_attn.in_proj_b.weight"), &device)?,
                    conv,
                    a_log,
                    dt_bias,
                    norm: store.vector(&format!("{prefix}.linear_attn.norm.weight"), &device)?,
                    out: store.matrix(&format!("{prefix}.linear_attn.out_proj.weight"), &device)?,
                    input_norm,
                    post_norm,
                    gate,
                    up,
                    down,
                }));
            }
        }
        let (patch_values, patch_shape) = store.read("model.visual.patch_embed.proj.weight")?;
        let patch_in: usize = patch_shape[1..].iter().product();
        let patch_weight = Tensor::from_data(TensorData::new(patch_values, [patch_shape[0], patch_in]), &device);
        let patch_bias = store.vector("model.visual.patch_embed.proj.bias", &device)?;
        let pos_embed = store.matrix("model.visual.pos_embed.weight", &device)?;
        let mut vision_layers = Vec::with_capacity(VISION_LAYERS);
        for index in 0..VISION_LAYERS {
            let prefix = format!("model.visual.blocks.{index}");
            vision_layers.push(VisionLayer {
                norm1_w: store.vector(&format!("{prefix}.norm1.weight"), &device)?,
                norm1_b: store.vector(&format!("{prefix}.norm1.bias"), &device)?,
                norm2_w: store.vector(&format!("{prefix}.norm2.weight"), &device)?,
                norm2_b: store.vector(&format!("{prefix}.norm2.bias"), &device)?,
                qkv_w: store.matrix(&format!("{prefix}.attn.qkv.weight"), &device)?,
                qkv_b: store.vector(&format!("{prefix}.attn.qkv.bias"), &device)?,
                proj_w: store.matrix(&format!("{prefix}.attn.proj.weight"), &device)?,
                proj_b: store.vector(&format!("{prefix}.attn.proj.bias"), &device)?,
                fc1_w: store.matrix(&format!("{prefix}.mlp.linear_fc1.weight"), &device)?,
                fc1_b: store.vector(&format!("{prefix}.mlp.linear_fc1.bias"), &device)?,
                fc2_w: store.matrix(&format!("{prefix}.mlp.linear_fc2.weight"), &device)?,
                fc2_b: store.vector(&format!("{prefix}.mlp.linear_fc2.bias"), &device)?,
            });
        }
        let merger = store.merger("model.visual.merger", &device)?;
        store.drop_bytes();
        Ok(Self {
            tokenizer,
            device,
            embed,
            text_norm,
            blocks,
            patch_weight,
            patch_bias,
            pos_embed,
            vision_layers,
            merger,
            head_w,
            head_b,
        })
    }

    pub fn choice(
        &self,
        image: &RgbImage,
        state: &str,
        instructions: &str,
        options: &[&str],
    ) -> Result<ChoiceAnswer, Error> {
        if options.is_empty() || options.len() > 255 {
            return Err(Error::OptionCount {
                label: "vision".into(),
                kind: "choice".into(),
                min: 1,
                max: 255,
                got: options.len(),
            });
        }
        let logits = self.score_options(image, state, instructions, options)?;
        let probs = softmax_vec(&logits);
        let best = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(index, _)| index)
            .unwrap_or(0);
        let mut probabilities = std::collections::BTreeMap::new();
        for (option, prob) in options.iter().zip(probs.iter()) {
            probabilities.insert((*option).to_string(), *prob);
        }
        Ok(ChoiceAnswer {
            choice: options[best].to_string(),
            confidence: listwise_confidence(&probs),
            probabilities,
        })
    }

    pub fn noul(&self, image: &RgbImage, state: &str, statement: &str) -> Result<NoulAnswer, Error> {
        let logit = self.score_options(image, state, statement, &[])?;
        let probability = 1.0 / (1.0 + (-logit[0]).exp());
        Ok(NoulAnswer {
            answer: probability >= 0.5,
            probability,
            confidence: probability.max(1.0 - probability),
        })
    }

    pub fn score(
        &self,
        image: &RgbImage,
        state: &str,
        instructions: &str,
        levels: &[&str],
    ) -> Result<ScoreAnswer, Error> {
        if levels.len() < 2 {
            return Err(Error::OptionCount {
                label: "vision".into(),
                kind: "score".into(),
                min: 2,
                max: 255,
                got: levels.len(),
            });
        }
        let answer = self.choice(image, state, instructions, levels)?;
        let mut expected = 0.0f32;
        for (index, level) in levels.iter().enumerate() {
            expected += index as f32 * answer.probabilities.get(*level).copied().unwrap_or(0.0);
        }
        let last = levels.len() - 1;
        let nearest = (expected.round() as usize).min(last);
        Ok(ScoreAnswer {
            score: expected,
            normalized: expected / last as f32,
            level: levels[nearest].to_string(),
            confidence: answer.confidence,
            probabilities: answer.probabilities,
        })
    }

    fn score_options(
        &self,
        image: &RgbImage,
        state: &str,
        instructions: &str,
        options: &[&str],
    ) -> Result<Vec<f32>, Error> {
        let fitted = fit_longest(image, MAX_SIDE);
        let (pixels, grid_h, grid_w) = patchify(&fitted).map_err(|message| Error::Weights { message })?;
        let n_image = (grid_h / MERGE) * (grid_w / MERGE);
        let (ids, slots) = self.render(state, instructions, n_image, options)?;
        let image_at = ids.iter().position(|id| *id == IMAGE_PAD).ok_or_else(|| Error::Weights {
            message: "prompt is missing image tokens".into(),
        })?;
        let rows = self.see(&pixels, grid_h, grid_w);
        let hidden = self.forward(&ids, image_at, n_image, grid_h, grid_w, &rows);
        let data = hidden.into_data();
        let values = data.as_slice::<f32>().unwrap();
        let mut logits = Vec::with_capacity(slots.len());
        for slot in slots {
            let mut logit = self.head_b;
            let row = slot * TEXT_HIDDEN;
            for dim in 0..TEXT_HIDDEN {
                logit += values[row + dim] * self.head_w[dim];
            }
            logits.push(logit);
        }
        Ok(logits)
    }

    fn render(
        &self,
        state: &str,
        instructions: &str,
        n_image: usize,
        options: &[&str],
    ) -> Result<(Vec<u32>, Vec<usize>), Error> {
        let mut ids = Vec::new();
        ids.push(VISION_START);
        ids.extend(std::iter::repeat(IMAGE_PAD).take(n_image));
        ids.push(VISION_END);
        if !state.is_empty() {
            ids.extend(self.encode(state));
        }
        ids.extend(self.encode(&format!("\n\nQuestion: {instructions}")));
        let mut slots = Vec::new();
        if options.is_empty() {
            slots.push(ids.len() - 1);
        } else {
            for (index, option) in options.iter().enumerate() {
                ids.extend(self.encode(&format!("\n({}) {option}", letter(index))));
            }
            ids.extend(self.encode("\nAnswer:"));
            for index in 0..options.len() {
                ids.extend(self.encode(&format!(" ({})", letter(index))));
                slots.push(ids.len() - 1);
            }
        }
        if ids.len() > 2560 + 1536 {
            return Err(Error::Weights {
                message: format!("question is {} tokens, over the checkpoint limit", ids.len()),
            });
        }
        Ok((ids, slots))
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer
            .encode(text, false)
            .map(|enc| enc.get_ids().to_vec())
            .unwrap_or_default()
    }

    fn see(&self, pixels: &[f32], grid_h: usize, grid_w: usize) -> Tensor<Wgpu, 2> {
        let n = pixels.len() / (3 * 2 * PATCH * PATCH);
        let flat = Tensor::<Wgpu, 2>::from_data(
            TensorData::new(pixels.to_vec(), [n, 3 * 2 * PATCH * PATCH]),
            &self.device,
        );
        let mut hidden = linear2(&flat, &self.patch_weight, Some(&self.patch_bias));
        hidden = hidden + self.position_embed(grid_h, grid_w);
        let (cos, sin) = vision_rope(grid_h, grid_w, &self.device);
        for layer in &self.vision_layers {
            hidden = vision_block(hidden, layer, &cos, &sin);
        }
        merge(&hidden, &self.merger, &self.device)
    }

    fn position_embed(&self, grid_h: usize, grid_w: usize) -> Tensor<Wgpu, 2> {
        let side = 48usize;
        let n = grid_h * grid_w;
        let mut gathered = vec![0f32; n * VISION_HIDDEN];
        let table = self.pos_embed.clone().into_data();
        let table = table.as_slice::<f32>().unwrap();
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

    fn forward(
        &self,
        ids: &[u32],
        image_at: usize,
        n_image: usize,
        grid_h: usize,
        grid_w: usize,
        image_rows: &Tensor<Wgpu, 2>,
    ) -> Tensor<Wgpu, 2> {
        let mut gathered = vec![0f32; ids.len() * TEXT_HIDDEN];
        for (index, id) in ids.iter().enumerate() {
            let src = *id as usize * TEXT_HIDDEN;
            gathered[index * TEXT_HIDDEN..(index + 1) * TEXT_HIDDEN]
                .copy_from_slice(&self.embed[src..src + TEXT_HIDDEN]);
        }
        let mut hidden = Tensor::from_data(TensorData::new(gathered, [ids.len(), TEXT_HIDDEN]), &self.device);
        let before = hidden.clone().narrow(0, 0, image_at);
        let after = hidden.clone().narrow(0, image_at + n_image, ids.len() - image_at - n_image);
        hidden = Tensor::cat(vec![before, image_rows.clone(), after], 0);
        let (cos, sin) = text_mrope(ids, image_at, n_image, grid_h, grid_w, &self.device);
        let mask = causal_mask(ids.len(), &self.device);
        for block in &self.blocks {
            hidden = match block {
                Block::Linear(layer) => linear_block(hidden, layer, &self.device),
                Block::Full(layer) => full_block(hidden, layer, &cos, &sin, &mask),
            };
        }
        rms35(hidden, &self.text_norm)
    }
}

fn linear_block(hidden: Tensor<Wgpu, 2>, layer: &LinearBlock, device: &WgpuDevice) -> Tensor<Wgpu, 2> {
    let seq = hidden.dims()[0];
    let normed = rms35(hidden.clone(), &layer.input_norm);
    let mixed = cpu_f32(&linear2(&normed, &layer.qkv, None));
    let conv = causal_conv(&mixed, &layer.conv, seq);
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
    let core = gated_delta(&q, &k, &v, &g, &beta, seq);
    let norm = layer.norm.clone().into_data().as_slice::<f32>().unwrap().to_vec();
    let gated = gate_norm(&core, &z, &norm);
    let core_t = Tensor::<Wgpu, 2>::from_data(TensorData::new(gated, [seq, VALUE_DIM]), device);
    let mixed = linear2(&core_t, &layer.out, None);
    let hidden = hidden + mixed;
    let normed = rms35(hidden.clone(), &layer.post_norm);
    hidden + mlp(&normed, &layer.gate, &layer.up, &layer.down)
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

fn causal_conv(mixed: &[f32], weight: &[f32], seq: usize) -> Vec<f32> {
    let mut out = vec![0f32; seq * CONV_DIM];
    for t in 0..seq {
        for channel in 0..CONV_DIM {
            let mut acc = 0.0f32;
            for tap in 0..CONV_K {
                let src = t as isize - (CONV_K as isize - 1 - tap as isize);
                if src >= 0 {
                    acc += mixed[src as usize * CONV_DIM + channel] * weight[channel * CONV_K + tap];
                }
            }
            let y = acc;
            out[t * CONV_DIM + channel] = y / (1.0 + (-y).exp());
        }
    }
    out
}

fn gated_delta(q: &[f32], k: &[f32], v: &[f32], g: &[f32], beta: &[f32], seq: usize) -> Vec<f32> {
    let mut state = vec![0f32; V_HEADS * LIN_DIM * LIN_DIM];
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
    mask: &Tensor<Wgpu, 2>,
) -> Tensor<Wgpu, 2> {
    let seq = hidden.dims()[0];
    let normed = rms35(hidden.clone(), &layer.input_norm);
    let projected = linear2(&normed, &layer.q, None).reshape([seq, TEXT_HEADS, TEXT_HEAD_DIM * 2]);
    let q = projected.clone().narrow(2, 0, TEXT_HEAD_DIM).swap_dims(0, 1);
    let gate = sigmoid(projected.narrow(2, TEXT_HEAD_DIM, TEXT_HEAD_DIM).reshape([seq, TEXT_HEADS * TEXT_HEAD_DIM]));
    let k = heads(&linear2(&normed, &layer.k, None), seq, TEXT_KV);
    let v = heads(&linear2(&normed, &layer.v, None), seq, TEXT_KV);
    let q = rope_partial(rms_heads(q, &layer.q_norm), cos, sin);
    let k = rope_partial(rms_heads(k, &layer.k_norm), cos, sin);
    let k = repeat_kv(k);
    let v = repeat_kv(v);
    let scores = q.clone().matmul(k.swap_dims(1, 2)) / (TEXT_HEAD_DIM as f32).sqrt();
    let probs = softmax(scores + mask.clone().unsqueeze_dim::<3>(0), 2);
    let mixed = probs.matmul(v).swap_dims(0, 1).reshape([seq, TEXT_HEADS * TEXT_HEAD_DIM]) * gate;
    let hidden = hidden + linear2(&mixed, &layer.o, None);
    let normed = rms35(hidden.clone(), &layer.post_norm);
    hidden + mlp(&normed, &layer.gate, &layer.up, &layer.down)
}

fn mlp(x: &Tensor<Wgpu, 2>, gate: &Tensor<Wgpu, 2>, up: &Tensor<Wgpu, 2>, down: &Tensor<Wgpu, 2>) -> Tensor<Wgpu, 2> {
    let gated = silu(linear2(x, gate, None));
    let up = linear2(x, up, None);
    linear2(&(gated * up), down, None)
}

impl Store {
    fn parse(dir: &Path, index: &str) -> Result<Self, Error> {
        let mut map = HashMap::new();
        for line in index.lines() {
            let line = line.trim().trim_end_matches(',');
            if let Some((name, file)) = line.split_once(':') {
                let name = name.trim().trim_matches('"');
                let file = file.trim().trim_matches('"');
                if name.starts_with("model.") {
                    map.insert(name.to_string(), file.to_string());
                }
            }
        }
        if map.is_empty() {
            return Err(Error::Weights {
                message: "weight index has no tensors".into(),
            });
        }
        let mut left = HashMap::new();
        for file in map.values() {
            *left.entry(file.clone()).or_default() += 1;
        }
        Ok(Self {
            files: map.values().cloned().map(|file| (file, (dir.to_path_buf(), None))).collect(),
            map,
            left,
        })
    }

    fn drop_bytes(&mut self) {
        for slot in self.files.values_mut() {
            slot.1 = None;
        }
    }

    fn matrix(&mut self, name: &str, device: &WgpuDevice) -> Result<Tensor<Wgpu, 2>, Error> {
        let (values, shape) = self.read(name)?;
        if shape.len() != 2 {
            return Err(Error::Weights {
                message: format!("{name} has shape {shape:?}"),
            });
        }
        Ok(Tensor::from_data(TensorData::new(values, [shape[0], shape[1]]), device))
    }

    fn vector(&mut self, name: &str, device: &WgpuDevice) -> Result<Tensor<Wgpu, 1>, Error> {
        let (values, shape) = self.read(name)?;
        if shape.len() != 1 {
            return Err(Error::Weights {
                message: format!("{name} has shape {shape:?}"),
            });
        }
        Ok(Tensor::from_data(TensorData::new(values, [shape[0]]), device))
    }

    fn merger(&mut self, prefix: &str, device: &WgpuDevice) -> Result<Merger, Error> {
        Ok(Merger {
            norm_w: self.vector(&format!("{prefix}.norm.weight"), device)?,
            norm_b: self.vector(&format!("{prefix}.norm.bias"), device)?,
            fc1_w: self.matrix(&format!("{prefix}.linear_fc1.weight"), device)?,
            fc1_b: self.vector(&format!("{prefix}.linear_fc1.bias"), device)?,
            fc2_w: self.matrix(&format!("{prefix}.linear_fc2.weight"), device)?,
            fc2_b: self.vector(&format!("{prefix}.linear_fc2.bias"), device)?,
        })
    }

    fn read(&mut self, name: &str) -> Result<(Vec<f32>, Vec<usize>), Error> {
        let file = self.map.get(name).cloned().ok_or_else(|| Error::Weights {
            message: format!("missing tensor {name}"),
        })?;
        let slot = self.files.get_mut(&file).ok_or_else(|| Error::Weights {
            message: format!("missing shard {file}"),
        })?;
        if slot.1.is_none() {
            let path = slot.0.join(&file);
            let bytes = std::fs::read(&path).map_err(|err| Error::Weights {
                message: format!("read {}: {err}", path.display()),
            })?;
            slot.1 = Some(bytes);
        }
        let bytes = slot.1.as_ref().unwrap();
        let tensors = SafeTensors::deserialize(bytes).map_err(|err| Error::Weights { message: err.to_string() })?;
        let view = tensors.tensor(name).map_err(|err| Error::Weights {
            message: format!("{name}: {err}"),
        })?;
        let shape = view.shape().to_vec();
        let values = to_f32(&view)?;
        let remaining = self.left.get_mut(&file).expect("shard count");
        *remaining -= 1;
        if *remaining == 0 {
            self.files.get_mut(&file).expect("shard").1 = None;
        }
        Ok((values, shape))
    }
}

fn fit_longest(image: &RgbImage, max_side: usize) -> RgbImage {
    let width = image.width as usize;
    let height = image.height as usize;
    let longest = width.max(height);
    if longest <= max_side {
        return RgbImage {
            width: image.width,
            height: image.height,
            data: image.data.clone(),
        };
    }
    let scale = max_side as f32 / longest as f32;
    let dw = (width as f32 * scale).round().max(1.0) as usize;
    let dh = (height as f32 * scale).round().max(1.0) as usize;
    RgbImage {
        width: dw as u32,
        height: dh as u32,
        data: resize(&image.data, width, height, dw, dh),
    }
}

fn patchify(image: &RgbImage) -> Result<(Vec<f32>, usize, usize), String> {
    if image.data.len() != image.width as usize * image.height as usize * 3 {
        return Err("RGB image byte length does not match its size".into());
    }
    let (height, width) = smart_size(image.height as usize, image.width as usize);
    let rgb = if height == image.height as usize && width == image.width as usize {
        image.data.clone()
    } else {
        resize(&image.data, image.width as usize, image.height as usize, width, height)
    };
    let grid_h = height / PATCH;
    let grid_w = width / PATCH;
    let mut out = Vec::with_capacity(grid_h * grid_w * 3 * 2 * PATCH * PATCH);
    for (row, col) in block_coords(grid_h, grid_w) {
        for channel in 0..3 {
            for _time in 0..2 {
                for py in 0..PATCH {
                    for px in 0..PATCH {
                        let y = row * PATCH + py;
                        let x = col * PATCH + px;
                        let pixel = rgb[(y * width + x) * 3 + channel] as f32 / 255.0;
                        out.push(pixel * 2.0 - 1.0);
                    }
                }
            }
        }
    }
    Ok((out, grid_h, grid_w))
}

fn smart_size(height: usize, width: usize) -> (usize, usize) {
    let h = (height / FACTOR).max(1) * FACTOR;
    let w = (width / FACTOR).max(1) * FACTOR;
    (h, w)
}

fn resize(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let mut out = vec![0u8; dw * dh * 3];
    for y in 0..dh {
        let sy = y * sh / dh;
        for x in 0..dw {
            let sx = x * sw / dw;
            let from = (sy * sw + sx) * 3;
            let to = (y * dw + x) * 3;
            out[to..to + 3].copy_from_slice(&src[from..from + 3]);
        }
    }
    out
}

fn block_coords(grid_h: usize, grid_w: usize) -> Vec<(usize, usize)> {
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

fn vision_block(mut hidden: Tensor<Wgpu, 2>, layer: &VisionLayer, cos: &Tensor<Wgpu, 2>, sin: &Tensor<Wgpu, 2>) -> Tensor<Wgpu, 2> {
    let normed = layer_norm(hidden.clone(), &layer.norm1_w, &layer.norm1_b);
    hidden = hidden + vision_attn(&normed, layer, cos, sin);
    let normed = layer_norm(hidden.clone(), &layer.norm2_w, &layer.norm2_b);
    let mid = gelu_tanh(linear2(&normed, &layer.fc1_w, Some(&layer.fc1_b)));
    hidden + linear2(&mid, &layer.fc2_w, Some(&layer.fc2_b))
}

fn vision_attn(x: &Tensor<Wgpu, 2>, layer: &VisionLayer, cos: &Tensor<Wgpu, 2>, sin: &Tensor<Wgpu, 2>) -> Tensor<Wgpu, 2> {
    let seq = x.dims()[0];
    let qkv = linear2(x, &layer.qkv_w, Some(&layer.qkv_b));
    let q = rope_vision(take_heads(&qkv, seq, 0), cos, sin);
    let k = rope_vision(take_heads(&qkv, seq, 1), cos, sin);
    let v = take_heads(&qkv, seq, 2);
    let scores = q.clone().matmul(k.swap_dims(1, 2)) / (VISION_HEAD_DIM as f32).sqrt();
    let mixed = softmax(scores, 2).matmul(v).swap_dims(0, 1).reshape([seq, VISION_HIDDEN]);
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
    let wide = layer_norm(hidden.clone(), &merger.norm_w, &merger.norm_b).reshape([groups, VISION_HIDDEN * MERGE * MERGE]);
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

fn text_mrope(
    ids: &[u32],
    image_at: usize,
    n_image: usize,
    grid_h: usize,
    grid_w: usize,
    device: &WgpuDevice,
) -> (Tensor<Wgpu, 2>, Tensor<Wgpu, 2>) {
    let llm_h = grid_h / MERGE;
    let llm_w = grid_w / MERGE;
    let mut pos = vec![[0i32; 3]; ids.len()];
    let mut cursor = 0i32;
    for index in 0..image_at {
        pos[index] = [cursor, cursor, cursor];
        cursor += 1;
    }
    for h in 0..llm_h {
        for w in 0..llm_w {
            let slot = image_at + h * llm_w + w;
            pos[slot] = [cursor, cursor + h as i32, cursor + w as i32];
        }
    }
    cursor += llm_h.max(llm_w) as i32;
    for index in image_at + n_image..ids.len() {
        pos[index] = [cursor, cursor, cursor];
        cursor += 1;
    }
    let pairs = ROTARY / 2;
    let mut cos = vec![0f32; ids.len() * ROTARY];
    let mut sin = vec![0f32; ids.len() * ROTARY];
    for (index, axes) in pos.iter().enumerate() {
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
        Tensor::from_data(TensorData::new(cos, [ids.len(), ROTARY]), device),
        Tensor::from_data(TensorData::new(sin, [ids.len(), ROTARY]), device),
    )
}

fn causal_mask(n: usize, device: &WgpuDevice) -> Tensor<Wgpu, 2> {
    let mut mask = vec![0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            mask[i * n + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_data(TensorData::new(mask, [n, n]), device)
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
    centered / (variance + 1e-6).sqrt() * weight.clone().unsqueeze_dim::<2>(0) + bias.clone().unsqueeze_dim::<2>(0)
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
            let poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
            let erf = sign * (1.0 - poly * (-a * a).exp());
            0.5 * value * (1.0 + erf)
        })
        .collect();
    Tensor::from_data(TensorData::new(values, [dims[0], dims[1]]), device)
}

fn cpu_f32(tensor: &Tensor<Wgpu, 2>) -> Vec<f32> {
    tensor.clone().into_data().as_slice::<f32>().unwrap().to_vec()
}

fn letter(index: usize) -> String {
    let mut n = index + 1;
    let mut out = String::new();
    while n > 0 {
        n -= 1;
        out.insert(0, (b'A' + (n % 26) as u8) as char);
        n /= 26;
    }
    out
}

fn listwise_confidence(probs: &[f32]) -> f32 {
    let n = probs.len();
    if n <= 1 {
        return 0.0;
    }
    let max = probs.iter().copied().fold(0.0f32, f32::max);
    ((max - 1.0 / n as f32) / (1.0 - 1.0 / n as f32)).max(0.0)
}

fn softmax_vec(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = values.iter().map(|value| (value - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.into_iter().map(|value| value / sum).collect()
}

fn to_f32(view: &TensorView<'_>) -> Result<Vec<f32>, Error> {
    let bytes = view.data();
    match view.dtype() {
        safetensors::Dtype::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()),
        safetensors::Dtype::BF16 => Ok(bytes
            .chunks_exact(2)
            .map(|chunk| f32::from_bits((u16::from_le_bytes([chunk[0], chunk[1]]) as u32) << 16))
            .collect()),
        other => Err(Error::Weights {
            message: format!("unsupported dtype {other:?}"),
        }),
    }
}

fn read_head(path: &Path) -> Result<(Vec<f32>, f32), Error> {
    let bytes = std::fs::read(path).map_err(|_| Error::MissingFile {
        dir: path.parent().unwrap_or(Path::new(".")).to_path_buf(),
        file: "head.pt",
    })?;
    let mut weight = None;
    let mut bias = None;
    let mut cursor = 0usize;
    while cursor + 30 <= bytes.len() && &bytes[cursor..cursor + 4] == b"PK\x03\x04" {
        let method = u16::from_le_bytes([bytes[cursor + 8], bytes[cursor + 9]]);
        let size = u32::from_le_bytes(bytes[cursor + 18..cursor + 22].try_into().unwrap()) as usize;
        let name_len = u16::from_le_bytes([bytes[cursor + 26], bytes[cursor + 27]]) as usize;
        let extra_len = u16::from_le_bytes([bytes[cursor + 28], bytes[cursor + 29]]) as usize;
        let name_at = cursor + 30;
        let data_at = name_at + name_len + extra_len;
        if data_at + size > bytes.len() {
            break;
        }
        let name = std::str::from_utf8(&bytes[name_at..name_at + name_len]).unwrap_or("");
        let (payload, next) = if size == 0 {
            let marker = bytes[data_at..]
                .windows(4)
                .position(|mark| mark == b"PK\x03\x04" || mark == b"PK\x01\x02")
                .map(|at| data_at + at)
                .unwrap_or(bytes.len());
            let end = if marker >= 16 && &bytes[marker - 16..marker - 12] == b"PK\x07\x08" {
                marker - 16
            } else {
                marker
            };
            (&bytes[data_at..end], marker)
        } else {
            (&bytes[data_at..data_at + size], data_at + size)
        };
        if method == 0 && name.ends_with("data/0") {
            weight = Some(f32s(payload));
        } else if method == 0 && name.ends_with("data/1") {
            bias = f32s(payload).first().copied();
        }
        cursor = next;
    }
    match (weight, bias) {
        (Some(weight), Some(bias)) if weight.len() == TEXT_HIDDEN => Ok((weight, bias)),
        _ => Err(Error::Weights {
            message: "head.pt did not contain the listwise weight and bias".into(),
        }),
    }
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
