//! Vision mode: `Qwen/Qwen3-VL-4B-Instruct` on the same WGPU device.
//!
//! The caller places the Hub snapshot in a directory. This module does not download it.
//! An image is patch-embedded, merged into the token sequence, and each option is scored
//! by the mean log-probability of its tokens as the assistant reply.

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

const IMAGE_PAD: u32 = 151655;
const VISION_START: u32 = 151652;
const VISION_END: u32 = 151653;
const PATCH: usize = 16;
const MERGE: usize = 2;
const FACTOR: usize = PATCH * MERGE;
const VISION_LAYERS: usize = 24;
const VISION_HIDDEN: usize = 1024;
const VISION_HEADS: usize = 16;
const VISION_HEAD_DIM: usize = VISION_HIDDEN / VISION_HEADS;
const TEXT_LAYERS: usize = 36;
const TEXT_HIDDEN: usize = 2560;
const TEXT_HEADS: usize = 32;
const TEXT_KV: usize = 8;
const TEXT_HEAD_DIM: usize = 128;
const TEXT_GROUP: usize = TEXT_HEADS / TEXT_KV;
const ROPE_THETA: f32 = 5_000_000.0;
const MROPE: [usize; 3] = [24, 20, 20];
const DEEPSTACK: [usize; 3] = [5, 11, 17];

pub struct RgbImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

pub struct VisionDecision {
    tokenizer: Tokenizer,
    device: WgpuDevice,
    embed: Tensor<Wgpu, 2>,
    text_norm: Tensor<Wgpu, 1>,
    text_layers: Vec<TextLayer>,
    patch_weight: Tensor<Wgpu, 2>,
    patch_bias: Tensor<Wgpu, 1>,
    pos_embed: Tensor<Wgpu, 2>,
    vision_layers: Vec<VisionLayer>,
    merger: Merger,
    deepstack: Vec<Merger>,
    im_start: u32,
    im_end: u32,
}

struct TextLayer {
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
    postshuffle: bool,
}

struct Store {
    files: HashMap<String, (PathBuf, Option<Vec<u8>>)>,
    map: HashMap<String, String>,
    left: HashMap<String, usize>,
}

impl VisionDecision {
    /// Load a local `Qwen/Qwen3-VL-4B-Instruct` snapshot.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, Error> {
        let dir = dir.as_ref();
        let device = WgpuDevice::default();
        let index_path = dir.join("model.safetensors.index.json");
        let index = std::fs::read_to_string(&index_path).map_err(|_| Error::MissingFile {
            dir: dir.to_path_buf(),
            file: "model.safetensors.index.json",
        })?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|err| Error::Weights {
            message: format!("read tokenizer: {err}"),
        })?;
        let im_start = tokenizer.token_to_id("<|im_start|>").ok_or_else(|| Error::Weights {
            message: "tokenizer is missing <|im_start|>".into(),
        })?;
        let im_end = tokenizer.token_to_id("<|im_end|>").ok_or_else(|| Error::Weights {
            message: "tokenizer is missing <|im_end|>".into(),
        })?;
        let mut store = Store::parse(dir, &index)?;

        let embed = store.matrix("model.language_model.embed_tokens.weight", &device)?;
        if embed.dims()[1] != TEXT_HIDDEN {
            return Err(Error::Weights {
                message: format!("text hidden size is {}, expected {TEXT_HIDDEN}", embed.dims()[1]),
            });
        }
        let text_norm = store.vector("model.language_model.norm.weight", &device)?;
        let mut text_layers = Vec::with_capacity(TEXT_LAYERS);
        for i in 0..TEXT_LAYERS {
            let p = format!("model.language_model.layers.{i}");
            text_layers.push(TextLayer {
                q: store.matrix(&format!("{p}.self_attn.q_proj.weight"), &device)?,
                k: store.matrix(&format!("{p}.self_attn.k_proj.weight"), &device)?,
                v: store.matrix(&format!("{p}.self_attn.v_proj.weight"), &device)?,
                o: store.matrix(&format!("{p}.self_attn.o_proj.weight"), &device)?,
                q_norm: store.vector(&format!("{p}.self_attn.q_norm.weight"), &device)?,
                k_norm: store.vector(&format!("{p}.self_attn.k_norm.weight"), &device)?,
                gate: store.matrix(&format!("{p}.mlp.gate_proj.weight"), &device)?,
                up: store.matrix(&format!("{p}.mlp.up_proj.weight"), &device)?,
                down: store.matrix(&format!("{p}.mlp.down_proj.weight"), &device)?,
                input_norm: store.vector(&format!("{p}.input_layernorm.weight"), &device)?,
                post_norm: store.vector(&format!("{p}.post_attention_layernorm.weight"), &device)?,
            });
        }

        let (patch_values, patch_shape) = store.read("model.visual.patch_embed.proj.weight")?;
        let patch_in: usize = patch_shape[1..].iter().product();
        let patch_weight = Tensor::from_data(
            TensorData::new(patch_values, [patch_shape[0], patch_in]),
            &device,
        );
        let patch_bias = store.vector("model.visual.patch_embed.proj.bias", &device)?;
        let pos_embed = store.matrix("model.visual.pos_embed.weight", &device)?;

        let mut vision_layers = Vec::with_capacity(VISION_LAYERS);
        for i in 0..VISION_LAYERS {
            let p = format!("model.visual.blocks.{i}");
            vision_layers.push(VisionLayer {
                norm1_w: store.vector(&format!("{p}.norm1.weight"), &device)?,
                norm1_b: store.vector(&format!("{p}.norm1.bias"), &device)?,
                norm2_w: store.vector(&format!("{p}.norm2.weight"), &device)?,
                norm2_b: store.vector(&format!("{p}.norm2.bias"), &device)?,
                qkv_w: store.matrix(&format!("{p}.attn.qkv.weight"), &device)?,
                qkv_b: store.vector(&format!("{p}.attn.qkv.bias"), &device)?,
                proj_w: store.matrix(&format!("{p}.attn.proj.weight"), &device)?,
                proj_b: store.vector(&format!("{p}.attn.proj.bias"), &device)?,
                fc1_w: store.matrix(&format!("{p}.mlp.linear_fc1.weight"), &device)?,
                fc1_b: store.vector(&format!("{p}.mlp.linear_fc1.bias"), &device)?,
                fc2_w: store.matrix(&format!("{p}.mlp.linear_fc2.weight"), &device)?,
                fc2_b: store.vector(&format!("{p}.mlp.linear_fc2.bias"), &device)?,
            });
        }
        let merger = store.merger("model.visual.merger", false, &device)?;
        let deepstack = (0..3)
            .map(|i| store.merger(&format!("model.visual.deepstack_merger_list.{i}"), true, &device))
            .collect::<Result<Vec<_>, _>>()?;
        store.drop_bytes();

        Ok(Self {
            tokenizer,
            device,
            embed,
            text_norm,
            text_layers,
            patch_weight,
            patch_bias,
            pos_embed,
            vision_layers,
            merger,
            deepstack,
            im_start,
            im_end,
        })
    }

    pub fn choice(
        &self,
        image: &RgbImage,
        state: &str,
        instructions: &str,
        options: &[&str],
    ) -> Result<ChoiceAnswer, Error> {
        if options.is_empty() {
            return Err(Error::OptionCount {
                label: "vision".into(),
                kind: "choice".into(),
                min: 1,
                max: 255,
                got: 0,
            });
        }
        let (pixels, grid_h, grid_w) = patchify(image).map_err(|message| Error::Weights { message })?;
        let n_image = (grid_h / MERGE) * (grid_w / MERGE);
        let prompt = self.prompt_ids(state, instructions, n_image)?;
        let image_at = prompt.iter().position(|id| *id == IMAGE_PAD).ok_or_else(|| Error::Weights {
            message: "prompt is missing image tokens".into(),
        })?;
        let (image_rows, deep_rows) = self.see(&pixels, grid_h, grid_w);
        let mut scores = Vec::with_capacity(options.len());
        for option in options {
            let extra = self.encode(option);
            if extra.is_empty() {
                return Err(Error::BadOption { label: "vision".into() });
            }
            let mut ids = prompt.clone();
            ids.extend(extra.iter().copied());
            let hidden = self.prefill(&ids, image_at, n_image, grid_h, grid_w, &image_rows, &deep_rows);
            let mut total = 0.0f32;
            for (offset, token) in extra.iter().enumerate() {
                let pos = prompt.len() + offset - 1;
                total += self.logprob(&hidden, pos, *token);
            }
            scores.push(total / extra.len() as f32);
        }
        let best = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let probs = softmax_vec(&scores);
        let mut probabilities = std::collections::BTreeMap::new();
        for (option, prob) in options.iter().zip(probs.iter()) {
            probabilities.insert((*option).to_string(), *prob);
        }
        Ok(ChoiceAnswer {
            choice: options[best].to_string(),
            confidence: probs[best],
            probabilities,
        })
    }

    pub fn noul(&self, image: &RgbImage, state: &str, statement: &str) -> Result<NoulAnswer, Error> {
        let answer = self.choice(image, state, &format!("Is this true: {statement}"), &["yes", "no"])?;
        let yes = *answer.probabilities.get("yes").unwrap_or(&0.0);
        let no = *answer.probabilities.get("no").unwrap_or(&0.0);
        Ok(NoulAnswer {
            answer: yes >= no,
            probability: yes,
            confidence: yes.max(no),
        })
    }

    pub fn score(
        &self,
        image: &RgbImage,
        state: &str,
        instructions: &str,
        levels: &[&str],
    ) -> Result<ScoreAnswer, Error> {
        let answer = self.choice(image, state, instructions, levels)?;
        let mut ordered = Vec::with_capacity(levels.len());
        let mut expected = 0.0f32;
        for (index, level) in levels.iter().enumerate() {
            let prob = *answer.probabilities.get(*level).unwrap_or(&0.0);
            expected += index as f32 * prob;
            ordered.push(((*level).to_string(), prob));
        }
        let last = levels.len().saturating_sub(1);
        let nearest = (expected.round() as usize).min(last);
        Ok(ScoreAnswer {
            score: expected,
            normalized: if last > 0 { expected / last as f32 } else { 0.0 },
            level: levels[nearest].to_string(),
            confidence: ordered[nearest].1,
            probabilities: ordered.into_iter().collect(),
        })
    }

    fn prompt_ids(&self, state: &str, instructions: &str, n_image: usize) -> Result<Vec<u32>, Error> {
        let mut ids = vec![self.im_start];
        ids.extend(self.encode("user\n"));
        ids.push(VISION_START);
        ids.extend(std::iter::repeat(IMAGE_PAD).take(n_image));
        ids.push(VISION_END);
        let body = if state.is_empty() {
            instructions.to_string()
        } else {
            format!("{state}\n{instructions}")
        };
        ids.extend(self.encode(&body));
        ids.push(self.im_end);
        ids.extend(self.encode("\n"));
        ids.push(self.im_start);
        ids.extend(self.encode("assistant\n"));
        Ok(ids)
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer
            .encode(text, false)
            .map(|enc| enc.get_ids().to_vec())
            .unwrap_or_default()
    }

    fn see(&self, pixels: &[f32], grid_h: usize, grid_w: usize) -> (Tensor<Wgpu, 2>, Vec<Tensor<Wgpu, 2>>) {
        let n = pixels.len() / (3 * 2 * PATCH * PATCH);
        let flat = Tensor::<Wgpu, 2>::from_data(
            TensorData::new(pixels.to_vec(), [n, 3 * 2 * PATCH * PATCH]),
            &self.device,
        );
        let mut hidden = linear2(&flat, &self.patch_weight, Some(&self.patch_bias));
        hidden = hidden + self.position_embed(grid_h, grid_w);
        let (cos, sin) = vision_rope(grid_h, grid_w, &self.device);
        let mut deep = Vec::new();
        for (index, layer) in self.vision_layers.iter().enumerate() {
            hidden = vision_block(hidden, layer, &cos, &sin);
            if let Some(slot) = DEEPSTACK.iter().position(|layer_index| *layer_index == index) {
                deep.push(merge(&hidden, &self.deepstack[slot]));
            }
        }
        (merge(&hidden, &self.merger), deep)
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

    fn prefill(
        &self,
        ids: &[u32],
        image_at: usize,
        n_image: usize,
        grid_h: usize,
        grid_w: usize,
        image_rows: &Tensor<Wgpu, 2>,
        deep_rows: &[Tensor<Wgpu, 2>],
    ) -> Tensor<Wgpu, 2> {
        let id_tensor = Tensor::<Wgpu, 1, burn::tensor::Int>::from_data(
            TensorData::new(ids.iter().map(|id| *id as i32).collect(), [ids.len()]),
            &self.device,
        );
        let mut hidden = self.embed.clone().select(0, id_tensor);
        let before = hidden.clone().narrow(0, 0, image_at);
        let after = hidden.clone().narrow(0, image_at + n_image, ids.len() - image_at - n_image);
        hidden = Tensor::cat(vec![before, image_rows.clone(), after], 0);
        let (cos, sin) = text_mrope(ids, image_at, n_image, grid_h, grid_w, &self.device);
        let mask = causal_mask(ids.len(), &self.device);
        for (index, layer) in self.text_layers.iter().enumerate() {
            let normed = rms(hidden.clone(), &layer.input_norm);
            hidden = hidden.clone() + text_attn(&normed, layer, &cos, &sin, &mask);
            let normed = rms(hidden.clone(), &layer.post_norm);
            hidden = hidden + text_mlp(&normed, layer);
            if index < deep_rows.len() {
                let before = hidden.clone().narrow(0, 0, image_at);
                let mid = hidden.clone().narrow(0, image_at, n_image) + deep_rows[index].clone();
                let after = hidden.narrow(0, image_at + n_image, ids.len() - image_at - n_image);
                hidden = Tensor::cat(vec![before, mid, after], 0);
            }
        }
        rms(hidden, &self.text_norm)
    }

    fn logprob(&self, hidden: &Tensor<Wgpu, 2>, position: usize, token: u32) -> f32 {
        let row = hidden.clone().narrow(0, position, 1);
        let logits = row.matmul(self.embed.clone().transpose());
        let data = logits.into_data();
        let values = data.as_slice::<f32>().unwrap();
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for value in values {
            sum += (value - max).exp();
        }
        let log_z = max + sum.ln();
        values[token as usize] - log_z
    }
}

impl Store {
    fn parse(dir: &Path, index: &str) -> Result<Self, Error> {
        let mut map = HashMap::new();
        for line in index.lines() {
            let line = line.trim().trim_end_matches(',');
            if let Some((name, file)) = line.split_once(':') {
                let name = name.trim().trim_matches('"');
                let file = file.trim().trim_matches('"');
                if name.starts_with("model.") || name.starts_with("lm_head") {
                    map.insert(name.to_string(), file.to_string());
                }
            }
        }
        if map.is_empty() {
            return Err(Error::Weights {
                message: "weight index has no tensors".into(),
            });
        }
        let mut left: HashMap<String, usize> = HashMap::new();
        for file in map.values() {
            *left.entry(file.clone()).or_default() += 1;
        }
        Ok(Self {
            files: map
                .values()
                .cloned()
                .map(|file| (file, (dir.to_path_buf(), None)))
                .collect(),
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
        Ok(Tensor::from_data(
            TensorData::new(values, [shape[0], shape[1]]),
            device,
        ))
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

    fn merger(&mut self, prefix: &str, postshuffle: bool, device: &WgpuDevice) -> Result<Merger, Error> {
        Ok(Merger {
            norm_w: self.vector(&format!("{prefix}.norm.weight"), device)?,
            norm_b: self.vector(&format!("{prefix}.norm.bias"), device)?,
            fc1_w: self.matrix(&format!("{prefix}.linear_fc1.weight"), device)?,
            fc1_b: self.vector(&format!("{prefix}.linear_fc1.bias"), device)?,
            fc2_w: self.matrix(&format!("{prefix}.linear_fc2.weight"), device)?,
            fc2_b: self.vector(&format!("{prefix}.linear_fc2.bias"), device)?,
            postshuffle,
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
        let tensors = SafeTensors::deserialize(bytes).map_err(|err| Error::Weights {
            message: err.to_string(),
        })?;
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
    let mut h = (height / FACTOR).max(1) * FACTOR;
    let mut w = (width / FACTOR).max(1) * FACTOR;
    let min_pixels = 64 * 64;
    if h * w < min_pixels {
        let scale = (min_pixels as f32 / (h * w) as f32).sqrt().ceil() as usize;
        h = ((h * scale) / FACTOR).max(1) * FACTOR;
        w = ((w * scale) / FACTOR).max(1) * FACTOR;
    }
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
    let blocks_h = grid_h / MERGE;
    let blocks_w = grid_w / MERGE;
    for block_row in 0..blocks_h {
        for block_col in 0..blocks_w {
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
    let q = take_heads(&qkv, seq, 0);
    let k = take_heads(&qkv, seq, 1);
    let v = take_heads(&qkv, seq, 2);
    let q = rope_vision(q, cos, sin);
    let k = rope_vision(k, cos, sin);
    let scores = q.clone().matmul(k.swap_dims(1, 2)) / (VISION_HEAD_DIM as f32).sqrt();
    let probs = softmax(scores, 2);
    let mixed = probs.matmul(v).swap_dims(0, 1).reshape([seq, VISION_HIDDEN]);
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
        let mut freq_h = vec![0f32; spatial / 2];
        let mut freq_w = vec![0f32; spatial / 2];
        for i in 0..(spatial / 2) {
            let inv = 1.0 / 10000f32.powf((2 * i) as f32 / spatial as f32);
            freq_h[i] = row as f32 * inv;
            freq_w[i] = col as f32 * inv;
        }
        for i in 0..(spatial / 2) {
            let (ch, sh) = (freq_h[i].cos(), freq_h[i].sin());
            let (cw, sw) = (freq_w[i].cos(), freq_w[i].sin());
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
    let wide = if merger.postshuffle {
        let grouped = hidden.clone().reshape([groups, VISION_HIDDEN * MERGE * MERGE]);
        layer_norm(grouped, &merger.norm_w, &merger.norm_b)
    } else {
        layer_norm(hidden.clone(), &merger.norm_w, &merger.norm_b)
            .reshape([groups, VISION_HIDDEN * MERGE * MERGE])
    };
    let mid = gelu_tanh(linear2(&wide, &merger.fc1_w, Some(&merger.fc1_b)));
    linear2(&mid, &merger.fc2_w, Some(&merger.fc2_b))
}

fn text_attn(
    x: &Tensor<Wgpu, 2>,
    layer: &TextLayer,
    cos: &Tensor<Wgpu, 2>,
    sin: &Tensor<Wgpu, 2>,
    mask: &Tensor<Wgpu, 2>,
) -> Tensor<Wgpu, 2> {
    let seq = x.dims()[0];
    let q = heads(&linear2(x, &layer.q, None), seq, TEXT_HEADS);
    let k = heads(&linear2(x, &layer.k, None), seq, TEXT_KV);
    let v = heads(&linear2(x, &layer.v, None), seq, TEXT_KV);
    let q = rope_text(rms_heads(q, &layer.q_norm), cos, sin);
    let k = rope_text(rms_heads(k, &layer.k_norm), cos, sin);
    let k = repeat_kv(k);
    let v = repeat_kv(v);
    let scores = q.clone().matmul(k.swap_dims(1, 2)) / (TEXT_HEAD_DIM as f32).sqrt();
    let probs = softmax(scores + mask.clone().unsqueeze_dim::<3>(0), 2);
    let mixed = probs.matmul(v).swap_dims(0, 1).reshape([seq, TEXT_HEADS * TEXT_HEAD_DIM]);
    linear2(&mixed, &layer.o, None)
}

fn text_mlp(x: &Tensor<Wgpu, 2>, layer: &TextLayer) -> Tensor<Wgpu, 2> {
    let gate = silu(linear2(x, &layer.gate, None));
    let up = linear2(x, &layer.up, None);
    linear2(&(gate * up), &layer.down, None)
}

fn heads(x: &Tensor<Wgpu, 2>, seq: usize, n: usize) -> Tensor<Wgpu, 3> {
    x.clone().reshape([seq, n, TEXT_HEAD_DIM]).swap_dims(0, 1)
}

fn rms_heads(x: Tensor<Wgpu, 3>, weight: &Tensor<Wgpu, 1>) -> Tensor<Wgpu, 3> {
    let variance = x.clone().powf_scalar(2.0).mean_dim(2);
    let normed = x / (variance + 1e-6).sqrt();
    normed * weight.clone().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0)
}

fn repeat_kv(x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
    let seq = x.dims()[1];
    x.unsqueeze_dim::<4>(2)
        .expand([TEXT_KV, seq, TEXT_GROUP, TEXT_HEAD_DIM])
        .swap_dims(1, 2)
        .reshape([TEXT_HEADS, seq, TEXT_HEAD_DIM])
}

fn rope_text(x: Tensor<Wgpu, 3>, cos: &Tensor<Wgpu, 2>, sin: &Tensor<Wgpu, 2>) -> Tensor<Wgpu, 3> {
    let half = TEXT_HEAD_DIM / 2;
    let x1 = x.clone().narrow(2, 0, half);
    let x2 = x.clone().narrow(2, half, half);
    let rotated = Tensor::cat(vec![-x2, x1], 2);
    x * cos.clone().unsqueeze_dim::<3>(0) + rotated * sin.clone().unsqueeze_dim::<3>(0)
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
    for t in 0..1 {
        for h in 0..llm_h {
            for w in 0..llm_w {
                let slot = image_at + t * llm_h * llm_w + h * llm_w + w;
                pos[slot] = [cursor, cursor + h as i32, cursor + w as i32];
            }
        }
    }
    cursor += llm_h.max(llm_w) as i32;
    for index in image_at + n_image..ids.len() {
        pos[index] = [cursor, cursor, cursor];
        cursor += 1;
    }
    let half = TEXT_HEAD_DIM / 2;
    let mut cos = vec![0f32; ids.len() * TEXT_HEAD_DIM];
    let mut sin = vec![0f32; ids.len() * TEXT_HEAD_DIM];
    for (index, axes) in pos.iter().enumerate() {
        let mut freq = [[0f32; 64]; 3];
        for axis in 0..3 {
            for i in 0..half {
                let inv = 1.0 / ROPE_THETA.powf((2 * i) as f32 / TEXT_HEAD_DIM as f32);
                freq[axis][i] = axes[axis] as f32 * inv;
            }
        }
        let mut mixed = freq[0];
        for (axis, offset) in [(1usize, 1usize), (2, 2)] {
            let length = MROPE[axis] * 3;
            let mut slot = offset;
            let mut taken = 0;
            while slot < length && taken < MROPE[axis] {
                mixed[slot] = freq[axis][slot];
                slot += 3;
                taken += 1;
            }
        }
        let base = index * TEXT_HEAD_DIM;
        for i in 0..half {
            let (c, s) = (mixed[i].cos(), mixed[i].sin());
            cos[base + i] = c;
            sin[base + i] = s;
            cos[base + half + i] = c;
            sin[base + half + i] = s;
        }
    }
    (
        Tensor::from_data(TensorData::new(cos, [ids.len(), TEXT_HEAD_DIM]), device),
        Tensor::from_data(TensorData::new(sin, [ids.len(), TEXT_HEAD_DIM]), device),
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

fn rms(x: Tensor<Wgpu, 2>, weight: &Tensor<Wgpu, 1>) -> Tensor<Wgpu, 2> {
    let variance = x.clone().powf_scalar(2.0).mean_dim(1);
    x / (variance + 1e-6).sqrt() * weight.clone().unsqueeze_dim::<2>(0)
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

