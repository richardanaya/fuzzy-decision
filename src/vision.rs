//! SigLIP2 NaFlex vision tower and the LFM2-VL tiling that feeds it.
//!
//! A large image is cut into up to ten 512 px tiles plus a thumbnail. Each crop
//! is 16 px patches, a position embedding resized with antialiased bilinear,
//! twelve pre-norm layers, and a 2x2 pixel-unshuffle projector. Image resize
//! rounds back to bytes, so a pixel can differ by one level from torchvision.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::nn::{attend, gelu_erf, gelu_tanh, tensor2, LayerNorm, Linear};
use crate::resample::{resize_hwc, resize_rgb_u8};
use crate::weights::{TensorSource, VisionSpec};

const FACTOR: usize = 32;
const MAXIMUM: usize = 256 * 1024;
const MINIMUM: usize = 64 * 1024;
pub const TILE: usize = 512;
pub const PATCH: usize = 16;

/// Tightly packed 8-bit RGB.
pub struct RgbImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImagePlan {
    pub grid_w: usize,
    pub grid_h: usize,
    pub thumb_h: usize,
    pub thumb_w: usize,
    pub tiled: bool,
}

pub fn layout(width: usize, height: usize) -> Result<ImagePlan, String> {
    if width < 1 || height < 1 {
        return Err("empty image".into());
    }
    let snap = |side: usize| -> usize {
        let rounded = py_round(side as f64 / FACTOR as f64) * FACTOR as i64;
        rounded.max(FACTOR as i64) as usize
    };
    let raw_h = snap(height);
    let raw_w = snap(width);
    let mut thumb_h = raw_h;
    let mut thumb_w = raw_w;
    if thumb_h * thumb_w > MAXIMUM {
        let beta = ((height * width) as f64 / MAXIMUM as f64).sqrt();
        thumb_h = factor_floor(height as f64 / beta);
        thumb_w = factor_floor(width as f64 / beta);
    } else if thumb_h * thumb_w < MINIMUM {
        let beta = (MINIMUM as f64 / (height * width) as f64).sqrt();
        thumb_h = factor_ceil(height as f64 * beta);
        thumb_w = factor_ceil(width as f64 * beta);
    }
    let large = raw_h * raw_w > MAXIMUM * 2;
    let (grid_w, grid_h) = if large {
        tile_grid(width, height)
    } else {
        (1, 1)
    };
    Ok(ImagePlan {
        grid_w,
        grid_h,
        thumb_h,
        thumb_w,
        tiled: large,
    })
}

fn factor_floor(value: f64) -> usize {
    ((value / FACTOR as f64).floor() as usize).max(1) * FACTOR
}

fn factor_ceil(value: f64) -> usize {
    (value / FACTOR as f64).ceil() as usize * FACTOR
}

fn tile_grid(width: usize, height: usize) -> (usize, usize) {
    let mut ratios = Vec::new();
    for n in 2..=10 {
        for x in 1..=n {
            for y in 1..=n {
                let cells = x * y;
                if (2..=10).contains(&cells) {
                    ratios.push((x, y));
                }
            }
        }
    }
    ratios.sort_by_key(|(x, y)| x * y);
    ratios.dedup();
    let mut best = f64::INFINITY;
    let mut grid = (1, 1);
    for (x, y) in ratios {
        let diff = (width as f64 / height as f64 - x as f64 / y as f64).abs();
        let roomy = (width * height) as f64 > 0.5 * (TILE * TILE * x * y) as f64;
        if diff < best || (diff == best && roomy) {
            grid = (x, y);
            best = diff;
        }
    }
    grid
}

/// Python 3 `round`: halves go to the even integer.
fn py_round(value: f64) -> i64 {
    let floor = value.floor();
    let diff = value - floor;
    if (diff - 0.5).abs() < 1e-9 {
        let down = floor as i64;
        if down % 2 == 0 {
            down
        } else {
            down + 1
        }
    } else if diff > 0.5 {
        floor as i64 + 1
    } else {
        floor as i64
    }
}

pub struct Crop {
    pub patches: Vec<f32>,
    pub height: usize,
    pub width: usize,
    pub patch_dim: usize,
}

pub fn image_crops(image: &RgbImage) -> Result<Vec<Crop>, String> {
    let width = image.width as usize;
    let height = image.height as usize;
    if image.data.len() != width * height * 3 {
        return Err("RGB image byte length does not match its size".into());
    }
    let plan = layout(width, height)?;
    let mut crops = Vec::new();
    if plan.tiled {
        let big_w = plan.grid_w * TILE;
        let big_h = plan.grid_h * TILE;
        let big = resize_rgb_u8(&image.data, width, height, big_w, big_h);
        for row in 0..plan.grid_h {
            for col in 0..plan.grid_w {
                let tile = crop_rgb(&big, big_w, col * TILE, row * TILE, TILE, TILE);
                crops.push(patchify(&tile, TILE, TILE));
            }
        }
    }
    let thumb = if plan.thumb_w == width && plan.thumb_h == height {
        image.data.clone()
    } else {
        resize_rgb_u8(&image.data, width, height, plan.thumb_w, plan.thumb_h)
    };
    crops.push(patchify(&thumb, plan.thumb_w, plan.thumb_h));
    Ok(crops)
}

fn crop_rgb(
    src: &[u8],
    stride: usize,
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        let start = ((y0 + y) * stride + x0) * 3;
        out.extend_from_slice(&src[start..start + width * 3]);
    }
    out
}

fn patchify(rgb: &[u8], width: usize, height: usize) -> Crop {
    let ph = height / PATCH;
    let pw = width / PATCH;
    let patch_dim = 3 * PATCH * PATCH;
    let mut patches = Vec::with_capacity(ph * pw * patch_dim);
    for py in 0..ph {
        for px in 0..pw {
            for dy in 0..PATCH {
                for dx in 0..PATCH {
                    let y = py * PATCH + dy;
                    let x = px * PATCH + dx;
                    for channel in 0..3 {
                        let pixel = rgb[(y * width + x) * 3 + channel] as f32;
                        patches.push((pixel - 127.5) / 127.5);
                    }
                }
            }
        }
    }
    Crop {
        patches,
        height: ph,
        width: pw,
        patch_dim,
    }
}

pub fn image_stamp(image: &RgbImage) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in image
        .width
        .to_le_bytes()
        .into_iter()
        .chain(image.height.to_le_bytes())
        .chain(image.data.iter().copied())
    {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

pub struct VisionTower<B: Backend> {
    patch: Linear<B>,
    positions: Vec<f32>,
    grid: usize,
    layers: Vec<VisionLayer<B>>,
    post: LayerNorm<B>,
    proj_1: Linear<B>,
    proj_2: Linear<B>,
    hidden: usize,
    device: B::Device,
}

impl<B: Backend> VisionTower<B> {
    pub fn load(
        source: &impl TensorSource,
        spec: &VisionSpec,
        device: &B::Device,
    ) -> Result<Self, String> {
        let root = "vision.tower.vision_model";
        let (positions, shape) =
            source.tensor(&format!("{root}.embeddings.position_embedding.weight"))?;
        if shape.len() != 2 {
            return Err(format!("position embedding has shape {shape:?}"));
        }
        let grid = (spec.num_patches as f64).sqrt() as usize;
        let mut layers = Vec::with_capacity(spec.layers);
        for index in 0..spec.layers {
            layers.push(VisionLayer::load(
                source,
                &format!("{root}.encoder.layers.{index}"),
                spec,
                device,
            )?);
        }
        Ok(Self {
            patch: {
                let patch = Linear::load(
                    source,
                    &format!("{root}.embeddings.patch_embedding"),
                    device,
                    true,
                )?;
                let expect = 3 * spec.patch * spec.patch;
                if patch.weight.dims()[1] != expect {
                    return Err(format!(
                        "patch embedding takes {}, expected {expect}",
                        patch.weight.dims()[1]
                    ));
                }
                patch
            },
            positions,
            grid,
            layers,
            post: LayerNorm::load(source, &format!("{root}.post_layernorm"), spec.eps, device)?,
            proj_1: {
                let proj = Linear::load(source, "vision.projector.linear_1", device, true)?;
                if proj.weight.dims()[0] != spec.projector_hidden {
                    return Err(format!(
                        "projector hidden is {}, expected {}",
                        proj.weight.dims()[0],
                        spec.projector_hidden
                    ));
                }
                proj
            },
            proj_2: Linear::load(source, "vision.projector.linear_2", device, true)?,
            hidden: spec.hidden,
            device: device.clone(),
        })
    }

    pub fn forward_crop(
        &self,
        patches: &[f32],
        ph: usize,
        pw: usize,
        patch_dim: usize,
    ) -> Tensor<B, 2> {
        let n = ph * pw;
        let mut hidden = self
            .patch
            .forward(tensor2(patches.to_vec(), n, patch_dim, &self.device));
        let pos = resize_hwc(&self.positions, self.grid, self.grid, self.hidden, ph, pw);
        hidden = hidden + tensor2(pos, n, self.hidden, &self.device);
        for layer in &self.layers {
            hidden = layer.forward(hidden);
        }
        hidden = self.post.forward(hidden);
        self.project(hidden, ph, pw)
    }

    fn project(&self, hidden: Tensor<B, 2>, ph: usize, pw: usize) -> Tensor<B, 2> {
        let f = 2;
        let channels = hidden.dims()[1];
        let x = hidden.reshape([ph, pw / f, channels * f]).swap_dims(0, 1);
        let x = x
            .reshape([pw / f, ph / f, channels * f * f])
            .swap_dims(0, 1);
        let x = x.reshape([(ph / f) * (pw / f), channels * f * f]);
        self.proj_2.forward(gelu_erf(self.proj_1.forward(x)))
    }
}

struct VisionLayer<B: Backend> {
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    heads: usize,
    head_dim: usize,
}

impl<B: Backend> VisionLayer<B> {
    fn load(
        source: &impl TensorSource,
        prefix: &str,
        spec: &VisionSpec,
        device: &B::Device,
    ) -> Result<Self, String> {
        Ok(Self {
            norm1: LayerNorm::load(source, &format!("{prefix}.layer_norm1"), spec.eps, device)?,
            norm2: LayerNorm::load(source, &format!("{prefix}.layer_norm2"), spec.eps, device)?,
            q: Linear::load(source, &format!("{prefix}.self_attn.q_proj"), device, true)?,
            k: Linear::load(source, &format!("{prefix}.self_attn.k_proj"), device, true)?,
            v: Linear::load(source, &format!("{prefix}.self_attn.v_proj"), device, true)?,
            out: Linear::load(
                source,
                &format!("{prefix}.self_attn.out_proj"),
                device,
                true,
            )?,
            fc1: {
                let fc1 = Linear::load(source, &format!("{prefix}.mlp.fc1"), device, true)?;
                if fc1.weight.dims()[0] != spec.intermediate {
                    return Err(format!(
                        "{prefix} mlp width is {}, expected {}",
                        fc1.weight.dims()[0],
                        spec.intermediate
                    ));
                }
                fc1
            },
            fc2: Linear::load(source, &format!("{prefix}.mlp.fc2"), device, true)?,
            heads: spec.heads,
            head_dim: spec.hidden / spec.heads,
        })
    }

    fn forward(&self, hidden: Tensor<B, 2>) -> Tensor<B, 2> {
        let seq = hidden.dims()[0];
        let width = hidden.dims()[1];
        let normed = self.norm1.forward(hidden.clone());
        let q = self
            .q
            .forward(normed.clone())
            .reshape([seq, self.heads, self.head_dim])
            .swap_dims(0, 1);
        let k = self
            .k
            .forward(normed.clone())
            .reshape([seq, self.heads, self.head_dim])
            .swap_dims(0, 1);
        let v = self
            .v
            .forward(normed)
            .reshape([seq, self.heads, self.head_dim])
            .swap_dims(0, 1);
        let y = attend(q, k, v, (self.head_dim as f32).powf(-0.5));
        let hidden = hidden + self.out.forward(y.swap_dims(0, 1).reshape([seq, width]));
        let fed = self.fc2.forward(gelu_tanh(
            self.fc1.forward(self.norm2.forward(hidden.clone())),
        ));
        hidden + fed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_matches_the_reference() {
        let plan = layout(320, 320).unwrap();
        assert!(!plan.tiled);
        assert_eq!((plan.thumb_w, plan.thumb_h), (320, 320));
        let small = layout(100, 100).unwrap();
        assert_eq!((small.thumb_w, small.thumb_h), (256, 256));
        let big = layout(4000, 3000).unwrap();
        assert!(big.tiled);
        assert_eq!((big.grid_w, big.grid_h), (3, 2));
        assert_eq!((big.thumb_w, big.thumb_h), (576, 416));
        let photo = layout(640, 480).unwrap();
        assert!(!photo.tiled);
        assert_eq!((photo.thumb_w, photo.thumb_h), (576, 416));
        assert_eq!(layout(512, 512).unwrap().thumb_w, 512);
        let wide = layout(800, 200).unwrap();
        assert_eq!((wide.thumb_w, wide.thumb_h), (800, 192));
        assert_eq!(layout(64, 64).unwrap().thumb_h, 256);
    }
}
