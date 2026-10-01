//! Vision mode: Clef-Flash with an image in the state.
//!
//! The caller places a local `Cloudflare/clef-flash` snapshot in a directory.
//! This module does not download it. An image plus text is scored by the same
//! joint schema head as the text mode; answers have the same shapes and the
//! same probability semantics.

use std::cell::RefCell;
use std::path::Path;

use crate::answers::{decode_answer, Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer};
use crate::backbone::{block_coords, ImageSpan, MERGE, PATCH};
use crate::clef::{Clef, Prefix};
use crate::questions::{choice, noul, score, validate_question, Question};
use crate::record::encode_record;
use crate::Error;

const FACTOR: usize = PATCH * MERGE;
/// The checkpoint's processor allows much larger images; this crate caps the
/// resized image so one picture stays near a thousand language-model tokens.
const MIN_PIXELS: usize = 65_536;
const MAX_PIXELS: usize = 1_048_576;

/// Tightly packed 8-bit RGB.
pub struct RgbImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

struct CachedPrefix {
    stamp: u64,
    prefix: Prefix,
}

pub struct VisionDecision {
    model: Clef,
    prefix: RefCell<Option<CachedPrefix>>,
}

impl VisionDecision {
    /// Load Clef-Flash with its vision tower from `dir`.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, Error> {
        Ok(Self {
            model: Clef::load(dir.as_ref(), true)?,
            prefix: RefCell::new(None),
        })
    }

    pub fn choice(
        &self,
        image: &RgbImage,
        state: &str,
        instructions: &str,
        options: &[&str],
    ) -> Result<ChoiceAnswer, Error> {
        match self.one(image, state, choice(instructions, options, None))? {
            Answer::Choice {
                choice,
                confidence,
                probabilities,
            } => Ok(ChoiceAnswer {
                choice,
                confidence,
                probabilities,
            }),
            _ => unreachable!("choice decodes to a choice answer"),
        }
    }

    pub fn noul(&self, image: &RgbImage, state: &str, statement: &str) -> Result<NoulAnswer, Error> {
        match self.one(image, state, noul(statement))? {
            Answer::Noul {
                answer,
                probability,
                confidence,
            } => Ok(NoulAnswer {
                answer,
                probability,
                confidence,
            }),
            _ => unreachable!("noul decodes to a noul answer"),
        }
    }

    pub fn score(
        &self,
        image: &RgbImage,
        state: &str,
        instructions: &str,
        levels: &[&str],
    ) -> Result<ScoreAnswer, Error> {
        match self.one(image, state, score(instructions, levels))? {
            Answer::Score {
                score,
                normalized,
                level,
                confidence,
                probabilities,
            } => Ok(ScoreAnswer {
                score,
                normalized,
                level,
                confidence,
                probabilities,
            }),
            _ => unreachable!("score decodes to a score answer"),
        }
    }

    fn one(&self, image: &RgbImage, state: &str, question: Question) -> Result<Answer, Error> {
        validate_question(&question, "vision", crate::limits())?;
        let (pixels, grid_h, grid_w) = patchify(image).map_err(|message| Error::Weights { message })?;
        let llm_h = grid_h / MERGE;
        let llm_w = grid_w / MERGE;
        let media_ids = self.model.tokenizer.media_ids(llm_h * llm_w);
        let layout = encode_record(
            &self.model.tokenizer,
            state,
            Some(&media_ids),
            std::slice::from_ref(&question),
            &["q1".to_string()],
            crate::DEFAULT_MAX_LENGTH,
            crate::DEFAULT_MAX_STATE,
            false,
        )?;
        let span = ImageSpan {
            at: layout.image_at.expect("media ids were provided"),
            llm_h,
            llm_w,
        };

        let stamp = image_stamp(image, state);
        let cached = {
            let slot = self.prefix.borrow();
            slot.as_ref()
                .map(|cached| cached.stamp == stamp && cached.prefix.len == layout.prefix_len)
                .unwrap_or(false)
        };
        let logits = if cached {
            let slot = self.prefix.borrow();
            let cached = slot.as_ref().expect("cache checked above");
            let (logits, _) = self
                .model
                .score(&layout, Some(span), None, Some(&cached.prefix), false)?;
            logits
        } else {
            let rows = self
                .model
                .backbone()
                .see(&pixels, grid_h, grid_w)
                .map_err(|message| Error::Weights { message })?;
            let (logits, prefix) = self.model.score(&layout, Some(span), Some(rows), None, true)?;
            *self.prefix.borrow_mut() = prefix.map(|prefix| CachedPrefix { stamp, prefix });
            logits
        };
        Ok(decode_answer(&question, &logits[0], 1.0))
    }
}

/// Resize with the processor's smart-resize rule and cut the image into
/// `16x16` patches (temporal size two, both frames equal) in merge order.
fn patchify(image: &RgbImage) -> Result<(Vec<f32>, usize, usize), String> {
    let width = image.width as usize;
    let height = image.height as usize;
    if image.data.len() != width * height * 3 {
        return Err("RGB image byte length does not match its size".into());
    }
    if width == 0 || height == 0 {
        return Err("image has a zero side".into());
    }
    let (new_height, new_width) = smart_size(height, width);
    let rgb = if new_height == height && new_width == width {
        image.data.clone()
    } else {
        resize_bilinear(&image.data, width, height, new_width, new_height)
    };
    let grid_h = new_height / PATCH;
    let grid_w = new_width / PATCH;
    let mut out = Vec::with_capacity(grid_h * grid_w * 3 * 2 * PATCH * PATCH);
    for (row, col) in block_coords(grid_h, grid_w) {
        for channel in 0..3 {
            for _time in 0..2 {
                for py in 0..PATCH {
                    for px in 0..PATCH {
                        let y = row * PATCH + py;
                        let x = col * PATCH + px;
                        let pixel = rgb[(y * new_width + x) * 3 + channel] as f32 / 255.0;
                        out.push(pixel * 2.0 - 1.0);
                    }
                }
            }
        }
    }
    Ok((out, grid_h, grid_w))
}

/// The Qwen image processor's `smart_resize`: sides snap to multiples of 32
/// and the pixel count is pushed inside `[MIN_PIXELS, MAX_PIXELS]`.
fn smart_size(height: usize, width: usize) -> (usize, usize) {
    let round = |value: f64| -> usize {
        ((value / FACTOR as f64).round() as usize).max(1) * FACTOR
    };
    let mut h = round(height as f64);
    let mut w = round(width as f64);
    if h * w > MAX_PIXELS {
        let beta = ((height * width) as f64 / MAX_PIXELS as f64).sqrt();
        h = (((height as f64 / beta) / FACTOR as f64).floor() as usize).max(1) * FACTOR;
        w = (((width as f64 / beta) / FACTOR as f64).floor() as usize).max(1) * FACTOR;
    } else if h * w < MIN_PIXELS {
        let beta = (MIN_PIXELS as f64 / (height * width) as f64).sqrt();
        h = ((height as f64 * beta / FACTOR as f64).ceil() as usize).max(1) * FACTOR;
        w = ((width as f64 * beta / FACTOR as f64).ceil() as usize).max(1) * FACTOR;
    }
    (h, w)
}

fn resize_bilinear(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let mut out = vec![0u8; dw * dh * 3];
    for y in 0..dh {
        let sy = ((y as f32 + 0.5) * sh as f32 / dh as f32 - 0.5).max(0.0);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let wy = sy - y0 as f32;
        for x in 0..dw {
            let sx = ((x as f32 + 0.5) * sw as f32 / dw as f32 - 0.5).max(0.0);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let wx = sx - x0 as f32;
            for channel in 0..3 {
                let p00 = src[(y0 * sw + x0) * 3 + channel] as f32;
                let p01 = src[(y0 * sw + x1) * 3 + channel] as f32;
                let p10 = src[(y1 * sw + x0) * 3 + channel] as f32;
                let p11 = src[(y1 * sw + x1) * 3 + channel] as f32;
                let top = p00 * (1.0 - wx) + p01 * wx;
                let bottom = p10 * (1.0 - wx) + p11 * wx;
                out[(y * dw + x) * 3 + channel] = (top * (1.0 - wy) + bottom * wy).round() as u8;
            }
        }
    }
    out
}

fn image_stamp(image: &RgbImage, state: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in image
        .width
        .to_le_bytes()
        .into_iter()
        .chain(image.height.to_le_bytes())
        .chain(image.data.iter().copied())
        .chain(state.as_bytes().iter().copied())
    {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_size_snaps_to_factor() {
        let (h, w) = smart_size(500, 700);
        assert_eq!(h % FACTOR, 0);
        assert_eq!(w % FACTOR, 0);
        assert!(h * w <= MAX_PIXELS && h * w >= MIN_PIXELS);
    }

    #[test]
    fn smart_size_grows_small_images() {
        let (h, w) = smart_size(100, 100);
        assert!(h * w >= MIN_PIXELS);
    }

    #[test]
    fn smart_size_shrinks_large_images() {
        let (h, w) = smart_size(4000, 4000);
        assert!(h * w <= MAX_PIXELS);
    }
}
