//! One mmap'd `model.safetensors` plus `config.json` for d1-omni-600M.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;
use serde_json::Value;

pub const REQUIRED_FILES: [&str; 3] = ["tokenizer.json", "config.json", "model.safetensors"];

pub fn default_weights_dir() -> PathBuf {
    PathBuf::from("models/d1-omni-600M")
}

pub fn weights_ready(dir: &Path) -> bool {
    REQUIRED_FILES.iter().all(|name| dir.join(name).is_file())
}

pub fn to_f32(view: &TensorView<'_>) -> Result<Vec<f32>, String> {
    let bytes = view.data();
    match view.dtype() {
        safetensors::Dtype::F32 => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(f32::from_le_bytes)
            .collect()),
        safetensors::Dtype::BF16 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .copied()
            .map(u16::from_le_bytes)
            .map(bf16_to_f32)
            .collect()),
        safetensors::Dtype::F16 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .copied()
            .map(u16::from_le_bytes)
            .map(half_to_f32)
            .collect()),
        other => Err(format!("unsupported dtype {other:?}")),
    }
}

fn bf16_to_f32(half: u16) -> f32 {
    f32::from_bits((half as u32) << 16)
}

fn half_to_f32(half: u16) -> f32 {
    let sign = ((half >> 15) & 1) as u32;
    let exp = ((half >> 10) & 0x1f) as u32;
    let frac = (half & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let mut mantissa = frac;
            let mut exponent = 127 - 14;
            while mantissa & 0x400 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            mantissa &= 0x3ff;
            (sign << 31) | (exponent << 23) | (mantissa << 13)
        }
    } else if exp == 31 {
        (sign << 31) | 0x7f80_0000 | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

pub trait TensorSource {
    fn tensor(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String>;
}

pub struct Snapshot {
    _file: File,
    map: Mmap,
}

impl Snapshot {
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
        // The file stays open for the life of the map.
        let map =
            unsafe { Mmap::map(&file) }.map_err(|err| format!("mmap {}: {err}", path.display()))?;
        SafeTensors::deserialize(&map)
            .map_err(|err| format!("safetensors {}: {err}", path.display()))?;
        Ok(Self { _file: file, map })
    }
}

impl TensorSource for Snapshot {
    fn tensor(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        let tensors = SafeTensors::deserialize(&self.map).map_err(|err| err.to_string())?;
        let view = tensors
            .tensor(name)
            .map_err(|err| format!("{name}: {err}"))?;
        let shape = view.shape().to_vec();
        let values = to_f32(&view)?;
        Ok((values, shape))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Conv,
    Attention,
}

#[derive(Debug, Clone)]
pub struct TrunkSpec {
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub layers: Vec<LayerKind>,
    pub eps: f32,
    pub multiple_of: usize,
    pub ffn_multiplier: f32,
    pub rope_theta: f32,
}

impl TrunkSpec {
    /// `int(2 * intermediate / 3)`, then the LFM multiple-of rounding.
    pub fn ffn_dim(&self) -> usize {
        let hidden = (2.0 * self.intermediate as f64 / 3.0) as usize;
        let hidden = (self.ffn_multiplier as f64 * hidden as f64) as usize;
        self.multiple_of * hidden.div_ceil(self.multiple_of)
    }
}

#[derive(Debug, Clone)]
pub struct AudioSpec {
    pub feat_in: usize,
    pub layers: usize,
    pub d_model: usize,
    pub channels: usize,
    pub ff_expansion: usize,
    pub heads: usize,
    pub kernel: usize,
    pub residual_width: usize,
}

#[derive(Debug, Clone)]
pub struct VisionSpec {
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub layers: usize,
    pub eps: f32,
    pub patch: usize,
    pub num_patches: usize,
    pub projector_hidden: usize,
}

#[derive(Debug, Clone)]
pub struct D1Config {
    pub max_length: usize,
    pub image_text_length: usize,
    pub audio_text_length: usize,
    pub head_layers: usize,
    pub temperatures: BTreeMap<String, f32>,
    pub text: TrunkSpec,
    pub audio: AudioSpec,
    pub vision: VisionSpec,
}

impl D1Config {
    pub fn open(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|err| format!("read {}: {err}", path.display()))?;
        let value: Value = serde_json::from_str(&text)
            .map_err(|err| format!("parse {}: {err}", path.display()))?;
        let text_config = value
            .get("text_config")
            .ok_or("config is missing text_config")?;
        let audio_config = value
            .get("audio_config")
            .ok_or("config is missing audio_config")?;
        let vision_config = value
            .get("vision_config")
            .ok_or("config is missing vision_config")?;
        let mut layers = Vec::new();
        for kind in text_config
            .get("layer_types")
            .and_then(Value::as_array)
            .ok_or("config is missing layer_types")?
        {
            match kind.as_str() {
                Some("conv") => layers.push(LayerKind::Conv),
                Some("full_attention") => layers.push(LayerKind::Attention),
                other => return Err(format!("unknown layer type {other:?}")),
            }
        }
        let mut temperatures = BTreeMap::new();
        if let Some(table) = value.get("temperatures").and_then(Value::as_object) {
            for (key, entry) in table {
                let number = entry
                    .as_f64()
                    .ok_or_else(|| format!("temperature {key} is not a number"))?;
                temperatures.insert(key.clone(), number as f32);
            }
        }
        Ok(Self {
            max_length: json_usize(&value, "max_length")?,
            image_text_length: json_usize(&value, "image_text_length")?,
            audio_text_length: json_usize(&value, "audio_text_length")?,
            head_layers: json_usize(&value, "head_layers")?,
            temperatures,
            text: TrunkSpec {
                hidden: json_usize(text_config, "hidden_size")?,
                intermediate: json_usize(text_config, "intermediate_size")?,
                heads: json_usize(text_config, "num_attention_heads")?,
                kv_heads: json_usize(text_config, "num_key_value_heads")?,
                layers,
                eps: json_f32(text_config, "norm_eps")?,
                multiple_of: json_usize(text_config, "block_multiple_of")?,
                ffn_multiplier: json_f32(text_config, "block_ffn_dim_multiplier")?,
                rope_theta: json_f32(text_config, "rope_theta")?,
            },
            audio: AudioSpec {
                feat_in: json_usize(audio_config, "feat_in")?,
                layers: json_usize(audio_config, "n_layers")?,
                d_model: json_usize(audio_config, "d_model")?,
                channels: json_usize(audio_config, "subsampling_conv_channels")?,
                ff_expansion: json_usize(audio_config, "ff_expansion_factor")?,
                heads: json_usize(audio_config, "n_heads")?,
                kernel: json_usize(audio_config, "conv_kernel_size")?,
                residual_width: json_usize(audio_config, "residual_width")?,
            },
            vision: VisionSpec {
                hidden: json_usize(vision_config, "hidden_size")?,
                intermediate: json_usize(vision_config, "intermediate_size")?,
                heads: json_usize(vision_config, "num_attention_heads")?,
                layers: json_usize(vision_config, "num_hidden_layers")?,
                eps: json_f32(vision_config, "layer_norm_eps")?,
                patch: json_usize(vision_config, "patch_size")?,
                num_patches: json_usize(vision_config, "num_patches")?,
                projector_hidden: json_usize(&value, "projector_hidden_size")?,
            },
        })
    }

    pub fn temperature(&self, kind: &str, options: usize) -> f32 {
        let bucket = if options <= 2 {
            "2"
        } else if options <= 5 {
            "3-5"
        } else if options <= 10 {
            "6-10"
        } else {
            "11+"
        };
        let key = format!("{kind}:{bucket}");
        self.temperatures
            .get(&key)
            .or_else(|| self.temperatures.get(kind))
            .copied()
            .unwrap_or(1.0)
    }
}

fn json_usize(value: &Value, key: &str) -> Result<usize, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|number| number as usize)
        .ok_or_else(|| format!("config is missing {key}"))
}

fn json_f32(value: &Value, key: &str) -> Result<f32, String> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .map(|number| number as f32)
        .ok_or_else(|| format!("config is missing {key}"))
}

/// Token rows kept on the CPU. A row is copied to the device when it is used.
pub struct EmbedTable {
    data: Vec<f32>,
    pub dim: usize,
}

impl EmbedTable {
    pub fn load(source: &impl TensorSource, name: &str) -> Result<Self, String> {
        let (data, shape) = source.tensor(name)?;
        if shape.len() != 2 {
            return Err(format!("{name} has shape {shape:?}"));
        }
        Ok(Self {
            data,
            dim: shape[1],
        })
    }

    pub fn gather(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let mut out = vec![0.0; ids.len() * self.dim];
        let rows = self.data.len() / self.dim;
        for (index, id) in ids.iter().enumerate() {
            let row = *id as usize;
            if row >= rows {
                return Err(format!("token id {id} is outside the embedding table"));
            }
            let start = row * self.dim;
            out[index * self.dim..(index + 1) * self.dim]
                .copy_from_slice(&self.data[start..start + self.dim]);
        }
        Ok(out)
    }
}
