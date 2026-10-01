//! Reading the Clef-Flash snapshot: sharded safetensors, the joint head file,
//! and the small JSON head config.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use safetensors::tensor::TensorView;
use safetensors::SafeTensors;

pub fn default_weights_dir() -> PathBuf {
    PathBuf::from("models/clef-flash")
}

pub fn to_f32(view: &TensorView<'_>) -> Result<Vec<f32>, String> {
    let bytes = view.data();
    match view.dtype() {
        safetensors::Dtype::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()),
        safetensors::Dtype::BF16 => Ok(bytes
            .chunks_exact(2)
            .map(|chunk| bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
            .collect()),
        safetensors::Dtype::F16 => Ok(bytes
            .chunks_exact(2)
            .map(|chunk| half_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
            .collect()),
        other => Err(format!("unsupported dtype {other:?}")),
    }
}

/// The raw rows of a large bf16 table, kept on the CPU. Rows are converted to
/// f32 as they are gathered, so the token tables stay at half their f32 size.
pub struct Bf16Table {
    data: Vec<u16>,
    pub rows: usize,
    pub dim: usize,
}

impl Bf16Table {
    pub fn from_view(view: &TensorView<'_>) -> Result<Self, String> {
        let shape = view.shape().to_vec();
        if shape.len() != 2 {
            return Err(format!("token table has shape {shape:?}"));
        }
        let bytes = view.data();
        let data: Vec<u16> = match view.dtype() {
            safetensors::Dtype::BF16 => bytes
                .chunks_exact(2)
                .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                .collect(),
            safetensors::Dtype::F32 => bytes
                .chunks_exact(4)
                .map(|chunk| {
                    let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    (value.to_bits() >> 16) as u16
                })
                .collect(),
            other => return Err(format!("unsupported table dtype {other:?}")),
        };
        Ok(Self {
            data,
            rows: shape[0],
            dim: shape[1],
        })
    }

    pub fn write_row(&self, row: usize, out: &mut [f32]) {
        let src = &self.data[row * self.dim..(row + 1) * self.dim];
        for (slot, half) in out.iter_mut().zip(src) {
            *slot = bf16_to_f32(*half);
        }
    }

    /// The mean of the given rows.
    pub fn mean_rows(&self, rows: &[u32]) -> Vec<f32> {
        let mut out = vec![0f32; self.dim];
        for row in rows {
            let src = &self.data[*row as usize * self.dim..(*row as usize + 1) * self.dim];
            for (slot, half) in out.iter_mut().zip(src) {
                *slot += bf16_to_f32(*half);
            }
        }
        let inv = 1.0 / rows.len().max(1) as f32;
        for slot in &mut out {
            *slot *= inv;
        }
        out
    }
}

pub fn bf16_to_f32(half: u16) -> f32 {
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
            let mut exponent = 127 - 15 + 1;
            while mantissa & 0x400 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            mantissa &= 0x3ff;
            (sign << 31) | (exponent << 23) | (mantissa << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

/// Reads tensors across the sharded `model-*.safetensors` files, keeping each
/// shard's bytes only while tensors remain to be read from it.
pub struct Store {
    files: HashMap<String, (PathBuf, Option<Vec<u8>>)>,
    map: HashMap<String, String>,
    left: HashMap<String, usize>,
}

impl Store {
    pub fn parse(dir: &Path, index: &str) -> Result<Self, String> {
        let mut map = HashMap::new();
        for line in index.lines() {
            let line = line.trim().trim_end_matches(',');
            if let Some((name, file)) = line.split_once(':') {
                let name = name.trim().trim_matches('"');
                let file = file.trim().trim_matches('"');
                if name.starts_with("model.") || name == "lm_head.weight" {
                    map.insert(name.to_string(), file.to_string());
                }
            }
        }
        if map.is_empty() {
            return Err("weight index has no tensors".into());
        }
        let mut left = HashMap::new();
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

    /// Marks tensors that will never be read (for example the vision tower in
    /// text mode) so their shards can be dropped once other reads finish.
    pub fn skip_prefix(&mut self, prefix: &str) {
        let names: Vec<String> = self
            .map
            .keys()
            .filter(|name| name.starts_with(prefix))
            .cloned()
            .collect();
        for name in names {
            let file = self.map.remove(&name).expect("name was present");
            let remaining = self.left.get_mut(&file).expect("shard count");
            *remaining -= 1;
            if *remaining == 0 {
                self.files.get_mut(&file).expect("shard").1 = None;
            }
        }
    }

    pub fn read(&mut self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        self.with_view(name, |view| {
            let shape = view.shape().to_vec();
            let values = to_f32(view)?;
            Ok((values, shape))
        })
    }

    pub fn table(&mut self, name: &str) -> Result<Bf16Table, String> {
        self.with_view(name, Bf16Table::from_view)
    }

    fn with_view<T>(
        &mut self,
        name: &str,
        convert: impl FnOnce(&TensorView<'_>) -> Result<T, String>,
    ) -> Result<T, String> {
        let file = self
            .map
            .get(name)
            .cloned()
            .ok_or_else(|| format!("missing tensor {name}"))?;
        let slot = self
            .files
            .get_mut(&file)
            .ok_or_else(|| format!("missing shard {file}"))?;
        if slot.1.is_none() {
            let path = slot.0.join(&file);
            let bytes = std::fs::read(&path).map_err(|err| format!("read {}: {err}", path.display()))?;
            slot.1 = Some(bytes);
        }
        let bytes = slot.1.as_ref().unwrap();
        let tensors = SafeTensors::deserialize(bytes).map_err(|err| err.to_string())?;
        let view = tensors.tensor(name).map_err(|err| format!("{name}: {err}"))?;
        let result = convert(&view)?;
        let remaining = self.left.get_mut(&file).expect("shard count");
        *remaining -= 1;
        if *remaining == 0 {
            self.files.get_mut(&file).expect("shard").1 = None;
        }
        Ok(result)
    }
}

/// Reads the flat integer fields of `joint_head_config.json`.
pub fn json_usize_fields(text: &str) -> HashMap<String, usize> {
    let mut out = HashMap::new();
    for part in text.trim().trim_matches(|c| c == '{' || c == '}').split(',') {
        if let Some((key, value)) = part.split_once(':') {
            let key = key.trim().trim_matches('"').to_string();
            if let Ok(value) = value.trim().parse::<usize>() {
                out.insert(key, value);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_head_config_fields() {
        let fields = json_usize_fields(
            "{\n  \"hidden_size\": 4096,\n  \"width\": 1024,\n  \"routing_layers\": 2,\n  \"layers\": 4,\n  \"heads\": 16,\n  \"feedforward\": 4096\n}",
        );
        assert_eq!(fields.get("hidden_size"), Some(&4096));
        assert_eq!(fields.get("heads"), Some(&16));
    }

    #[test]
    fn bf16_round_trips() {
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xc000), -2.0);
    }
}
