//! Reading the Clef-Flash snapshot: sharded safetensors, the joint head file,
//! and the small JSON head config.
//!
//! Tensor bytes come from burn-store's [`SafetensorsStore`]. This crate does
//! not call the `safetensors` crate itself.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use burn::tensor::{bf16, DType, TensorData};
use burn_store::{ModuleStore, SafetensorsStore};

pub fn default_weights_dir() -> PathBuf {
    PathBuf::from("models/clef-flash")
}

/// Materializes one tensor from a burn-store safetensors reader.
pub fn tensor_data(store: &mut SafetensorsStore, name: &str) -> Result<TensorData, String> {
    let snapshot = match store.get_snapshot(name) {
        Ok(Some(snapshot)) => snapshot,
        Ok(None) => return Err(format!("missing tensor {name}")),
        Err(err) => return Err(format!("{name}: {err}")),
    };
    snapshot.to_data().map_err(|err| format!("{name}: {err}"))
}

pub fn dims(data: &TensorData) -> Vec<usize> {
    data.shape.iter().copied().collect()
}

/// f32 values of an f32, bf16, or f16 tensor. Conversion goes through burn's
/// element types, which match the usual half-precision bit casts.
pub fn to_f32(data: TensorData) -> Result<Vec<f32>, String> {
    match data.dtype {
        DType::F32 | DType::BF16 | DType::F16 => data
            .convert::<f32>()
            .to_vec()
            .map_err(|err| err.to_string()),
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
    pub fn from_data(data: TensorData) -> Result<Self, String> {
        let shape = dims(&data);
        if shape.len() != 2 {
            return Err(format!("token table has shape {shape:?}"));
        }
        // Keep the top 16 bits. Native bf16 is already those bits; f32 is
        // truncated the same way (not rounded) so a float32 table matches the
        // previous loader.
        let values: Vec<u16> = match data.dtype {
            DType::BF16 => data
                .as_slice::<bf16>()
                .map_err(|err| err.to_string())?
                .iter()
                .map(|value| value.to_bits())
                .collect(),
            DType::F32 => data
                .as_slice::<f32>()
                .map_err(|err| err.to_string())?
                .iter()
                .map(|value| (value.to_bits() >> 16) as u16)
                .collect(),
            other => return Err(format!("unsupported table dtype {other:?}")),
        };
        Ok(Self {
            data: values,
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

/// Reads tensors across the sharded `model-*.safetensors` files. Each shard is
/// a burn-store [`SafetensorsStore`] (memory-mapped) and is dropped once every
/// tensor this loader asked for has been read.
pub struct Store {
    files: HashMap<String, Shard>,
    map: HashMap<String, String>,
    left: HashMap<String, usize>,
}

struct Shard {
    path: PathBuf,
    store: Option<SafetensorsStore>,
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
                .map(|file| {
                    (
                        file.clone(),
                        Shard {
                            path: dir.join(file),
                            store: None,
                        },
                    )
                })
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
                self.files.get_mut(&file).expect("shard").store = None;
            }
        }
    }

    pub fn read(&mut self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        self.with_data(name, |data| {
            let shape = dims(&data);
            let values = to_f32(data)?;
            Ok((values, shape))
        })
    }

    pub fn table(&mut self, name: &str) -> Result<Bf16Table, String> {
        self.with_data(name, Bf16Table::from_data)
    }

    fn with_data<T>(
        &mut self,
        name: &str,
        convert: impl FnOnce(TensorData) -> Result<T, String>,
    ) -> Result<T, String> {
        let file = self
            .map
            .get(name)
            .cloned()
            .ok_or_else(|| format!("missing tensor {name}"))?;
        let data = {
            let shard = self
                .files
                .get_mut(&file)
                .ok_or_else(|| format!("missing shard {file}"))?;
            if shard.store.is_none() {
                if !shard.path.is_file() {
                    return Err(format!(
                        "read {}: No such file or directory (os error 2)",
                        shard.path.display()
                    ));
                }
                let path = shard.path.clone();
                shard.store = Some(SafetensorsStore::from_file(path));
            }
            let store = shard.store.as_mut().expect("shard store");
            tensor_data(store, name)?
        };
        let result = convert(data)?;
        let remaining = self.left.get_mut(&file).expect("shard count");
        *remaining -= 1;
        if *remaining == 0 {
            self.files.get_mut(&file).expect("shard").store = None;
        }
        Ok(result)
    }
}

/// Reads the flat integer fields of `joint_head_config.json`.
pub fn json_usize_fields(text: &str) -> HashMap<String, usize> {
    let mut out = HashMap::new();
    for part in text
        .trim()
        .trim_matches(|c| c == '{' || c == '}')
        .split(',')
    {
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
