//! Load `Qwen/Qwen3-4B-Base`, merge the Kev-4B LoRA, and attach the pointer head.

use std::path::{Path, PathBuf};

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::{Tensor, TensorData};
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;

use crate::qwen::{LayerParts, Qwen3Kev};

const LAYERS: usize = 36;
const LORA_SCALE: f32 = 2.0;

pub fn default_weights_dir() -> PathBuf {
    PathBuf::from("models/kev-4b")
}

pub fn load_kev(dir: &Path, device: &WgpuDevice) -> Result<Qwen3Kev, String> {
    let base_bytes = std::fs::read(dir.join("model.safetensors"))
        .map_err(|err| format!("read model.safetensors in {}: {err}", dir.display()))?;
    let adapter_bytes = std::fs::read(dir.join("adapter_model.safetensors"))
        .map_err(|err| format!("read adapter_model.safetensors in {}: {err}", dir.display()))?;
    let head_bytes = std::fs::read(dir.join("head.safetensors"))
        .map_err(|err| format!("read head.safetensors in {}: {err}", dir.display()))?;

    let base = SafeTensors::deserialize(&base_bytes).map_err(|err| err.to_string())?;
    let adapter = SafeTensors::deserialize(&adapter_bytes).map_err(|err| err.to_string())?;
    let head = SafeTensors::deserialize(&head_bytes).map_err(|err| err.to_string())?;

    let embed = tensor2(&base, "model.embed_tokens.weight", device)?;
    let norm = tensor1(&base, "model.norm.weight", device)?;
    let mut layers = Vec::with_capacity(LAYERS);
    for index in 0..LAYERS {
        let prefix = format!("model.layers.{index}");
        layers.push(
            LayerParts {
                q: merged(&base, &adapter, &format!("{prefix}.self_attn.q_proj"), device)?,
                k: merged(&base, &adapter, &format!("{prefix}.self_attn.k_proj"), device)?,
                v: merged(&base, &adapter, &format!("{prefix}.self_attn.v_proj"), device)?,
                o: merged(&base, &adapter, &format!("{prefix}.self_attn.o_proj"), device)?,
                q_norm: tensor1(&base, &format!("{prefix}.self_attn.q_norm.weight"), device)?,
                k_norm: tensor1(&base, &format!("{prefix}.self_attn.k_norm.weight"), device)?,
                gate: merged(&base, &adapter, &format!("{prefix}.mlp.gate_proj"), device)?,
                up: merged(&base, &adapter, &format!("{prefix}.mlp.up_proj"), device)?,
                down: merged(&base, &adapter, &format!("{prefix}.mlp.down_proj"), device)?,
                input_norm: tensor1(&base, &format!("{prefix}.input_layernorm.weight"), device)?,
                post_norm: tensor1(
                    &base,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    device,
                )?,
            }
        );
    }

    let pointer_q = tensor2(&head, "q.weight", device)?;
    let pointer_k = tensor2(&head, "k.weight", device)?;
    let pointer_q_bias = tensor1(&head, "q.bias", device)?;
    let pointer_k_bias = tensor1(&head, "k.bias", device)?;
    let head_temperature = match head.names().iter().find(|name| name.as_str() == "temperature") {
        Some(_) => tensor1(&head, "temperature", device)?
            .into_data()
            .as_slice::<f32>()
            .map(|v| v[0])
            .unwrap_or(1.0),
        None => 1.0,
    };

    Ok(Qwen3Kev::from_parts(
        embed,
        layers,
        norm,
        pointer_q,
        pointer_k,
        pointer_q_bias,
        pointer_k_bias,
        head_temperature,
        device.clone(),
    ))
}

fn merged(
    base: &SafeTensors<'_>,
    adapter: &SafeTensors<'_>,
    name: &str,
    device: &WgpuDevice,
) -> Result<Tensor<Wgpu, 2>, String> {
    let mut weight = tensor2(base, &format!("{name}.weight"), device)?;
    let lora_name = lora_key(adapter, name)?;
    let a = tensor2(adapter, &format!("{lora_name}.lora_A.weight"), device)?;
    let b = tensor2(adapter, &format!("{lora_name}.lora_B.weight"), device)?;
    let delta = b.matmul(a) * LORA_SCALE;
    weight = weight + delta;
    Ok(weight)
}

fn lora_key(adapter: &SafeTensors<'_>, name: &str) -> Result<String, String> {
    let suffix = format!("{name}.lora_A.weight");
    adapter
        .names()
        .into_iter()
        .find(|key| key.ends_with(&suffix))
        .map(|key| key.trim_end_matches(".lora_A.weight").to_string())
        .ok_or_else(|| format!("LoRA weights for {name} are missing from the adapter"))
}

fn tensor2(
    file: &SafeTensors<'_>,
    name: &str,
    device: &WgpuDevice,
) -> Result<Tensor<Wgpu, 2>, String> {
    let view = file.tensor(name).map_err(|err| format!("{name}: {err}"))?;
    let shape = view.shape().to_vec();
    if shape.len() != 2 {
        return Err(format!("{name} has shape {shape:?}, expected a matrix"));
    }
    let values = to_f32(&view)?;
    Ok(Tensor::from_data(
        TensorData::new(values, [shape[0], shape[1]]),
        device,
    ))
}

fn tensor1(
    file: &SafeTensors<'_>,
    name: &str,
    device: &WgpuDevice,
) -> Result<Tensor<Wgpu, 1>, String> {
    let view = file.tensor(name).map_err(|err| format!("{name}: {err}"))?;
    let shape = view.shape().to_vec();
    if shape.len() != 1 {
        return Err(format!("{name} has shape {shape:?}, expected a vector"));
    }
    let values = to_f32(&view)?;
    Ok(Tensor::from_data(TensorData::new(values, [shape[0]]), device))
}

fn to_f32(view: &TensorView<'_>) -> Result<Vec<f32>, String> {
    let bytes = view.data();
    match view.dtype() {
        safetensors::Dtype::F32 => {
            let mut values = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(values)
        }
        safetensors::Dtype::BF16 => Ok(bytes
            .chunks_exact(2)
            .map(|chunk| f32::from_bits((u16::from_le_bytes([chunk[0], chunk[1]]) as u32) << 16))
            .collect()),
        safetensors::Dtype::F16 => Ok(bytes
            .chunks_exact(2)
            .map(|chunk| half_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
            .collect()),
        other => Err(format!("unsupported dtype {other:?}")),
    }
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
