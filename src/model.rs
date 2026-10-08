//! One loaded d1-omni checkpoint: trunk, decision head, and either the audio
//! tower or the vision tower.

use std::path::Path;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::conformer::AudioTower;
use crate::head::DecisionHead;
use crate::mel::{log_mel, prepare_waveform};
use crate::nn::tensor2;
use crate::prompt::Modality;
use crate::trunk::Trunk;
use crate::vision::{image_crops, RgbImage, VisionTower};
use crate::weights::{weights_ready, D1Config, EmbedTable, Snapshot};
use crate::Error;

pub use crate::weights::REQUIRED_FILES;

pub enum Which {
    TextAudio,
    Vision,
}

pub struct Session<B: Backend> {
    pub tokenizer: crate::tokenize::HfTokenizer,
    trunk: Trunk<B>,
    head: DecisionHead<B>,
    embed: EmbedTable,
    audio: Option<AudioTower<B>>,
    vision: Option<VisionTower<B>>,
    config: D1Config,
    device: B::Device,
}

impl Session<Wgpu> {
    pub fn load(dir: &Path, which: Which) -> Result<Self, Error> {
        // Missing files fail before a device exists, so a test with no
        // snapshot does not need a GPU.
        for file in REQUIRED_FILES {
            if !dir.join(file).is_file() {
                return Err(Error::MissingFile {
                    dir: dir.to_path_buf(),
                    file,
                });
            }
        }
        let config = D1Config::open(&dir.join("config.json"))
            .map_err(|message| Error::Weights { message })?;
        let device = WgpuDevice::default();
        Self::open(dir, which, config, device)
    }
}

impl<B: Backend> Session<B> {
    fn open(dir: &Path, which: Which, config: D1Config, device: B::Device) -> Result<Self, Error> {
        let tokenizer = crate::tokenize::HfTokenizer::open(&dir.join("tokenizer.json"))
            .map_err(|message| Error::Weights { message })?;
        let snapshot = Snapshot::open(&dir.join("model.safetensors"))
            .map_err(|message| Error::Weights { message })?;
        let embed = EmbedTable::load(&snapshot, "encoder.embed_tokens.weight")
            .map_err(|message| Error::Weights { message })?;
        let trunk = Trunk::load(&snapshot, &config.text, &device)
            .map_err(|message| Error::Weights { message })?;
        let head = DecisionHead::load(&snapshot, config.text.hidden, config.head_layers, &device)
            .map_err(|message| Error::Weights { message })?;
        let audio = match which {
            Which::TextAudio => Some(
                AudioTower::load(&snapshot, &config.audio, &device)
                    .map_err(|message| Error::Weights { message })?,
            ),
            Which::Vision => None,
        };
        let vision = match which {
            Which::Vision => Some(
                VisionTower::load(&snapshot, &config.vision, &device)
                    .map_err(|message| Error::Weights { message })?,
            ),
            Which::TextAudio => None,
        };
        drop(snapshot);
        Ok(Self {
            tokenizer,
            trunk,
            head,
            embed,
            audio,
            vision,
            config,
            device,
        })
    }

    pub fn calibrated(&self, kind: &str, options: usize) -> f32 {
        self.config.temperature(kind, options)
    }

    pub fn text_limit(
        &self,
        modality: Modality,
        prefix: usize,
        user_max: Option<usize>,
    ) -> Result<usize, Error> {
        let cap = match modality {
            Modality::Text => self.config.max_length,
            Modality::Audio => self.config.audio_text_length,
            Modality::Vision => self.config.image_text_length,
        };
        let max_len = user_max
            .unwrap_or(usize::MAX)
            .min(cap)
            .min(self.config.max_length.saturating_sub(prefix));
        if max_len < 64 {
            return Err(Error::Context {
                message: format!(
                    "the media take {prefix} of the {} positions; send a shorter clip or a smaller image",
                    self.config.max_length
                ),
            });
        }
        Ok(max_len)
    }

    pub fn audio_prefix(&self, samples_16k: &[f32]) -> Result<Tensor<B, 2>, Error> {
        let tower = self.audio.as_ref().ok_or_else(|| Error::Audio {
            message: "this checkpoint was loaded without the audio tower".into(),
        })?;
        let wave = prepare_waveform(samples_16k);
        let mel = log_mel(&wave);
        Ok(tower.forward_mel(&mel.features, mel.n_mels, mel.frames, mel.valid))
    }

    pub fn vision_prefix(&self, image: &RgbImage) -> Result<Tensor<B, 2>, Error> {
        let tower = self.vision.as_ref().ok_or_else(|| Error::Weights {
            message: "this checkpoint was loaded without the vision tower".into(),
        })?;
        let crops = image_crops(image).map_err(|message| Error::Weights { message })?;
        let mut parts = Vec::with_capacity(crops.len());
        for crop in &crops {
            parts.push(tower.forward_crop(&crop.patches, crop.height, crop.width, crop.patch_dim));
        }
        Ok(if parts.len() == 1 {
            parts.remove(0)
        } else {
            Tensor::cat(parts, 0)
        })
    }

    pub fn question_logits(
        &self,
        prefix: Option<&Tensor<B, 2>>,
        ids: &[u32],
        qtype: usize,
        markers: &[usize],
    ) -> Result<Vec<f32>, Error> {
        let text = tensor2(
            self.embed
                .gather(ids)
                .map_err(|message| Error::Weights { message })?,
            ids.len(),
            self.embed.dim,
            &self.device,
        );
        let (full, prefix_len) = match prefix {
            Some(prefix) => (Tensor::cat(vec![prefix.clone(), text], 0), prefix.dims()[0]),
            None => (text, 0),
        };
        let hidden = self.trunk.forward(full, prefix_len);
        let text_hidden = hidden.narrow(0, prefix_len, ids.len());
        Ok(self.head.logits(text_hidden, qtype, markers))
    }
}

pub fn ready(dir: &Path) -> bool {
    weights_ready(dir)
}
