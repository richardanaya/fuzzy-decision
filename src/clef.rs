//! One loaded Clef-Flash checkpoint: tokenizer, backbone, and joint head.

use std::path::Path;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::{Tensor, TensorData};

use crate::backbone::{mrope, positions, Backbone, Carry, ImageSpan, TEXT_HIDDEN};
use crate::head::{HeadConfig, HeadQuestion, JointSchemaHead};
use crate::record::Layout;
use crate::tokenize::HfTokenizer;
use crate::Error;

/// The files `Clef::load` reads directly. The safetensors shards named by the
/// index are read while loading and missing shards fail with their own message.
pub const REQUIRED_FILES: [&str; 4] = [
    "tokenizer.json",
    "model.safetensors.index.json",
    "joint_head.safetensors",
    "joint_head_config.json",
];

/// A finished prefix pass: the per-layer carries after the last prefix token
/// and the head's memory rows for the prefix.
pub struct Prefix {
    pub len: usize,
    carries: Vec<Carry>,
    memory: Vec<f32>,
    width: usize,
}

pub struct Clef {
    pub tokenizer: HfTokenizer,
    backbone: Backbone,
    head: JointSchemaHead<Wgpu>,
}

impl Clef {
    pub fn load(dir: &Path, with_vision: bool) -> Result<Self, Error> {
        for file in REQUIRED_FILES {
            if !dir.join(file).is_file() {
                return Err(Error::MissingFile {
                    dir: dir.to_path_buf(),
                    file,
                });
            }
        }
        let weights = |message: String| Error::Weights { message };
        let tokenizer = HfTokenizer::open(&dir.join("tokenizer.json")).map_err(weights)?;
        let device = WgpuDevice::default();
        let config_text =
            std::fs::read_to_string(dir.join("joint_head_config.json")).map_err(|err| Error::Weights {
                message: format!("read joint_head_config.json: {err}"),
            })?;
        let config = HeadConfig::parse(&config_text).map_err(weights)?;
        let head_bytes = std::fs::read(dir.join("joint_head.safetensors")).map_err(|err| Error::Weights {
            message: format!("read joint_head.safetensors: {err}"),
        })?;
        let head = JointSchemaHead::load(&head_bytes, config, &device).map_err(weights)?;
        if config.hidden_size != TEXT_HIDDEN {
            return Err(Error::Weights {
                message: format!(
                    "the joint head reads hidden size {}, the backbone produces {TEXT_HIDDEN}",
                    config.hidden_size
                ),
            });
        }
        let backbone = Backbone::load(dir, with_vision, &device).map_err(weights)?;
        Ok(Self {
            tokenizer,
            backbone,
            head,
        })
    }

    pub fn backbone(&self) -> &Backbone {
        &self.backbone
    }

    /// Scores a layout. `image_rows` are the merged vision-tower rows for the
    /// image named by `image`, needed unless a cached `prefix` is passed.
    /// Returns one logit per option per question, reordered to the caller's
    /// label order, and the prefix of this pass when `keep_prefix` is set.
    pub fn score(
        &self,
        layout: &Layout,
        image: Option<ImageSpan>,
        image_rows: Option<Tensor<Wgpu, 2>>,
        prefix: Option<&Prefix>,
        keep_prefix: bool,
    ) -> Result<(Vec<Vec<f32>>, Option<Prefix>), Error> {
        let device = self.backbone.device().clone();
        let len = layout.ids.len();
        let pos = positions(len, image);
        let (cos, sin) = mrope(&pos, &device);
        let width = self.head.config().width;

        let (memory, tail, tail_offset, new_prefix) = if let Some(prefix) = prefix {
            let suffix_len = len - prefix.len;
            let hidden = self.backbone.embed_ids(&layout.ids[prefix.len..]);
            let (hidden, _) = self.backbone.run(
                hidden,
                &cos.narrow(0, prefix.len, suffix_len),
                &sin.narrow(0, prefix.len, suffix_len),
                Some(&prefix.carries),
            );
            let tail = self.head.normalized(self.backbone.final_norm(hidden));
            let memory_prefix = Tensor::<Wgpu, 2>::from_data(
                TensorData::new(prefix.memory.clone(), [prefix.len, prefix.width]),
                &device,
            );
            let memory = Tensor::cat(vec![memory_prefix, self.head.memory(tail.clone())], 0);
            (memory, tail, prefix.len, None)
        } else if keep_prefix {
            let prefix_len = layout.prefix_len;
            let hidden = self.embed_with_image(&layout.ids[..prefix_len], image, image_rows)?;
            let (hidden, carries) = self.backbone.run(
                hidden,
                &cos.clone().narrow(0, 0, prefix_len),
                &sin.clone().narrow(0, 0, prefix_len),
                None,
            );
            let normalized = self.head.normalized(self.backbone.final_norm(hidden));
            let memory_prefix = self.head.memory(normalized);
            let memory_rows = memory_prefix
                .clone()
                .into_data()
                .as_slice::<f32>()
                .unwrap()
                .to_vec();
            let suffix_len = len - prefix_len;
            let hidden = self.backbone.embed_ids(&layout.ids[prefix_len..]);
            let (hidden, _) = self.backbone.run(
                hidden,
                &cos.narrow(0, prefix_len, suffix_len),
                &sin.narrow(0, prefix_len, suffix_len),
                Some(&carries),
            );
            let tail = self.head.normalized(self.backbone.final_norm(hidden));
            let memory = Tensor::cat(vec![memory_prefix, self.head.memory(tail.clone())], 0);
            (
                memory,
                tail,
                prefix_len,
                Some(Prefix {
                    len: prefix_len,
                    carries,
                    memory: memory_rows,
                    width,
                }),
            )
        } else {
            let hidden = self.embed_with_image(&layout.ids, image, image_rows)?;
            let (hidden, _) = self.backbone.run(hidden, &cos, &sin, None);
            let tail = self.head.normalized(self.backbone.final_norm(hidden));
            let memory = self.head.memory(tail.clone());
            (memory, tail, 0, None)
        };

        let head_questions: Vec<HeadQuestion> = layout
            .questions
            .iter()
            .map(|question| HeadQuestion {
                question_type: question.question_type,
                question_span: question.question_span,
                option_spans: question.option_spans.clone(),
            })
            .collect();
        let lexical = layout
            .questions
            .iter()
            .map(|question| {
                let mut rows = Vec::with_capacity(question.option_spans.len() * TEXT_HIDDEN);
                for (start, end) in &question.option_spans {
                    rows.extend(self.backbone.lexical_mean(&layout.ids[*start..*end]));
                }
                Tensor::<Wgpu, 2>::from_data(
                    TensorData::new(rows, [question.option_spans.len(), TEXT_HIDDEN]),
                    &device,
                )
            })
            .collect();
        let encoded = self.head.score(memory, tail, tail_offset, &head_questions, lexical);

        let logits = layout
            .questions
            .iter()
            .zip(encoded)
            .map(|(question, scores)| {
                question
                    .user_order
                    .iter()
                    .map(|index| scores[*index])
                    .collect()
            })
            .collect();
        Ok((logits, new_prefix))
    }

    fn embed_with_image(
        &self,
        ids: &[u32],
        image: Option<ImageSpan>,
        image_rows: Option<Tensor<Wgpu, 2>>,
    ) -> Result<Tensor<Wgpu, 2>, Error> {
        let hidden = self.backbone.embed_ids(ids);
        match image {
            None => Ok(hidden),
            Some(span) => {
                let rows = image_rows.ok_or_else(|| Error::Weights {
                    message: "an image span was given without its encoded rows".into(),
                })?;
                let n_image = span.llm_h * span.llm_w;
                let len = ids.len();
                let before = hidden.clone().narrow(0, 0, span.at);
                let after = hidden.narrow(0, span.at + n_image, len - span.at - n_image);
                Ok(Tensor::cat(vec![before, rows, after], 0))
            }
        }
    }
}

pub fn weights_ready(dir: &Path) -> bool {
    REQUIRED_FILES.iter().all(|name| dir.join(name).is_file())
}
