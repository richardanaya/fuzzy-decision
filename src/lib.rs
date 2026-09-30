//! Typed decisions scored with Kev-0.6B on Burn's WGPU backend.
//!
//! One piece of text (the state) and any number of typed questions go in. One
//! forward pass on Burn's WGPU backend returns a probability distribution per
//! question. Nothing is generated: an answer is always one of the options you
//! provided.
//!
//! `kev-0.6b` runs the published checkpoint: `Qwen/Qwen3-0.6B-Base` with the
//! `jaredpalmer/kev-0.6b` LoRA merged in, and that checkpoint's pointer head.
//! The forward pass is Burn on the WGPU device. Weights are read from
//! `models/kev-0.6b` (or `LoadOptions::weights_dir`).

#![recursion_limit = "256"]

mod answers;
mod encoding;
mod questions;
mod qwen;
mod tokenize;
mod weights;

pub use answers::{Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer};
pub use questions::{choice, noul, score, Question, Truncation};
pub use tokenize::TokenCounter;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use burn::backend::wgpu::WgpuDevice;

use answers::decode_answer;
use questions::{validate_question, QuestionLimits};
use qwen::Qwen3Kev;
use tokenize::HfTokenizer;

/// Hugging Face repos behind the built-in aliases.
pub const MODELS: &[(&str, &str)] = &[
    ("kev-0.6b", "onnx-community/kev-0.6b-ONNX"),
    ("kev-4b", "onnx-community/kev-4b-ONNX"),
];

pub const DEFAULT_MODEL: &str = "kev-0.6b";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    Kev,
}

impl Family {
    pub fn as_str(self) -> &'static str {
        match self {
            Family::Kev => "kev",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Alias (`kev-0.6b`, `kev-4b`) or any other repo id.
    /// Unknown ids are treated as the kev family.
    pub model: String,
    /// Recorded on [`Runtime`]. The in-process head is f32.
    pub dtype: String,
    pub device: String,
    pub max_length: Option<usize>,
    pub temperature: Option<f32>,
    pub max_state_tokens: Option<usize>,
    pub truncation: Truncation,
    /// Directory with `model.safetensors`, `adapter_model.safetensors`,
    /// `head.safetensors`, and `tokenizer.json`. Defaults to `models/kev-0.6b`.
    pub weights_dir: Option<PathBuf>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            model: DEFAULT_MODEL.to_string(),
            dtype: "auto".to_string(),
            device: "auto".to_string(),
            max_length: None,
            temperature: None,
            max_state_tokens: None,
            truncation: Truncation::Cut,
            weights_dir: None,
        }
    }
}

impl LoadOptions {
    /// Load `kev-0.6b` from `dir`, which holds the four weight files.
    pub fn dir(dir: impl Into<PathBuf>) -> Self {
        Self {
            weights_dir: Some(dir.into()),
            ..Self::default()
        }
    }

    pub fn max_length(mut self, tokens: usize) -> Self {
        self.max_length = Some(tokens);
        self
    }

    pub fn max_state_tokens(mut self, tokens: usize) -> Self {
        self.max_state_tokens = Some(tokens);
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn truncation(mut self, truncation: Truncation) -> Self {
        self.truncation = truncation;
        self
    }
}

#[derive(Debug, Clone)]
pub struct DecideOptions {
    pub temperature: Option<f32>,
    pub max_state_tokens: Option<usize>,
    pub truncation: Option<Truncation>,
}

impl Default for DecideOptions {
    fn default() -> Self {
        Self {
            temperature: None,
            max_state_tokens: None,
            truncation: None,
        }
    }
}

impl DecideOptions {
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn max_state_tokens(mut self, tokens: usize) -> Self {
        self.max_state_tokens = Some(tokens);
        self
    }

    pub fn truncation(mut self, truncation: Truncation) -> Self {
        self.truncation = Some(truncation);
        self
    }
}

#[derive(Debug, Clone)]
pub struct Runtime {
    pub model: String,
    pub family: Family,
    pub device: String,
    pub dtype: String,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub is_cached: bool,
    pub download_size: u64,
    pub model: String,
    pub family: Family,
    pub device: String,
    pub dtype: String,
}

pub struct FuzzyDecision {
    runtime: Runtime,
    limits: QuestionLimits,
    temperature: f32,
    max_state_tokens: usize,
    max_length: usize,
    truncation: Truncation,
    tokenizer: HfTokenizer,
    model: Qwen3Kev,
    disposed: bool,
}

impl std::fmt::Debug for FuzzyDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FuzzyDecision")
            .field("runtime", &self.runtime)
            .field("disposed", &self.disposed)
            .finish()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("fuzzy-decision instance has been disposed")]
    Disposed,
    #[error("Question {label} needs non-empty instructions.")]
    EmptyInstructions { label: String },
    #[error("Question {label} has unknown type \"{got}\". Expected \"choice\", \"score\" or \"noul\".")]
    UnknownType { label: String, got: String },
    #[error("Question {label} ({kind}) needs between {min} and {max} options for this model, got {got}.")]
    OptionCount {
        label: String,
        kind: String,
        min: usize,
        max: usize,
        got: usize,
    },
    #[error("Question {label} has an empty option.")]
    BadOption { label: String },
    #[error("Question {label} has a duplicate option \"{option}\".")]
    DuplicateOption { label: String, option: String },
    #[error("no option named \"{option}\"")]
    UnknownOption { option: String },
    #[error("temperature must be finite and greater than 0, got {value}")]
    Temperature { value: f32 },
    #[error(
        "a question needs {question_tokens} tokens with a {state_tokens}-token state, past the {limit} token row limit"
    )]
    Row {
        state_tokens: usize,
        question_tokens: usize,
        limit: usize,
    },
    #[error("{message}")]
    Context { message: String },
    #[error("State needs {state_tokens} tokens and only {kept} fit in the context.")]
    Truncated { state_tokens: usize, kept: usize },
    #[error("only kev-0.6b is implemented, got {model}")]
    UnsupportedModel { model: String },
    #[error("missing {file} in {}", dir.display())]
    MissingFile { dir: PathBuf, file: &'static str },
    #[error("{message}")]
    Weights { message: String },
}

struct FamilySpec {
    family: Family,
    limits: QuestionLimits,
    temperature: f32,
    max_state_tokens: usize,
    max_length: usize,
    dtype: &'static str,
}

fn family_for_model(_model: &str) -> FamilySpec {
    FamilySpec {
        family: Family::Kev,
        limits: QuestionLimits {
            min_choice_options: 1,
            max_choice_options: 255,
            min_score_levels: 2,
            max_score_levels: 255,
        },
        temperature: 1.0,
        max_state_tokens: 8192,
        max_length: 8192,
        dtype: "q4f16",
    }
}

fn resolve_repo(model: &str) -> String {
    MODELS
        .iter()
        .find(|(alias, _)| *alias == model)
        .map(|(_, repo)| (*repo).to_string())
        .unwrap_or_else(|| model.to_string())
}

impl FuzzyDecision {
    /// Load Kev-0.6B onto the default WGPU device.
    ///
    /// `options.model` must be `kev-0.6b`. The directory must contain the base
    /// `model.safetensors`, `adapter_model.safetensors`, `head.safetensors`,
    /// and `tokenizer.json`.
    /// Load Kev-0.6B from `dir`.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, Error> {
        Self::load(LoadOptions::dir(dir.as_ref()))
    }

    pub fn load(options: LoadOptions) -> Result<Self, Error> {
        if options.model != "kev-0.6b" {
            return Err(Error::UnsupportedModel {
                model: options.model,
            });
        }
        if let Some(temperature) = options.temperature {
            check_temperature(temperature)?;
        }
        let spec = family_for_model(&options.model);
        let dtype = if options.dtype == "auto" {
            "bf16".to_string()
        } else {
            options.dtype.clone()
        };
        let device_name = if options.device == "auto" {
            "webgpu".to_string()
        } else {
            options.device.clone()
        };
        let dir = options
            .weights_dir
            .clone()
            .unwrap_or_else(weights::default_weights_dir);
        for file in [
            "model.safetensors",
            "adapter_model.safetensors",
            "head.safetensors",
            "tokenizer.json",
        ] {
            if !dir.join(file).is_file() {
                return Err(Error::MissingFile {
                    dir: dir.clone(),
                    file,
                });
            }
        }
        let device = WgpuDevice::default();
        let tokenizer = HfTokenizer::open(&dir.join("tokenizer.json"))
            .map_err(|message| Error::Weights { message })?;
        let model = weights::load_kev(&dir, &device).map_err(|message| Error::Weights { message })?;
        Ok(Self {
            runtime: Runtime {
                model: options.model,
                family: spec.family,
                device: device_name,
                dtype,
            },
            limits: spec.limits,
            temperature: options.temperature.unwrap_or(spec.temperature),
            max_state_tokens: options.max_state_tokens.unwrap_or(spec.max_state_tokens),
            max_length: options.max_length.unwrap_or(spec.max_length),
            truncation: options.truncation,
            tokenizer,
            model,
            disposed: false,
        })
    }

    /// Cache metadata. Weights live in the process, so nothing is downloaded.
    pub fn info(options: &LoadOptions) -> ModelInfo {
        let spec = family_for_model(&options.model);
        let dtype = if options.dtype == "auto" {
            spec.dtype.to_string()
        } else {
            options.dtype.clone()
        };
        let device = if options.device == "auto" {
            "webgpu".to_string()
        } else {
            options.device.clone()
        };
        ModelInfo {
            is_cached: true,
            download_size: 0,
            model: resolve_repo(&options.model),
            family: spec.family,
            device,
            dtype,
        }
    }

    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    pub fn count_tokens(&self, text: &str) -> usize {
        self.tokenizer.count(text)
    }

    pub fn dispose(&mut self) {
        self.disposed = true;
    }

    /// Yes/no on `statement`. `answer` is true when the yes probability is at least 0.5.
    pub fn noul(
        &self,
        state: &str,
        statement: impl Into<String>,
    ) -> Result<NoulAnswer, Error> {
        self.noul_with(state, statement, DecideOptions::default())
    }

    pub fn noul_with(
        &self,
        state: &str,
        statement: impl Into<String>,
        options: DecideOptions,
    ) -> Result<NoulAnswer, Error> {
        match self.one(state, noul(statement), options)? {
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

    /// Pick one of `options`. Probabilities sum to 1 over those options.
    pub fn choice(
        &self,
        state: &str,
        instructions: impl Into<String>,
        options: &[&str],
    ) -> Result<ChoiceAnswer, Error> {
        self.choice_with(state, instructions, options, None, DecideOptions::default())
    }

    pub fn choice_with(
        &self,
        state: &str,
        instructions: impl Into<String>,
        options: &[&str],
        descriptions: Option<&[(&str, &str)]>,
        decide: DecideOptions,
    ) -> Result<ChoiceAnswer, Error> {
        match self.one(state, choice(instructions, options, descriptions), decide)? {
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

    /// Expected level index over `levels`, which are ordered from low to high.
    pub fn score(
        &self,
        state: &str,
        instructions: impl Into<String>,
        levels: &[&str],
    ) -> Result<ScoreAnswer, Error> {
        self.score_with(state, instructions, levels, DecideOptions::default())
    }

    pub fn score_with(
        &self,
        state: &str,
        instructions: impl Into<String>,
        levels: &[&str],
        options: DecideOptions,
    ) -> Result<ScoreAnswer, Error> {
        match self.one(state, score(instructions, levels), options)? {
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

    fn one(&self, state: &str, question: Question, options: DecideOptions) -> Result<Answer, Error> {
        let mut answers = self.decide(state, &[question], options)?;
        Ok(answers.pop().expect("one question returns one answer"))
    }

    pub fn decide(
        &self,
        state: &str,
        questions: &[Question],
        options: DecideOptions,
    ) -> Result<Vec<Answer>, Error> {
        self.ensure_live()?;
        for (index, question) in questions.iter().enumerate() {
            validate_question(question, &format!("#{index}"), self.limits)?;
        }
        self.decide_validated(state, questions, &options)
    }

    pub fn decide_map(
        &self,
        state: &str,
        questions: &BTreeMap<String, Question>,
        options: DecideOptions,
    ) -> Result<BTreeMap<String, Answer>, Error> {
        self.ensure_live()?;
        let mut ordered: Vec<(String, Question)> = Vec::with_capacity(questions.len());
        for (key, question) in questions {
            validate_question(question, key, self.limits)?;
            ordered.push((key.clone(), question.clone()));
        }
        let qs: Vec<Question> = ordered.iter().map(|(_, q)| q.clone()).collect();
        let answers = self.decide_validated(state, &qs, &options)?;
        Ok(ordered
            .into_iter()
            .map(|(k, _)| k)
            .zip(answers)
            .collect())
    }

    fn decide_validated(
        &self,
        state: &str,
        questions: &[Question],
        options: &DecideOptions,
    ) -> Result<Vec<Answer>, Error> {
        let strict = options.truncation.unwrap_or(self.truncation) == Truncation::Error;
        let packed = self.tokenizer.pack(
            state,
            questions,
            options.max_state_tokens.unwrap_or(self.max_state_tokens),
            self.max_length,
            strict,
        )?;
        let temperature = options.temperature.unwrap_or(self.temperature);
        check_temperature(temperature)?;
        let grouped = self.model.score(&packed, temperature);
        Ok(questions
            .iter()
            .zip(grouped)
            .map(|(question, logits)| decode_answer(question, &logits, 1.0))
            .collect())
    }

    fn ensure_live(&self) -> Result<(), Error> {
        if self.disposed {
            Err(Error::Disposed)
        } else {
            Ok(())
        }
    }
}

fn check_temperature(temperature: f32) -> Result<(), Error> {
    if temperature.is_finite() && temperature > 0.0 {
        Ok(())
    } else {
        Err(Error::Temperature { value: temperature })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn other_aliases_are_not_loaded() {
        let err = FuzzyDecision::load(LoadOptions {
            model: "other".into(),
            ..LoadOptions::default()
        })
        .unwrap_err();
        assert!(matches!(err, Error::UnsupportedModel { .. }));
    }

    #[test]
    fn missing_weight_file_names_the_directory() {
        let err = FuzzyDecision::open("/tmp/fuzzy-decision-missing-weights").unwrap_err();
        let Error::MissingFile { file, .. } = err else {
            panic!("expected a missing file, got {err}");
        };
        assert_eq!(file, "model.safetensors");
    }

    #[test]
    fn temperature_must_be_positive() {
        let err = FuzzyDecision::load(LoadOptions::default().temperature(0.0)).unwrap_err();
        assert!(matches!(err, Error::Temperature { value } if value == 0.0));
    }

    #[test]
    fn info_names_webgpu() {
        let info = FuzzyDecision::info(&LoadOptions::default());
        assert_eq!(info.device, "webgpu");
        assert_eq!(info.family, Family::Kev);
        assert!(info.is_cached);
    }
}
