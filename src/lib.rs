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
//! `models/kev-0.6b` (or `LoadOptions::weights_dir`). The library does not
//! download them.

#![recursion_limit = "256"]

mod answers;
#[cfg(test)]
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

/// The only checkpoint [`FuzzyDecision::load`] accepts.
pub const DEFAULT_MODEL: &str = "kev-0.6b";

/// Base weights. File: `model.safetensors`. License: Apache-2.0.
pub const BASE_REPO: &str = "Qwen/Qwen3-0.6B-Base";

/// LoRA, tokenizer, and pointer head. License: Apache-2.0.
pub const ADAPTER_REPO: &str = "jaredpalmer/kev-0.6b";

#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Must be [`DEFAULT_MODEL`]. Any other name returns [`Error::UnsupportedModel`].
    pub model: String,
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

/// Where the weight files are, and whether that directory already holds them.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model: &'static str,
    pub base_repo: &'static str,
    pub adapter_repo: &'static str,
    pub weights_dir: PathBuf,
    /// `model.safetensors`, `adapter_model.safetensors`, `head.safetensors`, and `tokenizer.json` are present.
    pub ready: bool,
}

pub struct FuzzyDecision {
    limits: QuestionLimits,
    temperature: f32,
    max_state_tokens: usize,
    max_length: usize,
    truncation: Truncation,
    tokenizer: HfTokenizer,
    model: Qwen3Kev,
}

impl std::fmt::Debug for FuzzyDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FuzzyDecision")
            .field("model", &DEFAULT_MODEL)
            .finish()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
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

fn limits() -> QuestionLimits {
    QuestionLimits {
        min_choice_options: 1,
        max_choice_options: 255,
        min_score_levels: 2,
        max_score_levels: 255,
    }
}

const DEFAULT_TEMPERATURE: f32 = 1.0;
const DEFAULT_MAX_STATE: usize = 8192;
const DEFAULT_MAX_LENGTH: usize = 8192;

fn weights_ready(dir: &Path) -> bool {
    ["model.safetensors", "adapter_model.safetensors", "head.safetensors", "tokenizer.json"]
        .iter()
        .all(|name| dir.join(name).is_file())
}

impl FuzzyDecision {
    /// Load Kev-0.6B from `dir`. The directory must already hold the four weight files.
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
            limits: limits(),
            temperature: options.temperature.unwrap_or(DEFAULT_TEMPERATURE),
            max_state_tokens: options.max_state_tokens.unwrap_or(DEFAULT_MAX_STATE),
            max_length: options.max_length.unwrap_or(DEFAULT_MAX_LENGTH),
            truncation: options.truncation,
            tokenizer,
            model,
        })
    }

    /// Names the checkpoint and reports whether `weights_dir` already holds the four files.
    /// Does not read the tensors and does not download them.
    pub fn info(options: &LoadOptions) -> ModelInfo {
        let weights_dir = options
            .weights_dir
            .clone()
            .unwrap_or_else(weights::default_weights_dir);
        let ready = weights_ready(&weights_dir);
        ModelInfo {
            model: DEFAULT_MODEL,
            base_repo: BASE_REPO,
            adapter_repo: ADAPTER_REPO,
            weights_dir,
            ready,
        }
    }

    pub fn count_tokens(&self, text: &str) -> usize {
        self.tokenizer.count(text)
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
    fn temperature_must_be_positive() {
        let err = FuzzyDecision::load(LoadOptions::default().temperature(0.0)).unwrap_err();
        assert!(matches!(err, Error::Temperature { value } if value == 0.0));
    }

    #[test]
    fn info_names_webgpu() {
        let info = FuzzyDecision::info(&LoadOptions::default());
        assert_eq!(info.model, DEFAULT_MODEL);
        assert_eq!(info.base_repo, BASE_REPO);
        assert_eq!(info.adapter_repo, ADAPTER_REPO);
        assert!(info.weights_dir.ends_with("kev-0.6b"));
    }
}
