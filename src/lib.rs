//! Typed decisions scored by Clef-Flash on Burn's WGPU backend.
//!
//! Text mode takes one piece of text (the state) and any number of typed
//! questions. One forward pass returns a probability distribution per question.
//! Nothing is generated: an answer is always one of the options you provided.
//! Vision mode takes an image plus text and scores it with the same checkpoint.
//!
//! `clef-flash` is `Cloudflare/clef-flash`: a Qwen3.5-9B backbone with its
//! vision encoder and the Clef joint schema head, which routes evidence from
//! the state to every question and scores all options jointly in one pass.
//! The forward pass is Burn on the WGPU device. Weights are read from
//! `models/clef-flash` (or `LoadOptions::weights_dir`). The library does not
//! download them.

#![recursion_limit = "256"]

mod answers;
mod backbone;
mod delta;
mod clef;
mod head;
#[cfg(test)]
mod parity;
mod questions;
mod record;
mod tokenize;
mod vision;
mod weights;

pub use answers::{Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer};
pub use questions::{choice, noul, score, Question, Truncation};
pub use vision::{RgbImage, VisionDecision};

/// Which input a decision uses.
///
/// [`Mode::Text`] is [`FuzzyDecision`]: the state is text only.
/// [`Mode::Vision`] is [`VisionDecision`]: an image plus text. Both run
/// Clef-Flash and return the same answer types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Text,
    Vision,
}
pub use tokenize::TokenCounter;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use answers::decode_answer;
use clef::Clef;
use questions::{validate_question, QuestionLimits};
use record::encode_record;

/// The only checkpoint [`FuzzyDecision::load`] accepts.
pub const DEFAULT_MODEL: &str = "clef-flash";

/// The Hugging Face repository with every weight file. License: Apache-2.0.
pub const MODEL_REPO: &str = "Cloudflare/clef-flash";

#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Must be [`DEFAULT_MODEL`]. Any other name returns [`Error::UnsupportedModel`].
    pub model: String,
    pub max_length: Option<usize>,
    pub temperature: Option<f32>,
    pub max_state_tokens: Option<usize>,
    pub truncation: Truncation,
    /// Directory with a local `Cloudflare/clef-flash` snapshot: `tokenizer.json`,
    /// `model.safetensors.index.json` and the shards it names,
    /// `joint_head.safetensors`, and `joint_head_config.json`.
    /// Defaults to `models/clef-flash`.
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
    /// Load `clef-flash` from `dir`, which holds the snapshot files.
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
    /// The Hugging Face repository the snapshot comes from.
    pub repo: &'static str,
    pub weights_dir: PathBuf,
    /// `tokenizer.json`, `model.safetensors.index.json`, `joint_head.safetensors`,
    /// and `joint_head_config.json` are present. The shards the index names are
    /// checked while loading.
    pub ready: bool,
}

pub struct FuzzyDecision {
    limits: QuestionLimits,
    temperature: f32,
    max_state_tokens: usize,
    max_length: usize,
    truncation: Truncation,
    model: Clef,
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
    /// Clef-Flash packs the state and every question into one sequence, so
    /// per-question row limits no longer apply. Kept for callers matching on
    /// this variant; it is not produced.
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
    #[error("only clef-flash is implemented, got {model}")]
    UnsupportedModel { model: String },
    #[error("missing {file} in {}", dir.display())]
    MissingFile { dir: PathBuf, file: &'static str },
    #[error("{message}")]
    Weights { message: String },
}

pub(crate) fn limits() -> QuestionLimits {
    QuestionLimits {
        min_choice_options: 1,
        max_choice_options: 255,
        min_score_levels: 2,
        max_score_levels: 255,
    }
}

const DEFAULT_TEMPERATURE: f32 = 1.0;
pub(crate) const DEFAULT_MAX_STATE: usize = 16384;
pub(crate) const DEFAULT_MAX_LENGTH: usize = 16384;

impl FuzzyDecision {
    /// Load Clef-Flash from `dir`. The directory must already hold the
    /// snapshot files.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, Error> {
        Self::load(LoadOptions::dir(dir.as_ref()))
    }

    pub fn load(options: LoadOptions) -> Result<Self, Error> {
        if options.model != DEFAULT_MODEL {
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
        let model = Clef::load(&dir, false)?;
        Ok(Self {
            limits: limits(),
            temperature: options.temperature.unwrap_or(DEFAULT_TEMPERATURE),
            max_state_tokens: options.max_state_tokens.unwrap_or(DEFAULT_MAX_STATE),
            max_length: options.max_length.unwrap_or(DEFAULT_MAX_LENGTH),
            truncation: options.truncation,
            model,
        })
    }

    /// Names the checkpoint and reports whether `weights_dir` already holds the
    /// snapshot files. Does not read the tensors and does not download them.
    pub fn info(options: &LoadOptions) -> ModelInfo {
        let weights_dir = options
            .weights_dir
            .clone()
            .unwrap_or_else(weights::default_weights_dir);
        let ready = clef::weights_ready(&weights_dir);
        ModelInfo {
            model: DEFAULT_MODEL,
            repo: MODEL_REPO,
            weights_dir,
            ready,
        }
    }

    pub fn count_tokens(&self, text: &str) -> usize {
        self.model.tokenizer.count(text)
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
        let ids: Vec<String> = (1..=questions.len()).map(|n| format!("q{n}")).collect();
        self.decide_validated(state, questions, &ids, &options)
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
        let ids: Vec<String> = ordered.iter().map(|(key, _)| key.clone()).collect();
        let qs: Vec<Question> = ordered.iter().map(|(_, q)| q.clone()).collect();
        let answers = self.decide_validated(state, &qs, &ids, &options)?;
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
        question_ids: &[String],
        options: &DecideOptions,
    ) -> Result<Vec<Answer>, Error> {
        let temperature = options.temperature.unwrap_or(self.temperature);
        check_temperature(temperature)?;
        if questions.is_empty() {
            return Ok(Vec::new());
        }
        let strict = options.truncation.unwrap_or(self.truncation) == Truncation::Error;
        let layout = encode_record(
            &self.model.tokenizer,
            state,
            None,
            questions,
            question_ids,
            self.max_length,
            options.max_state_tokens.unwrap_or(self.max_state_tokens),
            strict,
        )?;
        let (logits, _) = self.model.score(&layout, None, None, None, false)?;
        Ok(questions
            .iter()
            .zip(logits)
            .map(|(question, question_logits)| decode_answer(question, &question_logits, temperature))
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
            model: "kev-4b".into(),
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
    fn info_names_the_checkpoint() {
        let info = FuzzyDecision::info(&LoadOptions::default());
        assert_eq!(info.model, DEFAULT_MODEL);
        assert_eq!(info.repo, MODEL_REPO);
        assert!(info.weights_dir.ends_with("clef-flash"));
    }
}
