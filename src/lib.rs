//! Typed decisions scored by Liquid AI's d1-omni-600M on Burn's WGPU backend.
//!
//! Text mode takes one piece of text and any number of typed questions. Audio
//! mode prepends a clip. Vision mode prepends an image. One forward pass per
//! question returns a probability distribution. Nothing is generated: an
//! answer is always one of the options you provided.
//!
//! The checkpoint is `LiquidAI/d1-omni-600M`. Weights are read from
//! `models/d1-omni-600M` (or `LoadOptions::weights_dir`). The library does not
//! download them. Questions are scored independently. Text questions use the
//! per-type temperatures in `config.json`; image and audio questions stay at
//! temperature 1 unless you set one.

#![recursion_limit = "256"]

mod answers;
mod conformer;
mod head;
mod mel;
mod model;
mod nn;
#[cfg(test)]
mod parity;
mod prompt;
mod questions;
mod resample;
mod tokenize;
mod trunk;
mod vision;
mod wav;
mod weights;

pub use answers::{Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer};
pub use questions::{choice, noul, score, Question, Truncation};
pub use vision::{RgbImage, TILE};
pub use wav::AudioClip;

/// Which input a decision uses.
///
/// [`Mode::Text`] and [`Mode::Audio`] are [`FuzzyDecision`]. [`Mode::Vision`]
/// is [`VisionDecision`]. A request carries an image or a clip, not both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Text,
    Vision,
    Audio,
}

pub use tokenize::TokenCounter;

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use answers::decode_answer;
use burn::tensor::Tensor;
use model::{ready, Session, Which};
use prompt::{encode, kind_name, Modality};
use questions::{validate_question, QuestionLimits};
use vision::image_stamp;

use burn::backend::wgpu::Wgpu;

/// The only checkpoint [`FuzzyDecision::load`] accepts.
pub const DEFAULT_MODEL: &str = "d1-omni-600M";

/// The Hugging Face repository. The weights use the LFM Open License v1.0,
/// which limits commercial use to entities under $10M annual revenue.
pub const MODEL_REPO: &str = "LiquidAI/d1-omni-600M";

#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Must be [`DEFAULT_MODEL`]. Any other name returns [`Error::UnsupportedModel`].
    pub model: String,
    pub max_length: Option<usize>,
    /// Replaces the checkpoint's text temperatures. Image and audio decisions
    /// also use this value when it is set.
    pub temperature: Option<f32>,
    pub max_state_tokens: Option<usize>,
    pub truncation: Truncation,
    /// Directory with `tokenizer.json`, `config.json`, and `model.safetensors`.
    /// Defaults to `models/d1-omni-600M`.
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
    /// Load `d1-omni-600M` from `dir`.
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

#[derive(Debug, Clone, Default)]
pub struct DecideOptions {
    pub temperature: Option<f32>,
    pub max_state_tokens: Option<usize>,
    pub truncation: Option<Truncation>,
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
    /// `tokenizer.json`, `config.json`, and `model.safetensors` are present.
    pub ready: bool,
}

pub struct FuzzyDecision {
    limits: QuestionLimits,
    user_temperature: Option<f32>,
    max_state_tokens: usize,
    max_length: Option<usize>,
    truncation: Truncation,
    session: Session<Wgpu>,
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
    #[error(
        "Question {label} has unknown type \"{got}\". Expected \"choice\", \"score\" or \"noul\"."
    )]
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
    /// Kept for callers matching on this variant. d1 scores one question per
    /// sequence, so a row limit is not produced.
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
    #[error("only d1-omni-600M is implemented, got {model}")]
    UnsupportedModel { model: String },
    #[error("missing {file} in {}", dir.display())]
    MissingFile { dir: PathBuf, file: &'static str },
    #[error("{message}")]
    Weights { message: String },
    #[error("{message}")]
    Audio { message: String },
}

pub(crate) fn limits() -> QuestionLimits {
    QuestionLimits {
        min_choice_options: 2,
        max_choice_options: 255,
        min_score_levels: 2,
        max_score_levels: 10,
    }
}

pub(crate) const DEFAULT_MAX_STATE: usize = 16384;

impl FuzzyDecision {
    /// Load d1-omni-600M from `dir`. The directory must already hold the snapshot.
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
        let session = Session::load(&dir, Which::TextAudio)?;
        Ok(Self {
            limits: limits(),
            user_temperature: options.temperature,
            max_state_tokens: options.max_state_tokens.unwrap_or(DEFAULT_MAX_STATE),
            max_length: options.max_length,
            truncation: options.truncation,
            session,
        })
    }

    /// Names the checkpoint and reports whether `weights_dir` already holds the
    /// snapshot files. Does not read the tensors and does not download them.
    pub fn info(options: &LoadOptions) -> ModelInfo {
        let weights_dir = options
            .weights_dir
            .clone()
            .unwrap_or_else(weights::default_weights_dir);
        ModelInfo {
            model: DEFAULT_MODEL,
            repo: MODEL_REPO,
            weights_dir: weights_dir.clone(),
            ready: ready(&weights_dir),
        }
    }

    pub fn count_tokens(&self, text: &str) -> usize {
        self.session.tokenizer.count(text)
    }

    pub fn noul(&self, state: &str, statement: impl Into<String>) -> Result<NoulAnswer, Error> {
        self.noul_with(state, statement, DecideOptions::default())
    }

    pub fn noul_with(
        &self,
        state: &str,
        statement: impl Into<String>,
        options: DecideOptions,
    ) -> Result<NoulAnswer, Error> {
        match self.one(state, noul(statement), options, None, Modality::Text)? {
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
        match self.one(
            state,
            choice(instructions, options, descriptions),
            decide,
            None,
            Modality::Text,
        )? {
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
        match self.one(
            state,
            score(instructions, levels),
            options,
            None,
            Modality::Text,
        )? {
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

    pub fn decide(
        &self,
        state: &str,
        questions: &[Question],
        options: DecideOptions,
    ) -> Result<Vec<Answer>, Error> {
        self.decide_mode(state, questions, options, None, Modality::Text)
    }

    pub fn decide_map(
        &self,
        state: &str,
        questions: &BTreeMap<String, Question>,
        options: DecideOptions,
    ) -> Result<BTreeMap<String, Answer>, Error> {
        let mut ordered = Vec::with_capacity(questions.len());
        for (key, question) in questions {
            validate_question(question, key, self.limits)?;
            ordered.push((key.clone(), question.clone()));
        }
        let qs: Vec<Question> = ordered
            .iter()
            .map(|(_, question)| question.clone())
            .collect();
        let answers = self.decide_validated(state, &qs, &options, None, Modality::Text)?;
        Ok(ordered
            .into_iter()
            .map(|(key, _)| key)
            .zip(answers)
            .collect())
    }

    /// Score `questions` over a clip. Pass `"{}"` when the clip is the whole state.
    /// Samples that are not 16 kHz mono are linearly resampled and averaged.
    /// The waveform is then cut to 30 s and padded to 0.5 s.
    pub fn decide_audio(
        &self,
        audio: &AudioClip,
        state: &str,
        questions: &[Question],
        options: DecideOptions,
    ) -> Result<Vec<Answer>, Error> {
        let prefix = self.session.audio_prefix(&audio.at_16k())?;
        self.decide_mode(state, questions, options, Some(prefix), Modality::Audio)
    }

    pub fn choice_audio(
        &self,
        audio: &AudioClip,
        state: &str,
        instructions: impl Into<String>,
        options: &[&str],
        descriptions: Option<&[(&str, &str)]>,
    ) -> Result<ChoiceAnswer, Error> {
        match self.one(
            state,
            choice(instructions, options, descriptions),
            DecideOptions::default(),
            Some(self.session.audio_prefix(&audio.at_16k())?),
            Modality::Audio,
        )? {
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

    pub fn noul_audio(
        &self,
        audio: &AudioClip,
        state: &str,
        statement: impl Into<String>,
    ) -> Result<NoulAnswer, Error> {
        match self.one(
            state,
            noul(statement),
            DecideOptions::default(),
            Some(self.session.audio_prefix(&audio.at_16k())?),
            Modality::Audio,
        )? {
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

    pub fn score_audio(
        &self,
        audio: &AudioClip,
        state: &str,
        instructions: impl Into<String>,
        levels: &[&str],
    ) -> Result<ScoreAnswer, Error> {
        match self.one(
            state,
            score(instructions, levels),
            DecideOptions::default(),
            Some(self.session.audio_prefix(&audio.at_16k())?),
            Modality::Audio,
        )? {
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

    fn one(
        &self,
        state: &str,
        question: Question,
        options: DecideOptions,
        prefix: Option<Tensor<Wgpu, 2>>,
        modality: Modality,
    ) -> Result<Answer, Error> {
        let mut answers = self.decide_mode(state, &[question], options, prefix, modality)?;
        Ok(answers.pop().expect("one question returns one answer"))
    }

    fn decide_mode(
        &self,
        state: &str,
        questions: &[Question],
        options: DecideOptions,
        prefix: Option<Tensor<Wgpu, 2>>,
        modality: Modality,
    ) -> Result<Vec<Answer>, Error> {
        for (index, question) in questions.iter().enumerate() {
            validate_question(question, &format!("#{index}"), self.limits)?;
        }
        self.decide_validated(state, questions, &options, prefix, modality)
    }

    fn decide_validated(
        &self,
        state: &str,
        questions: &[Question],
        options: &DecideOptions,
        prefix: Option<Tensor<Wgpu, 2>>,
        modality: Modality,
    ) -> Result<Vec<Answer>, Error> {
        if questions.is_empty() {
            return Ok(Vec::new());
        }
        let strict = options.truncation.unwrap_or(self.truncation) == Truncation::Error;
        let prefix_len = prefix.as_ref().map(|prefix| prefix.dims()[0]).unwrap_or(0);
        let max_len = self
            .session
            .text_limit(modality, prefix_len, self.max_length)?;
        let max_state = options.max_state_tokens.unwrap_or(self.max_state_tokens);
        let calibrate = modality == Modality::Text;
        let mut answers = Vec::with_capacity(questions.len());
        for question in questions {
            let encoded = encode(
                &self.session.tokenizer,
                state,
                question,
                max_len,
                modality,
                max_state,
                strict,
            )?;
            let logits = self.session.question_logits(
                prefix.as_ref(),
                &encoded.ids,
                encoded.qtype,
                &encoded.markers,
            )?;
            let temperature =
                self.temperature_for(options, kind_name(question), encoded.n_options, calibrate)?;
            answers.push(decode_answer(question, &logits, temperature));
        }
        Ok(answers)
    }

    fn temperature_for(
        &self,
        options: &DecideOptions,
        kind: &str,
        n_options: usize,
        calibrate: bool,
    ) -> Result<f32, Error> {
        let temperature = match options.temperature.or(self.user_temperature) {
            Some(temperature) => temperature,
            None if calibrate => self.session.calibrated(kind, n_options),
            None => 1.0,
        };
        check_temperature(temperature)?;
        Ok(temperature)
    }
}

struct CachedPrefix {
    stamp: u64,
    prefix: Tensor<Wgpu, 2>,
}

/// Image decisions. The vision tower output is reused when the same image is
/// scored again. Each call is still one question, and it does not see other
/// questions.
pub struct VisionDecision {
    session: Session<Wgpu>,
    prefix: RefCell<Option<CachedPrefix>>,
}

impl VisionDecision {
    /// Load d1-omni-600M with its vision tower from `dir`.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, Error> {
        Ok(Self {
            session: Session::load(dir.as_ref(), Which::Vision)?,
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

    pub fn noul(
        &self,
        image: &RgbImage,
        state: &str,
        statement: &str,
    ) -> Result<NoulAnswer, Error> {
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
        validate_question(&question, "vision", limits())?;
        let stamp = image_stamp(image);
        let cached = self
            .prefix
            .borrow()
            .as_ref()
            .map(|cached| cached.stamp == stamp)
            .unwrap_or(false);
        let logits = if cached {
            let slot = self.prefix.borrow();
            let prefix = &slot.as_ref().expect("cache checked").prefix;
            self.logits_for(state, &question, prefix)?
        } else {
            let prefix = self.session.vision_prefix(image)?;
            let logits = self.logits_for(state, &question, &prefix)?;
            *self.prefix.borrow_mut() = Some(CachedPrefix { stamp, prefix });
            logits
        };
        Ok(decode_answer(&question, &logits, 1.0))
    }

    fn logits_for(
        &self,
        state: &str,
        question: &Question,
        prefix: &Tensor<Wgpu, 2>,
    ) -> Result<Vec<f32>, Error> {
        let max_len = self
            .session
            .text_limit(Modality::Vision, prefix.dims()[0], None)?;
        let encoded = encode(
            &self.session.tokenizer,
            state,
            question,
            max_len,
            Modality::Vision,
            DEFAULT_MAX_STATE,
            false,
        )?;
        self.session
            .question_logits(Some(prefix), &encoded.ids, encoded.qtype, &encoded.markers)
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
        assert!(info.weights_dir.ends_with("d1-omni-600M"));
    }

    #[test]
    fn limits_follow_the_d1_head() {
        let limits = limits();
        assert_eq!(limits.min_choice_options, 2);
        assert_eq!(limits.max_choice_options, 255);
        assert_eq!(limits.max_score_levels, 10);
        let err = validate_question(&score("Rate", &["only"]), "q", limits).unwrap_err();
        assert!(matches!(err, Error::OptionCount { min: 2, got: 1, .. }));
        let err = validate_question(&choice("Pick", &["only"], None), "q", limits).unwrap_err();
        assert!(matches!(err, Error::OptionCount { min: 2, got: 1, .. }));
        let many = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"];
        let err = validate_question(&score("Rate", &many), "q", limits).unwrap_err();
        assert!(matches!(
            err,
            Error::OptionCount {
                max: 10,
                got: 11,
                ..
            }
        ));
    }
}
