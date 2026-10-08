//! CPU decisions on Burn's NdArray backend.
//!
//! Enable the `cpu` feature and use [`CpuDecision`]. The default
//! [`crate::FuzzyDecision`] API stays on WGPU. This path is for tests and
//! machines without a GPU: it keeps the checkpoint in host memory.

use std::path::Path;

use burn::backend::NdArray;
use burn::tensor::Tensor;

use crate::answers::{decode_answer, Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer};
use crate::check_temperature;
use crate::model::{Session, Which};
use crate::prompt::{encode, kind_name, Modality};
use crate::questions::{
    choice, noul, score, validate_question, Question, QuestionLimits, Truncation,
};
use crate::wav::AudioClip;
use crate::{limits, DecideOptions, Error, DEFAULT_MAX_STATE};

/// d1-omni-600M on the CPU. Same questions and answers as [`crate::FuzzyDecision`].
pub struct CpuDecision {
    limits: QuestionLimits,
    user_temperature: Option<f32>,
    max_state_tokens: usize,
    max_length: Option<usize>,
    truncation: Truncation,
    session: Session<NdArray>,
}

impl CpuDecision {
    /// Load the snapshot in `dir` onto the CPU. The directory needs
    /// `tokenizer.json`, `config.json`, and `model.safetensors`.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, Error> {
        let session = Session::<NdArray>::load(dir.as_ref(), Which::TextAudio)?;
        Ok(Self {
            limits: limits(),
            user_temperature: None,
            max_state_tokens: DEFAULT_MAX_STATE,
            max_length: None,
            truncation: Truncation::Cut,
            session,
        })
    }

    pub fn noul(&self, state: &str, statement: impl Into<String>) -> Result<NoulAnswer, Error> {
        match self.one(
            state,
            noul(statement),
            DecideOptions::default(),
            None,
            Modality::Text,
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

    pub fn choice(
        &self,
        state: &str,
        instructions: impl Into<String>,
        options: &[&str],
    ) -> Result<ChoiceAnswer, Error> {
        match self.one(
            state,
            choice(instructions, options, None),
            DecideOptions::default(),
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
        match self.one(
            state,
            score(instructions, levels),
            DecideOptions::default(),
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
    ) -> Result<ChoiceAnswer, Error> {
        match self.one(
            state,
            choice(instructions, options, None),
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

    fn one(
        &self,
        state: &str,
        question: Question,
        options: DecideOptions,
        prefix: Option<Tensor<NdArray, 2>>,
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
        prefix: Option<Tensor<NdArray, 2>>,
        modality: Modality,
    ) -> Result<Vec<Answer>, Error> {
        for (index, question) in questions.iter().enumerate() {
            validate_question(question, &format!("#{index}"), self.limits)?;
        }
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
                self.temperature_for(&options, kind_name(question), encoded.n_options, calibrate)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ready;
    use crate::prompt::encode;
    use crate::questions::{choice, noul, score};
    use crate::tokenize::HfTokenizer;
    use crate::Question;

    fn manifest_dir() -> &'static Path {
        Path::new(env!("CARGO_MANIFEST_DIR"))
    }

    fn close(got: f32, expected: f32, what: &str) {
        let gap = (got - expected).abs();
        eprintln!("{what}: rust {got:.6}  python {expected:.6}  |diff| {gap:.3e}");
        assert!(gap < 1e-3, "{what} diverged by {gap}");
    }

    /// `speech.wav` is the public-domain clip File:En-us-hello.ogg, resampled
    /// to 16 kHz mono. `silence.wav` is one second of zeros.
    #[test]
    #[ignore = "loads models/d1-omni-600M on the CPU"]
    fn real_checkpoint_matches_python() {
        let dir = manifest_dir().join("models/d1-omni-600M");
        if !ready(&dir) {
            eprintln!("skipping: checkpoint is not in {}", dir.display());
            return;
        }
        let reference: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(manifest_dir().join("tests/data/cpu_reference.json"))
                .expect("python reference"),
        )
        .expect("reference json");
        let tokenizer = HfTokenizer::open(&dir.join("tokenizer.json")).expect("tokenizer");
        let state = reference["state"].as_str().unwrap();

        let noul_q = noul(reference["noul"]["statement"].as_str().unwrap());
        assert_ids(
            &tokenizer,
            state,
            &noul_q,
            16384,
            Modality::Text,
            &reference["noul"]["ids"],
        );
        let labels = strings(&reference["choice"]["labels"]);
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let choice_q = choice(
            reference["choice"]["instructions"].as_str().unwrap(),
            &label_refs,
            None,
        );
        assert_ids(
            &tokenizer,
            state,
            &choice_q,
            16384,
            Modality::Text,
            &reference["choice"]["ids"],
        );
        let levels = strings(&reference["score"]["levels"]);
        let level_refs: Vec<&str> = levels.iter().map(String::as_str).collect();
        let score_q = score(
            reference["score"]["instructions"].as_str().unwrap(),
            &level_refs,
        );
        assert_ids(
            &tokenizer,
            state,
            &score_q,
            16384,
            Modality::Text,
            &reference["score"]["ids"],
        );

        let audio_state = reference["audio_state"].as_str().unwrap();
        let audio_labels = strings(&reference["audio"]["labels"]);
        let audio_refs: Vec<&str> = audio_labels.iter().map(String::as_str).collect();
        let audio_q = choice(
            reference["audio"]["instructions"].as_str().unwrap(),
            &audio_refs,
            None,
        );
        for name in ["speech", "silence"] {
            assert_ids(
                &tokenizer,
                audio_state,
                &audio_q,
                15360,
                Modality::Audio,
                &reference["audio"][name]["ids"],
            );
        }

        let decider = CpuDecision::open(&dir).expect("cpu load");
        let noul_answer = decider
            .noul(state, reference["noul"]["statement"].as_str().unwrap())
            .unwrap();
        close(
            noul_answer.probability,
            reference["noul"]["yes"].as_f64().unwrap() as f32,
            "noul yes",
        );
        close(
            1.0 - noul_answer.probability,
            reference["noul"]["no"].as_f64().unwrap() as f32,
            "noul no",
        );

        let choice_answer = decider
            .choice(
                state,
                reference["choice"]["instructions"].as_str().unwrap(),
                &label_refs,
            )
            .unwrap();
        for (index, label) in label_refs.iter().enumerate() {
            let expected = reference["choice"]["probs"][index].as_f64().unwrap() as f32;
            close(choice_answer.probability(label).unwrap(), expected, label);
        }

        let score_answer = decider
            .score(
                state,
                reference["score"]["instructions"].as_str().unwrap(),
                &level_refs,
            )
            .unwrap();
        close(
            score_answer.score,
            reference["score"]["score"].as_f64().unwrap() as f32,
            "score",
        );
        for (index, level) in level_refs.iter().enumerate() {
            let expected = reference["score"]["probs"][index].as_f64().unwrap() as f32;
            close(score_answer.probability(level).unwrap(), expected, level);
        }

        for name in ["speech", "silence"] {
            let clip = AudioClip::from_wav(
                manifest_dir()
                    .join("tests/data")
                    .join(format!("{name}.wav")),
            )
            .unwrap();
            let answer = decider
                .choice_audio(
                    &clip,
                    audio_state,
                    reference["audio"]["instructions"].as_str().unwrap(),
                    &audio_refs,
                )
                .unwrap();
            for (index, label) in audio_refs.iter().enumerate() {
                let expected = reference["audio"][name]["probs"][index].as_f64().unwrap() as f32;
                close(
                    answer.probability(label).unwrap(),
                    expected,
                    &format!("{name}/{label}"),
                );
            }
        }
    }

    fn strings(value: &serde_json::Value) -> Vec<String> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item.as_str().unwrap().to_string())
            .collect()
    }

    fn assert_ids(
        tokenizer: &HfTokenizer,
        state: &str,
        question: &Question,
        max_len: usize,
        modality: Modality,
        expected: &serde_json::Value,
    ) {
        let encoded = encode(
            tokenizer,
            state,
            question,
            max_len,
            modality,
            DEFAULT_MAX_STATE,
            false,
        )
        .unwrap();
        let ids: Vec<u64> = expected
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item.as_u64().unwrap())
            .collect();
        let got: Vec<u64> = encoded.ids.iter().map(|id| *id as u64).collect();
        assert_eq!(got, ids, "{modality:?} token ids diverged");
    }
}
