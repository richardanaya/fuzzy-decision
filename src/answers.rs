use std::collections::BTreeMap;

use crate::questions::{question_labels, Question};

#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    Choice {
        choice: String,
        confidence: f32,
        probabilities: BTreeMap<String, f32>,
    },
    Score {
        score: f32,
        normalized: f32,
        level: String,
        confidence: f32,
        probabilities: BTreeMap<String, f32>,
    },
    Noul {
        answer: bool,
        probability: f32,
        confidence: f32,
    },
}

pub fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let temperature = if temperature == 0.0 { 1.0 } else { temperature };
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let scaled: Vec<f32> = logits
        .iter()
        .map(|x| ((x - max) / temperature).exp())
        .collect();
    let sum: f32 = scaled.iter().sum();
    scaled.into_iter().map(|x| x / sum).collect()
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

fn to_record(keys: &[String], values: &[f32]) -> BTreeMap<String, f32> {
    keys.iter().cloned().zip(values.iter().copied()).collect()
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChoiceAnswer {
    pub choice: String,
    pub confidence: f32,
    pub probabilities: BTreeMap<String, f32>,
}

impl ChoiceAnswer {
    pub fn probability(&self, option: &str) -> Result<f32, crate::Error> {
        self.probabilities
            .get(option)
            .copied()
            .ok_or_else(|| crate::Error::UnknownOption {
                option: option.to_string(),
            })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScoreAnswer {
    pub score: f32,
    pub normalized: f32,
    pub level: String,
    pub confidence: f32,
    pub probabilities: BTreeMap<String, f32>,
}

impl ScoreAnswer {
    pub fn probability(&self, level: &str) -> Result<f32, crate::Error> {
        self.probabilities
            .get(level)
            .copied()
            .ok_or_else(|| crate::Error::UnknownOption {
                option: level.to_string(),
            })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NoulAnswer {
    pub answer: bool,
    pub probability: f32,
    pub confidence: f32,
}

pub fn decode_answer(question: &Question, logits: &[f32], temperature: f32) -> Answer {
    let probabilities = softmax(logits, temperature);
    let labels = question_labels(question);
    match question {
        Question::Choice { .. } => {
            let best = argmax(&probabilities);
            Answer::Choice {
                choice: labels[best].clone(),
                confidence: probabilities[best],
                probabilities: to_record(&labels, &probabilities),
            }
        }
        Question::Score { .. } => {
            let expected = probabilities
                .iter()
                .enumerate()
                .map(|(index, p)| p * index as f32)
                .sum::<f32>();
            let best = argmax(&probabilities);
            let last = labels.len().saturating_sub(1);
            let nearest = (expected.round() as usize).min(last);
            Answer::Score {
                score: expected,
                normalized: if last > 0 {
                    expected / last as f32
                } else {
                    0.0
                },
                level: labels[nearest].clone(),
                confidence: probabilities[best],
                probabilities: to_record(&labels, &probabilities),
            }
        }
        Question::Noul { .. } => {
            let probability = probabilities.get(1).copied().unwrap_or(0.0);
            Answer::Noul {
                answer: probability >= 0.5,
                probability,
                confidence: probability.max(1.0 - probability),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_is_uniform_for_tied_logits() {
        let p = softmax(&[1.0, 1.0, 1.0], 1.0);
        for value in p {
            assert!((value - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn temperature_flattens() {
        let sharp = softmax(&[0.0, 4.0], 0.5);
        let flat = softmax(&[0.0, 4.0], 4.0);
        assert!(sharp[1] > flat[1]);
    }

    #[test]
    fn choice_probability_names_a_missing_option() {
        let answer = ChoiceAnswer {
            choice: "a".into(),
            confidence: 1.0,
            probabilities: BTreeMap::from([("a".into(), 1.0)]),
        };
        let err = answer.probability("b").unwrap_err();
        assert_eq!(err.to_string(), "no option named \"b\"");
    }
}
