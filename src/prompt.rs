//! d1 prompt packing, from `prompt.py`.
//!
//! One question is one sequence:
//! `<bos> <state> state <q> instructions <opt> <mask> option </opt> ... <decide>`.
//! The hidden state at each `<mask>` is the score for that option.

use crate::questions::Question;
use crate::tokenize::escape_delimiters;
use crate::Error;

pub const Q_CHOICE: usize = 0;
pub const Q_SCORE: usize = 1;
pub const Q_NOUL: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    Text,
    Audio,
    Vision,
}

#[derive(Debug)]
pub struct Encoded {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    pub qtype: usize,
    pub n_options: usize,
}

pub trait TextEncoder {
    fn piece(&self, text: &str) -> Vec<u32>;
    fn bos(&self) -> u32;
    fn delim_state(&self) -> u32;
    fn delim_q(&self) -> u32;
    fn delim_opt(&self) -> u32;
    fn delim_opt_end(&self) -> u32;
    fn delim_decide(&self) -> u32;
    fn marker(&self) -> u32;
}

pub fn encode(
    encoder: &impl TextEncoder,
    state: &str,
    question: &Question,
    max_len: usize,
    modality: Modality,
    max_state_tokens: usize,
    strict: bool,
) -> Result<Encoded, Error> {
    let options = option_texts(question, modality);
    let n = options.len();
    let budget = 96.max((n * 24 + 32).min(max_len / 2));
    let per = ((budget as isize - 3 * n as isize) / n as isize).max(2) as usize;

    let mut body = Vec::new();
    body.push(encoder.delim_q());
    body.extend(encoder.piece(&escape_delimiters(question.instructions())));
    body.truncate(16.max(budget));

    let mut markers = Vec::with_capacity(n);
    for text in &options {
        markers.push(body.len() + 1);
        body.push(encoder.delim_opt());
        body.push(encoder.marker());
        let mut piece = encoder.piece(&escape_delimiters(&format!(" {text}")));
        piece.truncate(per);
        body.extend(piece);
        body.push(encoder.delim_opt_end());
    }
    body.push(encoder.delim_decide());

    let room = max_len
        .saturating_sub(body.len())
        .saturating_sub(2)
        .min(max_state_tokens);
    let mut state_body = encoder.piece(&escape_delimiters(state));
    let state_tokens = state_body.len();
    if strict && state_tokens > room {
        return Err(Error::Truncated {
            state_tokens,
            kept: room,
        });
    }
    state_body.truncate(room);
    let mut state_ids = Vec::with_capacity(state_body.len() + 1);
    state_ids.push(encoder.delim_state());
    state_ids.extend(state_body);
    let shift = 1 + state_ids.len();
    for marker in &mut markers {
        *marker += shift;
    }

    let mut ids = Vec::with_capacity(1 + state_ids.len() + body.len());
    ids.push(encoder.bos());
    ids.extend(state_ids);
    ids.extend(body);
    ids.truncate(max_len);
    if markers.last().copied().unwrap_or(usize::MAX) >= max_len {
        return Err(Error::Context {
            message: "the options do not fit in the context".into(),
        });
    }
    Ok(Encoded {
        ids,
        markers,
        qtype: qtype(question),
        n_options: n,
    })
}

pub fn qtype(question: &Question) -> usize {
    match question {
        Question::Choice { .. } => Q_CHOICE,
        Question::Score { .. } => Q_SCORE,
        Question::Noul { .. } => Q_NOUL,
    }
}

pub fn kind_name(question: &Question) -> &'static str {
    question.type_name()
}

fn option_texts(question: &Question, modality: Modality) -> Vec<String> {
    match question {
        Question::Choice {
            options,
            descriptions,
            ..
        } => options
            .iter()
            .enumerate()
            .map(|(index, key)| {
                let description = descriptions.get(key).map(String::as_str).unwrap_or("");
                let text = if description.is_empty() {
                    key.clone()
                } else {
                    description.to_string()
                };
                if modality == Modality::Audio {
                    format!("option_{index:03}: {text}")
                } else if description.is_empty() {
                    key.clone()
                } else {
                    format!("{key}: {description}")
                }
            })
            .collect(),
        Question::Score { levels, .. } => levels
            .iter()
            .enumerate()
            .map(|(index, level)| format!("level {index}: {level}"))
            .collect(),
        Question::Noul { .. } => {
            if modality == Modality::Text {
                vec![
                    "false: no, the statement does not hold".into(),
                    "true: yes, the statement holds".into(),
                ]
            } else {
                vec!["false: no".into(), "true: yes".into()]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::questions::{choice, noul, score};

    struct Bytes;

    impl TextEncoder for Bytes {
        fn piece(&self, text: &str) -> Vec<u32> {
            text.bytes().map(|byte| byte as u32).collect()
        }
        fn bos(&self) -> u32 {
            1
        }
        fn delim_state(&self) -> u32 {
            17
        }
        fn delim_q(&self) -> u32 {
            18
        }
        fn delim_opt(&self) -> u32 {
            19
        }
        fn delim_opt_end(&self) -> u32 {
            20
        }
        fn delim_decide(&self) -> u32 {
            21
        }
        fn marker(&self) -> u32 {
            16
        }
    }

    fn pack(state: &str, question: Question, max_len: usize, modality: Modality) -> Encoded {
        encode(
            &Bytes,
            state,
            &question,
            max_len,
            modality,
            usize::MAX,
            false,
        )
        .unwrap()
    }

    #[test]
    fn choice_matches_the_reference_ids() {
        let encoded = pack(
            "the state",
            choice(
                "Pick one.",
                &["zeta", "alpha"],
                Some(&[("zeta", "the letter z")]),
            ),
            16384,
            Modality::Text,
        );
        assert_eq!(encoded.markers, vec![22, 44]);
        assert_eq!(encoded.qtype, Q_CHOICE);
        assert_eq!(
            encoded.ids,
            vec![
                1, 17, 116, 104, 101, 32, 115, 116, 97, 116, 101, 18, 80, 105, 99, 107, 32, 111,
                110, 101, 46, 19, 16, 32, 122, 101, 116, 97, 58, 32, 116, 104, 101, 32, 108, 101,
                116, 116, 101, 114, 32, 122, 20, 19, 16, 32, 97, 108, 112, 104, 97, 20, 21
            ]
        );
    }

    #[test]
    fn noul_score_and_audio_match_the_reference_ids() {
        let yes_no = pack("s", noul("Is it raining?"), 16384, Modality::Text);
        assert_eq!(yes_no.markers, vec![19, 61]);
        assert_eq!(yes_no.qtype, Q_NOUL);
        assert_eq!(
            yes_no.ids,
            vec![
                1, 17, 115, 18, 73, 115, 32, 105, 116, 32, 114, 97, 105, 110, 105, 110, 103, 63,
                19, 16, 32, 102, 97, 108, 115, 101, 58, 32, 110, 111, 44, 32, 116, 104, 101, 32,
                115, 116, 97, 116, 101, 109, 101, 110, 116, 32, 100, 111, 101, 115, 32, 110, 111,
                116, 32, 104, 111, 108, 100, 20, 19, 16, 32, 116, 114, 117, 101, 58, 32, 121, 101,
                115, 44, 32, 116, 104, 101, 32, 115, 116, 97, 116, 101, 109, 101, 110, 116, 32,
                104, 111, 108, 100, 115, 20, 21
            ]
        );

        let vision = pack("s", noul("Is it raining?"), 896, Modality::Vision);
        assert_eq!(vision.markers, vec![19, 32]);
        assert!(vision
            .ids
            .windows(4)
            .any(|window| window == [102, 97, 108, 115]));

        let scored = pack(
            "s",
            score("How hot?", &["cold", "warm", "hot"]),
            200,
            Modality::Text,
        );
        assert_eq!(scored.markers, vec![13, 30, 47]);
        assert_eq!(scored.qtype, Q_SCORE);

        let spoken = pack(
            "{}",
            choice("Pick", &["a", "b"], Some(&[("a", "A"), ("b", "B")])),
            15360,
            Modality::Audio,
        );
        assert_eq!(spoken.markers, vec![10, 27]);
        assert_eq!(
            spoken.ids,
            vec![
                1, 17, 123, 125, 18, 80, 105, 99, 107, 19, 16, 32, 111, 112, 116, 105, 111, 110,
                95, 48, 48, 48, 58, 32, 65, 20, 19, 16, 32, 111, 112, 116, 105, 111, 110, 95, 48,
                48, 49, 58, 32, 66, 20, 21
            ]
        );
    }

    #[test]
    fn strict_truncation_reports_the_kept_tokens() {
        let err = encode(
            &Bytes,
            "hello",
            &noul("Is it raining?"),
            16384,
            Modality::Text,
            2,
            true,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Error::Truncated {
                state_tokens: 5,
                kept: 2
            }
        ));
    }
}
