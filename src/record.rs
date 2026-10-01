//! The Clef-Flash record encoding.
//!
//! This is a port of `encode_record` from `joint_schema_model.py` in
//! `Cloudflare/clef-flash`. One record is:
//!
//! ```text
//! <|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n
//! [media]
//! {state}
//! \n\nSCHEMA FIELDS:\n
//!   \nFIELD {n}\nID: {id}\nTYPE: {type}\nINSTRUCTION: {instructions}
//!   \nALLOWED OPTIONS:\n
//!   OPTION {m}: {"option_id":...,"description":...}\n
//!   END FIELD\n
//! \n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:
//! ```
//!
//! The joint head reads the instruction span and each option span, so the
//! layout records both. Chunks are tokenized separately, in the same chunks as
//! the reference implementation, because token boundaries are part of the
//! encoding.

use crate::questions::Question;
use crate::Error;

pub const SYSTEM_PROMPT: &str = "Read the complete state and schema. Decide every field jointly. \
Each answer must be exactly one of that field's allowed options.";

pub const NOUL_TRUE_DESCRIPTION: &str = "The proposition is true or the answer is yes.";
pub const NOUL_FALSE_DESCRIPTION: &str = "The proposition is false or the answer is no.";

/// Tokenizes template text (`raw`) and user text (`user`, escaped).
pub trait TemplateEncoder {
    fn raw(&self, text: &str) -> Vec<u32>;
    fn user(&self, text: &str) -> Vec<u32>;
}

/// One question's spans into [`Layout::ids`].
#[derive(Debug, Clone)]
pub struct SpanQuestion {
    /// 0 = noul, 1 = choice, 2 = score, as the head's type embedding counts.
    pub question_type: usize,
    /// The instruction tokens.
    pub question_span: (usize, usize),
    /// The option-semantics tokens, one span per encoded option.
    pub option_spans: Vec<(usize, usize)>,
    /// `user_order[u]` is the encoded option index for the caller's label `u`.
    /// Choice options are encoded sorted by label; noul is encoded true-then-false
    /// while the crate reports `[no, yes]`.
    pub user_order: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct Layout {
    pub ids: Vec<u32>,
    pub questions: Vec<SpanQuestion>,
    /// Tokens before the schema: system template, media, and the kept state.
    pub prefix_len: usize,
    /// Index of the first image slot, when media ids were provided.
    pub image_at: Option<usize>,
    /// Tokens the full state needs, before any truncation.
    #[allow(dead_code)]
    pub state_tokens: usize,
    /// State tokens kept after `max_state_tokens` and the length budget.
    #[allow(dead_code)]
    pub kept_state: usize,
}

pub fn encode_record(
    encoder: &impl TemplateEncoder,
    state: &str,
    media_ids: Option<&[u32]>,
    questions: &[Question],
    question_ids: &[String],
    max_length: usize,
    max_state_tokens: usize,
    strict: bool,
) -> Result<Layout, Error> {
    debug_assert_eq!(questions.len(), question_ids.len());

    let mut schema_ids = encoder.raw("\n\nSCHEMA FIELDS:\n");
    let mut spans = Vec::with_capacity(questions.len());
    for (index, (question, question_id)) in questions.iter().zip(question_ids).enumerate() {
        schema_ids.extend(encoder.raw(&format!(
            "\nFIELD {n}\nID: {id}\nTYPE: {kind}\nINSTRUCTION: ",
            n = index + 1,
            id = crate::tokenize::escape_delimiters(question_id),
            kind = question.type_name(),
        )));
        let question_start = schema_ids.len();
        schema_ids.extend(encoder.user(question.instructions()));
        let question_end = schema_ids.len();
        schema_ids.extend(encoder.raw("\nALLOWED OPTIONS:\n"));

        let (options, user_order, question_type) = encoded_options(question);
        let mut option_spans = Vec::with_capacity(options.len());
        for (option_index, semantics) in options.iter().enumerate() {
            schema_ids.extend(encoder.raw(&format!("OPTION {}: ", option_index + 1)));
            let option_start = schema_ids.len();
            schema_ids.extend(encoder.user(semantics));
            option_spans.push((option_start, schema_ids.len()));
            schema_ids.extend(encoder.raw("\n"));
        }
        schema_ids.extend(encoder.raw("END FIELD\n"));
        spans.push(SpanQuestion {
            question_type,
            question_span: (question_start, question_end),
            option_spans,
            user_order,
        });
    }

    let mut prefix_ids = encoder.raw(&format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
    ));
    let image_at = media_ids.map(|media| {
        let at = prefix_ids.len() + 1;
        prefix_ids.extend_from_slice(media);
        at
    });
    let suffix_ids =
        encoder.raw("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:");

    let state_ids = encoder.user(state);
    let state_tokens = state_ids.len();
    let fixed_length = prefix_ids.len() + schema_ids.len() + suffix_ids.len();
    if fixed_length > max_length {
        return Err(Error::Context {
            message: format!(
                "The schema needs {fixed_length} tokens before the state; the maximum is {max_length}. Shorten the instructions or options, or ask fewer questions per call."
            ),
        });
    }
    let kept_state = state_tokens
        .min(max_state_tokens)
        .min(max_length - fixed_length);
    if strict && kept_state < state_tokens {
        return Err(Error::Truncated {
            state_tokens,
            kept: kept_state,
        });
    }

    let schema_offset = prefix_ids.len() + kept_state;
    let questions = spans
        .into_iter()
        .map(|span| SpanQuestion {
            question_type: span.question_type,
            question_span: (
                span.question_span.0 + schema_offset,
                span.question_span.1 + schema_offset,
            ),
            option_spans: span
                .option_spans
                .into_iter()
                .map(|(start, end)| (start + schema_offset, end + schema_offset))
                .collect(),
            user_order: span.user_order,
        })
        .collect();

    let mut ids = prefix_ids;
    ids.extend(state_ids.into_iter().take(kept_state));
    let prefix_len = ids.len();
    ids.extend(schema_ids);
    ids.extend(suffix_ids);

    Ok(Layout {
        ids,
        questions,
        prefix_len,
        image_at,
        state_tokens,
        kept_state,
    })
}

/// The option-semantics JSON strings in encoded order, the user-order mapping,
/// and the question type id.
fn encoded_options(question: &Question) -> (Vec<String>, Vec<usize>, usize) {
    match question {
        Question::Noul { .. } => {
            let options = vec![
                semantics("true", Some(NOUL_TRUE_DESCRIPTION)),
                semantics("false", Some(NOUL_FALSE_DESCRIPTION)),
            ];
            // The crate's noul labels are [no, yes]; encoded order is [true, false].
            (options, vec![1, 0], 0)
        }
        Question::Choice {
            options,
            descriptions,
            ..
        } => {
            let mut sorted: Vec<&String> = options.iter().collect();
            sorted.sort();
            let encoded = sorted
                .iter()
                .map(|label| {
                    let description = descriptions
                        .get(*label)
                        .filter(|text| !text.is_empty())
                        .map(String::as_str);
                    semantics(label, description)
                })
                .collect();
            let user_order = options
                .iter()
                .map(|label| sorted.iter().position(|item| *item == label).unwrap())
                .collect();
            (encoded, user_order, 1)
        }
        Question::Score { levels, .. } => {
            let encoded = levels
                .iter()
                .enumerate()
                .map(|(index, level)| semantics(&index.to_string(), Some(level)))
                .collect();
            let user_order = (0..levels.len()).collect();
            (encoded, user_order, 2)
        }
    }
}

/// `{"description":...,"option_id":...}` with keys sorted and compact
/// separators, exactly as the reference `render` emits it.
fn semantics(option_id: &str, description: Option<&str>) -> String {
    let mut out = String::from("{");
    if let Some(description) = description {
        out.push_str("\"description\":");
        out.push_str(&json_string(description));
        out.push(',');
    }
    out.push_str("\"option_id\":");
    out.push_str(&json_string(option_id));
    out.push('}');
    out
}

/// A JSON string literal matching Python's `json.dumps(ensure_ascii=False)`.
fn json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ch if (ch as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", ch as u32));
            }
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::questions::{choice, noul, score};

    /// One token per byte, so spans can be checked by hand. `<|...|>` named
    /// delimiters count as one token, like the real added tokens.
    struct ByteEncoder;

    fn byte_tokens(text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let mut ids = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'<' && i + 1 < bytes.len() && bytes[i + 1] == b'|' {
                if let Some(end) = text[i..].find("|>") {
                    ids.push(1_000_000);
                    i += end + 2;
                    continue;
                }
            }
            ids.push(bytes[i] as u32);
            i += 1;
        }
        ids
    }

    impl TemplateEncoder for ByteEncoder {
        fn raw(&self, text: &str) -> Vec<u32> {
            byte_tokens(text)
        }
        fn user(&self, text: &str) -> Vec<u32> {
            byte_tokens(&crate::tokenize::escape_delimiters(text))
        }
    }

    fn ids_of(layout: &Layout, span: (usize, usize)) -> String {
        layout.ids[span.0..span.1]
            .iter()
            .map(|id| char::from_u32(*id).unwrap())
            .collect()
    }

    #[test]
    fn json_strings_match_python() {
        assert_eq!(json_string("a\"b\\c\nd"), "\"a\\\"b\\\\c\\nd\"");
        assert_eq!(json_string("café"), "\"café\"");
        assert_eq!(json_string("\u{1}"), "\"\\u0001\"");
    }

    #[test]
    fn choice_options_are_sorted_and_mapped_back() {
        let layout = encode_record(
            &ByteEncoder,
            "the state",
            None,
            &[choice(
                "Pick one.",
                &["zeta", "alpha"],
                Some(&[("zeta", "the letter z")]),
            )],
            &["q1".to_string()],
            16384,
            16384,
            false,
        )
        .unwrap();
        let question = &layout.questions[0];
        assert_eq!(question.question_type, 1);
        assert_eq!(ids_of(&layout, question.question_span), "Pick one.");
        assert_eq!(
            ids_of(&layout, question.option_spans[0]),
            "{\"option_id\":\"alpha\"}"
        );
        assert_eq!(
            ids_of(&layout, question.option_spans[1]),
            "{\"description\":\"the letter z\",\"option_id\":\"zeta\"}"
        );
        // User order [zeta, alpha] maps onto encoded order [alpha, zeta].
        assert_eq!(question.user_order, vec![1, 0]);
    }

    #[test]
    fn noul_encodes_true_then_false() {
        let layout = encode_record(
            &ByteEncoder,
            "s",
            None,
            &[noul("Is it raining?")],
            &["q1".to_string()],
            16384,
            16384,
            false,
        )
        .unwrap();
        let question = &layout.questions[0];
        assert_eq!(question.question_type, 0);
        assert!(ids_of(&layout, question.option_spans[0]).contains("\"option_id\":\"true\""));
        assert!(ids_of(&layout, question.option_spans[1]).contains("\"option_id\":\"false\""));
        // Crate labels are [no, yes].
        assert_eq!(question.user_order, vec![1, 0]);
    }

    #[test]
    fn score_levels_become_indexed_options() {
        let layout = encode_record(
            &ByteEncoder,
            "s",
            None,
            &[score("How hot?", &["cold", "warm", "hot"])],
            &["q1".to_string()],
            16384,
            16384,
            false,
        )
        .unwrap();
        let question = &layout.questions[0];
        assert_eq!(question.question_type, 2);
        assert_eq!(
            ids_of(&layout, question.option_spans[1]),
            "{\"description\":\"warm\",\"option_id\":\"1\"}"
        );
        assert_eq!(question.user_order, vec![0, 1, 2]);
    }

    #[test]
    fn layout_orders_prefix_state_schema_suffix() {
        let layout = encode_record(
            &ByteEncoder,
            "STATE-TEXT",
            None,
            &[noul("Check.")],
            &["q1".to_string()],
            16384,
            16384,
            false,
        )
        .unwrap();
        let text: String = layout
            .ids
            .iter()
            .map(|id| char::from_u32((*id).min(0x7f)).unwrap())
            .collect();
        let state_at = text.find("STATE-TEXT").unwrap();
        let schema_at = text.find("SCHEMA FIELDS").unwrap();
        let decisions_at = text.find("JOINT SCHEMA DECISIONS:").unwrap();
        assert!(state_at < schema_at && schema_at < decisions_at);
        assert!(text.ends_with("JOINT SCHEMA DECISIONS:"));
        assert_eq!(layout.prefix_len, state_at + "STATE-TEXT".len());
    }

    #[test]
    fn media_ids_go_between_template_and_state() {
        let media = vec![900, 901, 901, 902];
        let layout = encode_record(
            &ByteEncoder,
            "state",
            Some(&media),
            &[noul("Check.")],
            &["q1".to_string()],
            16384,
            16384,
            false,
        )
        .unwrap();
        let image_at = layout.image_at.unwrap();
        assert_eq!(layout.ids[image_at - 1], 900);
        assert_eq!(layout.ids[image_at], 901);
    }

    #[test]
    fn state_is_cut_to_the_budget() {
        let layout = encode_record(
            &ByteEncoder,
            "abcdefgh",
            None,
            &[noul("Check.")],
            &["q1".to_string()],
            16384,
            3,
            false,
        )
        .unwrap();
        assert_eq!(layout.kept_state, 3);
        assert_eq!(layout.state_tokens, 8);
    }

    #[test]
    fn strict_truncation_fails() {
        let err = encode_record(
            &ByteEncoder,
            "abcdefgh",
            None,
            &[noul("Check.")],
            &["q1".to_string()],
            16384,
            3,
            true,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Error::Truncated {
                state_tokens: 8,
                kept: 3
            }
        ));
    }

    #[test]
    fn oversized_schema_is_an_error() {
        let err = encode_record(
            &ByteEncoder,
            "s",
            None,
            &[noul("Check.")],
            &["q1".to_string()],
            64,
            16384,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, Error::Context { .. }));
    }
}
