//! Qwen3 tokenizer and the Kev packing used by `jaredpalmer/kev`.

use std::path::Path;

use tokenizers::Tokenizer;

use crate::qwen::Packed;
use crate::questions::question_option_texts;
use crate::Question;

const SPECIAL: [&str; 5] = [
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
];

pub struct HfTokenizer {
    inner: Tokenizer,
    state: u32,
    question: u32,
    option_start: u32,
    option_end: u32,
    decide: u32,
}

impl HfTokenizer {
    pub fn open(path: &Path) -> Result<Self, String> {
        let inner = Tokenizer::from_file(path)
            .map_err(|err| format!("read tokenizer {}: {err}", path.display()))?;
        let id = |name: &str| {
            inner
                .token_to_id(name)
                .ok_or_else(|| format!("tokenizer is missing {name}"))
        };
        Ok(Self {
            state: id(SPECIAL[0])?,
            question: id(SPECIAL[1])?,
            option_start: id(SPECIAL[2])?,
            option_end: id(SPECIAL[3])?,
            decide: id(SPECIAL[4])?,
            inner,
        })
    }

    pub fn count(&self, text: &str) -> usize {
        self.encode_user(text).len()
    }

    pub fn encode_user(&self, text: &str) -> Vec<u32> {
        let escaped = escape_delimiters(text);
        self.inner
            .encode(escaped, false)
            .map(|enc| enc.get_ids().to_vec())
            .unwrap_or_default()
    }

    /// Pack state plus questions the way Kev's `encode` does.
    /// `max_state` includes the state delimiter. `max_row` is state plus one branch.
    pub fn pack(
        &self,
        state: &str,
        questions: &[Question],
        max_state: usize,
        max_row: usize,
        strict: bool,
    ) -> Result<Packed, crate::Error> {
        let state_tokens = self.encode_user(state);
        if strict && state_tokens.len() + 1 > max_state {
            return Err(crate::Error::Truncated {
                state_tokens: state_tokens.len() + 1,
                kept: max_state,
            });
        }
        let kept_state = state_tokens.len().min(max_state.saturating_sub(1));
        let mut ids = Vec::new();
        let mut pos = Vec::new();
        let mut seg = Vec::new();
        ids.push(self.state);
        ids.extend(state_tokens.iter().take(kept_state).copied());
        let state_len = ids.len();
        pos.extend(0..state_len as i32);
        seg.resize(state_len, 0);

        let mut readouts = Vec::with_capacity(questions.len());
        for (index, question) in questions.iter().enumerate() {
            let question_id = (index + 1) as i32;
            let mut branch = vec![self.question];
            branch.extend(self.encode_user(question.instructions()));
            let mut ends = Vec::new();
            for option in question_option_texts(question) {
                branch.push(self.option_start);
                branch.extend(self.encode_user(&option));
                branch.push(self.option_end);
                ends.push(branch.len() - 1);
            }
            branch.push(self.decide);
            if state_len + branch.len() > max_row {
                return Err(crate::Error::Row {
                    state_tokens: state_len,
                    question_tokens: branch.len(),
                    limit: max_row,
                });
            }
            let base = ids.len();
            let start_pos = state_len as i32;
            for (offset, id) in branch.iter().copied().enumerate() {
                ids.push(id);
                pos.push(start_pos + offset as i32);
                seg.push(question_id);
            }
            let decide = base + branch.len() - 1;
            readouts.push((decide, ends.into_iter().map(|end| base + end).collect()));
        }

        Ok(Packed {
            ids: ids.into_iter().map(|id| id as i32).collect(),
            pos,
            seg,
            readouts,
        })
    }
}

pub trait TokenCounter {
    fn count_tokens(&self, text: &str) -> usize;
}

impl TokenCounter for HfTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        self.count(text)
    }
}

fn escape_delimiters(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' && i + 1 < bytes.len() && bytes[i + 1] == b'|' {
            if let Some(end) = find_close(bytes, i + 2) {
                let name = &text[i + 2..end];
                if name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
                    out.push_str("<¦");
                    out.push_str(name);
                    out.push_str("¦>");
                    i = end + 2;
                    continue;
                }
            }
        }
        let ch = text[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn find_close(bytes: &[u8], start: usize) -> Option<usize> {
    let mut i = start;
    while i + 1 < bytes.len() {
        if bytes[i] == b'|' && bytes[i + 1] == b'>' {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_special_brackets() {
        assert_eq!(
            escape_delimiters("see <|fim_prefix|> now"),
            "see <¦fim_prefix¦> now"
        );
    }
}
