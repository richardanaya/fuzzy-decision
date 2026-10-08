//! d1 tokenizer. User text is encoded with special tokens off, then the
//! prompt inserts `<|startoftext|>` and the reserved delimiters by id.

use std::path::Path;

use tokenizers::Tokenizer;

use crate::prompt::TextEncoder;

pub struct HfTokenizer {
    inner: Tokenizer,
    bos: u32,
    state: u32,
    q: u32,
    opt: u32,
    opt_end: u32,
    decide: u32,
    marker: u32,
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
            bos: id("<|startoftext|>")?,
            state: id("<|reserved_7|>")?,
            q: id("<|reserved_8|>")?,
            opt: id("<|reserved_9|>")?,
            opt_end: id("<|reserved_10|>")?,
            decide: id("<|reserved_11|>")?,
            marker: id("<|mask|>")?,
            inner,
        })
    }

    pub fn count(&self, text: &str) -> usize {
        self.piece(text).len()
    }

    fn encode_plain(&self, text: &str) -> Vec<u32> {
        self.inner
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .unwrap_or_default()
    }
}

impl TextEncoder for HfTokenizer {
    fn piece(&self, text: &str) -> Vec<u32> {
        self.encode_plain(&escape_delimiters(text))
    }

    fn bos(&self) -> u32 {
        self.bos
    }

    fn delim_state(&self) -> u32 {
        self.state
    }

    fn delim_q(&self) -> u32 {
        self.q
    }

    fn delim_opt(&self) -> u32 {
        self.opt
    }

    fn delim_opt_end(&self) -> u32 {
        self.opt_end
    }

    fn delim_decide(&self) -> u32 {
        self.decide
    }

    fn marker(&self) -> u32 {
        self.marker
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

/// `<|name|>` becomes `<¦name¦>`, so caller text cannot emit a delimiter.
pub fn escape_delimiters(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' && i + 1 < bytes.len() && bytes[i + 1] == b'|' {
            if let Some(end) = find_close(bytes, i + 2) {
                let name = &text[i + 2..end];
                if name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
                {
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
            escape_delimiters("see <|im_start|> now"),
            "see <¦im_start¦> now"
        );
    }
}
