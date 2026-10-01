//! Clef-Flash tokenizer: the Qwen3.5 tokenizer plus the delimiter escaping
//! this crate applies to user text.

use std::path::Path;

use tokenizers::Tokenizer;

use crate::record::TemplateEncoder;

pub struct HfTokenizer {
    inner: Tokenizer,
    vision_start: u32,
    vision_end: u32,
    image_pad: u32,
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
            vision_start: id("<|vision_start|>")?,
            vision_end: id("<|vision_end|>")?,
            image_pad: id("<|image_pad|>")?,
            inner,
        })
    }

    pub fn count(&self, text: &str) -> usize {
        self.encode_user(text).len()
    }

    /// Template text. Special tokens such as `<|im_start|>` stay special.
    pub fn encode_raw(&self, text: &str) -> Vec<u32> {
        self.inner
            .encode(text, false)
            .map(|enc| enc.get_ids().to_vec())
            .unwrap_or_default()
    }

    /// User text. `<|name|>` spellings are escaped so user text cannot
    /// close the state or inject chat delimiters.
    pub fn encode_user(&self, text: &str) -> Vec<u32> {
        self.encode_raw(&escape_delimiters(text))
    }

    /// `<|vision_start|>`, `n` image slots, `<|vision_end|>`, and the newline
    /// the Clef processor appends after the media block.
    pub fn media_ids(&self, n_image: usize) -> Vec<u32> {
        let mut ids = Vec::with_capacity(n_image + 3);
        ids.push(self.vision_start);
        ids.extend(std::iter::repeat(self.image_pad).take(n_image));
        ids.push(self.vision_end);
        ids.extend(self.encode_raw("\n"));
        ids
    }
}

impl TemplateEncoder for HfTokenizer {
    fn raw(&self, text: &str) -> Vec<u32> {
        self.encode_raw(text)
    }

    fn user(&self, text: &str) -> Vec<u32> {
        self.encode_user(text)
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

pub fn escape_delimiters(text: &str) -> String {
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
            escape_delimiters("see <|im_start|> now"),
            "see <¦im_start¦> now"
        );
    }
}
