//! Sequence builders for the two model families.
//!
//! DeBERTa:
//! `[CLS] [STATE] state [Q] instructions [OPT] option … [SEP]`
//! plus span-slot ids (`seg`, `pair_q`, `pair_opt`).
//!
//! kev (Qwen3):
//! `<state> state <q> instructions <opt> option </opt> … <decide>`
//! Logits are read at each option's end delimiter.

#[derive(Debug, Clone)]
pub struct TokenizedQuestion {
    pub instructions: Vec<i32>,
    pub options: Vec<Vec<i32>>,
}

#[derive(Debug, Clone)]
pub struct EncodedSequence {
    pub input_ids: Vec<i32>,
    /// Span slots for the DeBERTa graph. The WGPU pointer head scores
    /// positions directly, so these stay with the sequence for callers.
    #[allow(dead_code)]
    pub extra_inputs: ExtraInputs,
    /// Per question: indices into the flat logit vector, one per option.
    pub groups: Vec<Vec<usize>>,
    #[allow(dead_code)]
    pub state_tokens: usize,
    pub state_truncated: bool,
}

#[derive(Debug, Clone)]
pub struct ExtraInputs {
    #[allow(dead_code)]
    pub seg: Vec<i64>,
    #[allow(dead_code)]
    pub pair_q: Vec<i64>,
    #[allow(dead_code)]
    pub pair_opt: Vec<i64>,
}

impl Default for ExtraInputs {
    fn default() -> Self {
        Self {
            seg: Vec::new(),
            pair_q: Vec::new(),
            pair_opt: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MarkerIds {
    pub cls: i32,
    pub sep: i32,
    pub state: i32,
    pub q: i32,
    pub opt: i32,
}

#[derive(Debug, Clone)]
pub struct EncodeParams {
    pub state: Vec<i32>,
    pub questions: Vec<TokenizedQuestion>,
    pub markers: MarkerIds,
    pub max_state_tokens: usize,
    pub max_length: usize,
}

pub fn encode_sequence(params: EncodeParams) -> Result<EncodedSequence, crate::Error> {
    let EncodeParams {
        state,
        questions,
        markers,
        max_state_tokens,
        max_length,
    } = params;

    let mut fixed_tokens = 3usize;
    for question in &questions {
        fixed_tokens += 1 + question.instructions.len();
        for option in &question.options {
            fixed_tokens += 1 + option.len();
        }
    }

    if fixed_tokens > max_length {
        return Err(crate::Error::Context {
            message: format!(
                "Questions need {fixed_tokens} tokens which exceeds the {max_length} token context. Shorten the instructions or options, or ask fewer questions per call."
            ),
        });
    }

    let budget = max_length - fixed_tokens;
    let state_limit = max_state_tokens.min(budget);
    let state_tokens: Vec<i32> = state.iter().copied().take(state_limit).collect();

    let mut input_ids: Vec<i32> = Vec::with_capacity(max_length);
    input_ids.push(markers.cls);
    input_ids.push(markers.state);
    input_ids.extend_from_slice(&state_tokens);

    let mut seg: Vec<i64> = vec![-1; input_ids.len()];
    let mut pair_q = Vec::new();
    let mut pair_opt = Vec::new();
    let mut groups = Vec::with_capacity(questions.len());

    let total_pairs: usize = questions.iter().map(|q| q.options.len()).sum();

    for (question_index, question) in questions.iter().enumerate() {
        let question_slot = (total_pairs + question_index) as i64;
        input_ids.push(markers.q);
        seg.push(-1);
        for id in &question.instructions {
            input_ids.push(*id);
            seg.push(question_slot);
        }

        let mut group = Vec::with_capacity(question.options.len());
        for option in &question.options {
            let pair_index = pair_opt.len() as i64;
            input_ids.push(markers.opt);
            seg.push(-1);
            for id in option {
                input_ids.push(*id);
                seg.push(pair_index);
            }
            pair_q.push(question_slot);
            pair_opt.push(pair_index);
            group.push(pair_index as usize);
        }
        groups.push(group);
    }

    input_ids.push(markers.sep);
    seg.push(-1);

    Ok(EncodedSequence {
        state_truncated: state_tokens.len() < state.len(),
        state_tokens: state_tokens.len(),
        input_ids,
        extra_inputs: ExtraInputs {
            seg,
            pair_q,
            pair_opt,
        },
        groups,
    })
}

#[derive(Debug, Clone, Copy)]
pub struct KevDelimiterIds {
    pub state: i32,
    pub question: i32,
    pub option_start: i32,
    pub option_end: i32,
    pub decide: i32,
}

#[derive(Debug, Clone)]
pub struct KevEncodeParams {
    pub state: Vec<i32>,
    pub questions: Vec<TokenizedQuestion>,
    pub delimiters: KevDelimiterIds,
    pub max_state_tokens: usize,
    pub max_length: usize,
}

pub fn encode_kev_sequence(params: KevEncodeParams) -> Result<EncodedSequence, crate::Error> {
    let KevEncodeParams {
        state,
        questions,
        delimiters,
        max_state_tokens,
        max_length,
    } = params;

    struct Branch {
        branch: Vec<i32>,
        ends: Vec<usize>,
    }

    let branches: Vec<Branch> = questions
        .iter()
        .map(|question| {
            let mut branch = Vec::new();
            branch.push(delimiters.question);
            branch.extend_from_slice(&question.instructions);
            let mut ends = Vec::new();
            for option in &question.options {
                branch.push(delimiters.option_start);
                branch.extend_from_slice(option);
                branch.push(delimiters.option_end);
                ends.push(branch.len() - 1);
            }
            branch.push(delimiters.decide);
            Branch { branch, ends }
        })
        .collect();

    let longest = branches.iter().map(|b| b.branch.len()).max().unwrap_or(0);
    if longest + 1 > max_length {
        return Err(crate::Error::Context {
            message: format!(
                "A question needs {} tokens which exceeds the {max_length} token context. Shorten the instructions or options.",
                longest + 1
            ),
        });
    }

    let budget = max_length - 1 - longest;
    let state_limit = max_state_tokens.min(budget);
    let state_tokens: Vec<i32> = state.iter().copied().take(state_limit).collect();

    let mut input_ids = Vec::new();
    input_ids.push(delimiters.state);
    input_ids.extend_from_slice(&state_tokens);

    let mut groups = Vec::with_capacity(branches.len());
    for branch in &branches {
        let base = input_ids.len();
        input_ids.extend_from_slice(&branch.branch);
        groups.push(branch.ends.iter().map(|end| base + end).collect());
    }

    Ok(EncodedSequence {
        state_truncated: state_tokens.len() < state.len(),
        state_tokens: state_tokens.len(),
        input_ids,
        extra_inputs: ExtraInputs::default(),
        groups,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(instructions: Vec<i32>, options: Vec<Vec<i32>>) -> TokenizedQuestion {
        TokenizedQuestion {
            instructions,
            options,
        }
    }

    #[test]
    fn kev_reads_option_end_positions() {
        let encoded = encode_kev_sequence(KevEncodeParams {
            state: vec![10, 11],
            questions: vec![q(vec![20], vec![vec![30], vec![31, 32]])],
            delimiters: KevDelimiterIds {
                state: 1,
                question: 2,
                option_start: 3,
                option_end: 4,
                decide: 5,
            },
            max_state_tokens: 100,
            max_length: 100,
        })
        .unwrap();

        assert_eq!(
            encoded.input_ids,
            vec![1, 10, 11, 2, 20, 3, 30, 4, 3, 31, 32, 4, 5]
        );
        assert_eq!(encoded.groups, vec![vec![7, 11]]);
        assert!(!encoded.state_truncated);
    }

    #[test]
    fn deberta_pair_slots() {
        let encoded = encode_sequence(EncodeParams {
            state: vec![9],
            questions: vec![q(vec![8], vec![vec![7], vec![6]])],
            markers: MarkerIds {
                cls: 1,
                sep: 2,
                state: 3,
                q: 4,
                opt: 5,
            },
            max_state_tokens: 100,
            max_length: 100,
        })
        .unwrap();

        assert_eq!(encoded.input_ids, vec![1, 3, 9, 4, 8, 5, 7, 5, 6, 2]);
        assert_eq!(encoded.groups, vec![vec![0, 1]]);
        assert_eq!(encoded.extra_inputs.pair_q, vec![2, 2]);
        assert_eq!(encoded.extra_inputs.pair_opt, vec![0, 1]);
        assert_eq!(
            encoded.extra_inputs.seg,
            vec![-1, -1, -1, -1, 2, -1, 0, -1, 1, -1]
        );
    }

    #[test]
    fn kev_rejects_a_branch_past_the_limit() {
        let err = encode_kev_sequence(KevEncodeParams {
            state: vec![1],
            questions: vec![q(vec![2, 2, 2], vec![vec![3]])],
            delimiters: KevDelimiterIds {
                state: 10,
                question: 11,
                option_start: 12,
                option_end: 13,
                decide: 14,
            },
            max_state_tokens: 8,
            max_length: 4,
        })
        .unwrap_err();
        assert!(matches!(err, crate::Error::Context { .. }));
    }
}
