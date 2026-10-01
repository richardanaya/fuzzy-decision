use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Truncation {
    /// Drop trailing state tokens that do not fit.
    Cut,
    /// Fail when the state does not fit.
    Error,
}

#[derive(Debug, Clone)]
pub enum Question {
    Choice {
        instructions: String,
        options: Vec<String>,
        descriptions: BTreeMap<String, String>,
    },
    Score {
        instructions: String,
        levels: Vec<String>,
    },
    Noul {
        instructions: String,
    },
}

impl Question {
    pub fn instructions(&self) -> &str {
        match self {
            Question::Choice { instructions, .. }
            | Question::Score { instructions, .. }
            | Question::Noul { instructions } => instructions,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Question::Choice { .. } => "choice",
            Question::Score { .. } => "score",
            Question::Noul { .. } => "noul",
        }
    }
}

/// Options the models were trained with for noul (index 1 = yes).
pub const NOUL_OPTIONS: [&str; 2] = ["no", "yes"];

#[derive(Debug, Clone, Copy)]
pub struct QuestionLimits {
    pub min_choice_options: usize,
    pub max_choice_options: usize,
    pub min_score_levels: usize,
    pub max_score_levels: usize,
}

pub fn choice(
    instructions: impl Into<String>,
    options: &[&str],
    descriptions: Option<&[(&str, &str)]>,
) -> Question {
    let mut map = BTreeMap::new();
    if let Some(pairs) = descriptions {
        for (key, value) in pairs {
            map.insert((*key).to_string(), (*value).to_string());
        }
    }
    Question::Choice {
        instructions: instructions.into(),
        options: options.iter().map(|s| (*s).to_string()).collect(),
        descriptions: map,
    }
}

pub fn score(instructions: impl Into<String>, levels: &[&str]) -> Question {
    Question::Score {
        instructions: instructions.into(),
        levels: levels.iter().map(|s| (*s).to_string()).collect(),
    }
}

pub fn noul(statement: impl Into<String>) -> Question {
    Question::Noul {
        instructions: statement.into(),
    }
}

pub fn question_labels(question: &Question) -> Vec<String> {
    match question {
        Question::Noul { .. } => NOUL_OPTIONS.iter().map(|s| (*s).to_string()).collect(),
        Question::Choice { options, .. } => options.clone(),
        Question::Score { levels, .. } => levels.clone(),
    }
}

pub fn validate_question(
    question: &Question,
    label: &str,
    limits: QuestionLimits,
) -> Result<(), crate::Error> {
    if question.instructions().trim().is_empty() {
        return Err(crate::Error::EmptyInstructions {
            label: label.to_string(),
        });
    }

    if matches!(question, Question::Noul { .. }) {
        return Ok(());
    }

    let (kind, options, min, max) = match question {
        Question::Choice { options, .. } => (
            "choice",
            options,
            limits.min_choice_options,
            limits.max_choice_options,
        ),
        Question::Score { levels, .. } => (
            "score",
            levels,
            limits.min_score_levels,
            limits.max_score_levels,
        ),
        Question::Noul { .. } => unreachable!(),
    };

    if options.len() < min || options.len() > max {
        return Err(crate::Error::OptionCount {
            label: label.to_string(),
            kind: kind.to_string(),
            min,
            max,
            got: options.len(),
        });
    }

    let mut seen = std::collections::BTreeSet::new();
    for option in options {
        if option.trim().is_empty() {
            return Err(crate::Error::BadOption {
                label: label.to_string(),
            });
        }
        if !seen.insert(option.clone()) {
            return Err(crate::Error::DuplicateOption {
                label: label.to_string(),
                option: option.clone(),
            });
        }
    }
    Ok(())
}
