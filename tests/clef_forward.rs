//! One forward pass of Clef-Flash on WGPU.
//! Skips when `models/clef-flash` has not been downloaded.

use std::path::Path;

use fuzzy_decision::{choice, noul, score, Answer, DecideOptions, FuzzyDecision, LoadOptions};

fn weights_dir() -> &'static Path {
    Path::new("models/clef-flash")
}

fn ready() -> bool {
    let dir = weights_dir();
    [
        "tokenizer.json",
        "model.safetensors.index.json",
        "joint_head.safetensors",
        "joint_head_config.json",
    ]
    .into_iter()
    .all(|name| dir.join(name).is_file())
}

#[test]
fn clef_flash_scores_a_typed_question() {
    if !ready() {
        eprintln!("skipping: models/clef-flash is not downloaded");
        return;
    }
    let clef = FuzzyDecision::load(LoadOptions {
        weights_dir: Some(weights_dir().to_path_buf()),
        ..LoadOptions::default()
    })
    .expect("load clef-flash");

    let answers = clef
        .decide(
            "I was charged twice for the same order and I want my money back.",
            &[
                choice(
                    "Which product area is the message about?",
                    &["fees & charges", "refund & dispute", "card", "other"],
                    None,
                ),
                score(
                    "How positive is the sentiment?",
                    &["very negative", "negative", "neutral", "positive", "very positive"],
                ),
                noul("The customer is asking for a refund."),
            ],
            DecideOptions::default(),
        )
        .expect("forward");

    match &answers[0] {
        Answer::Choice { probabilities, choice, .. } => {
            let sum: f32 = probabilities.values().sum();
            assert!((sum - 1.0).abs() < 1e-3, "{sum}");
            assert!(probabilities.contains_key(choice));
            assert!(probabilities.values().all(|p| p.is_finite() && *p >= 0.0));
            eprintln!("choice={choice} {probabilities:?}");
        }
        other => panic!("expected choice, got {other:?}"),
    }
    match &answers[2] {
        Answer::Noul { probability, answer, .. } => {
            assert!((0.0..=1.0).contains(probability));
            assert_eq!(*answer, *probability >= 0.5);
            eprintln!("noul={probability}");
        }
        other => panic!("expected noul, got {other:?}"),
    }
}
