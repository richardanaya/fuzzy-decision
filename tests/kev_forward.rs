//! One forward pass of the published Kev-0.6B weights on WGPU.
//! Skips when `models/kev-0.6b` has not been downloaded.

use std::path::Path;

use fuzzy_decision::{choice, noul, score, Answer, DecideOptions, LoadOptions, FuzzyDecision};

fn weights_dir() -> &'static Path {
    Path::new("models/kev-0.6b")
}

fn ready() -> bool {
    let dir = weights_dir();
    ["model.safetensors", "adapter_model.safetensors", "head.safetensors", "tokenizer.json"]
        .into_iter()
        .all(|name| dir.join(name).is_file())
}

#[test]
fn kev_0_6b_scores_a_typed_question() {
    if !ready() {
        eprintln!("skipping: models/kev-0.6b is not downloaded");
        return;
    }
    let jev = FuzzyDecision::load(LoadOptions {
        weights_dir: Some(weights_dir().to_path_buf()),
        ..LoadOptions::default()
    })
    .expect("load kev-0.6b");

    let answers = jev
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
