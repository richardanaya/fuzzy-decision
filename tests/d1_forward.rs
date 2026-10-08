//! One text decision and one audio decision of d1-omni-600M on WGPU.
//! Skips when `models/d1-omni-600M` has not been downloaded, and when the
//! process cannot create a WGPU device.

use std::path::Path;

use fuzzy_decision::{
    choice, noul, score, Answer, AudioClip, DecideOptions, FuzzyDecision, LoadOptions,
};

fn weights_dir() -> &'static Path {
    Path::new("models/d1-omni-600M")
}

fn ready() -> bool {
    let dir = weights_dir();
    ["tokenizer.json", "config.json", "model.safetensors"]
        .into_iter()
        .all(|name| dir.join(name).is_file())
}

fn load() -> Option<FuzzyDecision> {
    if !ready() {
        eprintln!("skipping: models/d1-omni-600M is not downloaded");
        return None;
    }
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        FuzzyDecision::load(LoadOptions {
            weights_dir: Some(weights_dir().to_path_buf()),
            ..LoadOptions::default()
        })
    })) {
        Ok(Ok(model)) => Some(model),
        Ok(Err(err)) => panic!("load failed: {err}"),
        Err(_) => {
            eprintln!("skipping: the WGPU device could not be created");
            None
        }
    }
}

#[test]
fn d1_scores_a_typed_text_question() {
    let Some(decider) = load() else {
        return;
    };

    let answers = decider
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
                    &[
                        "very negative",
                        "negative",
                        "neutral",
                        "positive",
                        "very positive",
                    ],
                ),
                noul("The customer is asking for a refund."),
            ],
            DecideOptions::default(),
        )
        .expect("forward");

    match &answers[0] {
        Answer::Choice {
            probabilities,
            choice,
            ..
        } => {
            let sum: f32 = probabilities.values().sum();
            assert!((sum - 1.0).abs() < 1e-3, "{sum}");
            assert!(probabilities.contains_key(choice));
            assert!(probabilities.values().all(|p| p.is_finite() && *p >= 0.0));
            eprintln!("choice={choice} {probabilities:?}");
        }
        other => panic!("expected choice, got {other:?}"),
    }
    match &answers[2] {
        Answer::Noul {
            probability,
            answer,
            ..
        } => {
            assert!((0.0..=1.0).contains(probability));
            assert_eq!(*answer, *probability >= 0.5);
            eprintln!("noul={probability}");
        }
        other => panic!("expected noul, got {other:?}"),
    }
}

#[test]
fn d1_scores_a_synthetic_tone() {
    let Some(decider) = load() else {
        return;
    };
    let samples: Vec<f32> = (0..16_000)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            0.2 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect();
    let clip = AudioClip::from_samples(samples, 16_000).expect("clip");
    let answer = decider
        .choice_audio(
            &clip,
            "{}",
            "What is in this clip?",
            &["speech", "music", "noise", "silence"],
            None,
        )
        .expect("audio forward");
    let sum: f32 = answer.probabilities.values().sum();
    assert!((sum - 1.0).abs() < 1e-3, "{sum}");
    assert!(answer.probabilities.contains_key(&answer.choice));
    eprintln!(
        "audio choice={} ({:.3}) {:?}",
        answer.choice, answer.confidence, answer.probabilities
    );
}
