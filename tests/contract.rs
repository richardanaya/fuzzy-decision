//! API checks that do not load the checkpoint.
//! The forward pass is covered by `tests/d1_forward.rs` when
//! `models/d1-omni-600M` is present.

use fuzzy_decision::{FuzzyDecision, LoadOptions, DEFAULT_MODEL, MODEL_REPO};

#[test]
fn checkpoint_names_match_the_weight_files() {
    assert_eq!(DEFAULT_MODEL, "d1-omni-600M");
    assert_eq!(MODEL_REPO, "LiquidAI/d1-omni-600M");
}

#[test]
fn info_names_the_repo_and_the_directory() {
    let info = FuzzyDecision::info(&LoadOptions {
        weights_dir: Some("/tmp/fuzzy-decision-absent".into()),
        ..LoadOptions::default()
    });
    assert_eq!(info.model, DEFAULT_MODEL);
    assert_eq!(info.repo, MODEL_REPO);
    assert!(!info.ready);
}

#[test]
fn other_model_names_are_rejected() {
    let err = FuzzyDecision::load(LoadOptions {
        model: "clef-flash".into(),
        ..LoadOptions::default()
    })
    .unwrap_err();
    let message = err.to_string();
    assert!(message.contains("d1-omni-600M"), "{message}");
    assert!(message.contains("clef-flash"), "{message}");
}

#[test]
fn a_missing_snapshot_names_the_file() {
    let err = FuzzyDecision::load(LoadOptions {
        weights_dir: Some("/tmp/fuzzy-decision-absent".into()),
        ..LoadOptions::default()
    })
    .unwrap_err();
    let message = err.to_string();
    assert!(message.contains("tokenizer.json"), "{message}");
}
