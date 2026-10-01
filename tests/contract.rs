//! API checks that do not load the 9B checkpoint.
//! The forward pass is covered by `tests/clef_forward.rs` when
//! `models/clef-flash` is present.

use fuzzy_decision::{FuzzyDecision, LoadOptions, DEFAULT_MODEL, MODEL_REPO};

#[test]
fn checkpoint_names_match_the_weight_files() {
    assert_eq!(DEFAULT_MODEL, "clef-flash");
    assert_eq!(MODEL_REPO, "Cloudflare/clef-flash");
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
        model: "kev-4b".into(),
        ..LoadOptions::default()
    })
    .unwrap_err();
    let message = err.to_string();
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
