//! API checks that do not load the 0.6B checkpoint.
//! The forward pass is covered by `tests/kev_forward.rs` when `models/kev-0.6b` is present.

use fuzzy_decision::{FuzzyDecision, LoadOptions, ADAPTER_REPO, BASE_REPO, DEFAULT_MODEL};

#[test]
fn checkpoint_names_match_the_weight_files() {
    assert_eq!(DEFAULT_MODEL, "kev-0.6b");
    assert_eq!(BASE_REPO, "Qwen/Qwen3-0.6B-Base");
    assert_eq!(ADAPTER_REPO, "jaredpalmer/kev-0.6b");
}

#[test]
fn info_names_the_repos_and_the_directory() {
    let info = FuzzyDecision::info(&LoadOptions {
        weights_dir: Some("/tmp/fuzzy-decision-absent".into()),
        ..LoadOptions::default()
    });
    assert_eq!(info.model, DEFAULT_MODEL);
    assert_eq!(info.base_repo, BASE_REPO);
    assert_eq!(info.adapter_repo, ADAPTER_REPO);
    assert!(!info.ready);
}

#[test]
fn only_kev_0_6b_loads() {
    let err = FuzzyDecision::load(LoadOptions {
        model: "kev-4b".into(),
        ..LoadOptions::default()
    })
    .unwrap_err();
    let message = err.to_string();
    assert!(message.contains("kev-0.6b"), "{message}");
}
