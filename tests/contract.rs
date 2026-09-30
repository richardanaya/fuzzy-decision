//! API checks that do not load the 0.6B checkpoint.
//! The forward pass is covered by `tests/kev_forward.rs` when `models/kev-0.6b` is present.

use fuzzy_decision::{Family, FuzzyDecision, LoadOptions, DEFAULT_MODEL, MODELS};

#[test]
fn aliases_match_the_checkpoint_names() {
    assert_eq!(DEFAULT_MODEL, "kev-0.6b");
    assert_eq!(
        MODELS,
        &[
            ("kev-0.6b", "onnx-community/kev-0.6b-ONNX"),
            ("kev-4b", "onnx-community/kev-4b-ONNX"),
        ]
    );
}

#[test]
fn info_reports_the_kev_family() {
    let info = FuzzyDecision::info(&LoadOptions::default());
    assert_eq!(info.family, Family::Kev);
    assert_eq!(info.device, "webgpu");
    assert_eq!(info.model, "onnx-community/kev-0.6b-ONNX");
    assert_eq!(Family::Kev.as_str(), "kev");
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
