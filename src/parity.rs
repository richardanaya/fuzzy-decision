//! Parity against fixtures generated from the d1 reference (`audio.py`,
//! `encoder.py`, `vision.py`) with tiny or random weights.

use burn::tensor::Device;

use crate::conformer::AudioTower;
use crate::head::DecisionHead;
use crate::mel::{log_mel, prepare_waveform};
use crate::nn::{tensor2, to_vec};
use crate::resample::resize_hwc;
use crate::trunk::Trunk;
use crate::vision::VisionTower;
use crate::weights::{AudioSpec, LayerKind, Snapshot, TensorSource, TrunkSpec, VisionSpec};

fn cpu_device() -> Device {
    // Same backend as `Session::load_cpu`. Flex is the 0.22 replacement, but
    // these fixtures were checked on NdArray.
    #[allow(deprecated)]
    Device::ndarray()
}

fn fixture(name: &str) -> Snapshot {
    Snapshot::open(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data")
            .join(name),
    )
    .unwrap_or_else(|err| panic!("open {name}: {err}"))
}

fn values(source: &Snapshot, name: &str) -> (Vec<f32>, Vec<usize>) {
    source
        .tensor(name)
        .unwrap_or_else(|err| panic!("{name}: {err}"))
}

fn assert_close(got: &[f32], expected: &[f32], atol: f32, label: &str) {
    assert_eq!(got.len(), expected.len(), "{label} length");
    let mut max = 0.0f32;
    for (left, right) in got.iter().zip(expected) {
        max = max.max((left - right).abs());
    }
    assert!(max <= atol, "{label} max abs {max} > {atol}");
}

#[test]
fn mel_matches_the_reference_frontend() {
    let source = fixture("mel_fixture.safetensors");
    let (samples, _) = values(&source, "fixture.samples");
    let (expected, shape) = values(&source, "fixture.mel");
    let wave = prepare_waveform(&samples);
    let mel = log_mel(&wave);
    assert_eq!(mel.n_mels, shape[0]);
    assert_eq!(mel.frames, shape[1]);
    assert_eq!(mel.valid, 50);
    assert_close(&mel.features, &expected, 2e-4, "mel");
}

#[test]
fn trunk_matches_the_reference_forward() {
    let source = fixture("trunk_fixture.safetensors");
    let spec = TrunkSpec {
        hidden: 32,
        intermediate: 48,
        heads: 4,
        kv_heads: 2,
        layers: vec![LayerKind::Conv, LayerKind::Attention],
        eps: 1e-5,
        multiple_of: 8,
        ffn_multiplier: 1.0,
        rope_theta: 10_000.0,
    };
    let device = cpu_device();
    let trunk = Trunk::load(&source, &spec, &device).expect("trunk");
    let (input, shape) = values(&source, "fixture.input");
    let hidden = tensor2(input, shape[0], shape[1], &device);
    let prefix = values(&source, "fixture.prefix").0[0] as usize;
    let got = to_vec(trunk.forward(hidden, prefix));
    let (expected, _) = values(&source, "fixture.output");
    assert_close(&got, &expected, 2e-4, "trunk");
}

#[test]
fn head_matches_the_reference_logits() {
    let source = fixture("head_fixture.safetensors");
    let device = cpu_device();
    let head = DecisionHead::load(&source, 64, 2, &device).expect("head");
    let (hidden, shape) = values(&source, "fixture.hidden");
    let markers: Vec<usize> = values(&source, "fixture.markers")
        .0
        .iter()
        .map(|value| *value as usize)
        .collect();
    let qtype = values(&source, "fixture.qtype").0[0] as usize;
    let got = head.logits(
        tensor2(hidden, shape[0], shape[1], &device),
        qtype,
        &markers,
    );
    let (expected, _) = values(&source, "fixture.logits");
    assert_close(&got, &expected, 2e-4, "head");
}

#[test]
fn audio_tower_matches_the_reference() {
    let source = fixture("audio_fixture.safetensors");
    let spec = AudioSpec {
        feat_in: 8,
        layers: 1,
        d_model: 32,
        channels: 4,
        ff_expansion: 2,
        heads: 4,
        kernel: 3,
        residual_width: 8,
    };
    let device = cpu_device();
    let tower = AudioTower::load(&source, &spec, &device).expect("audio");
    let (mel, shape) = values(&source, "fixture.mel");
    let valid = values(&source, "fixture.lengths").0[0] as usize;
    let got = to_vec(tower.forward_mel(&mel, shape[0], shape[1], valid));
    let (expected, _) = values(&source, "fixture.output");
    assert_close(&got, &expected, 2e-4, "audio");
}

#[test]
fn resize_matches_torch_antialias() {
    let source = fixture("resize_fixture.safetensors");
    let (input, shape) = values(&source, "fixture.input");
    let (expected, out_shape) = values(&source, "fixture.output");
    let got = resize_hwc(
        &input,
        shape[0],
        shape[1],
        shape[2],
        out_shape[0],
        out_shape[1],
    );
    assert_close(&got, &expected, 1e-5, "resize");
}

#[test]
fn vision_tower_matches_the_reference() {
    let source = fixture("vision_fixture.safetensors");
    let spec = VisionSpec {
        hidden: 32,
        intermediate: 48,
        heads: 4,
        layers: 1,
        eps: 1e-6,
        patch: 2,
        num_patches: 16,
        projector_hidden: 16,
    };
    let device = cpu_device();
    let tower = VisionTower::load(&source, &spec, &device).expect("vision");
    let (patches, shape) = values(&source, "fixture.patches");
    let spatial = values(&source, "fixture.spatial").0;
    let got =
        to_vec(tower.forward_crop(&patches, spatial[0] as usize, spatial[1] as usize, shape[1]));
    let (expected, _) = values(&source, "fixture.output");
    assert_close(&got, &expected, 2e-4, "vision");
}

#[test]
fn safetensors_helper_reads_f32() {
    let source = Snapshot::open(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/resize_fixture.safetensors"),
    )
    .unwrap();
    let (values, _) = source.tensor("fixture.output").unwrap();
    assert!(!values.is_empty());
}
