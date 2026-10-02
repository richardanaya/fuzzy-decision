//! Parity tests against fixtures produced by the reference Python
//! implementation in `Cloudflare/clef-flash` (`joint_schema_model.py`).
//!
//! `tests/data/head_fixture.safetensors` holds a small random joint head, a
//! synthetic record, and the logits PyTorch computed for it.
//! `tests/data/encoding_fixture.safetensors` holds the ids and spans the
//! reference `encode_record` produced over `tests/data/small_tokenizer.json`.
//! The scripts that build them are linked from the pull request that added
//! this module; regenerating them needs only `torch`, `tokenizers`, and
//! `safetensors`.

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use safetensors::SafeTensors;

use crate::head::{HeadConfig, HeadQuestion, JointSchemaHead};
use crate::questions::{choice, noul, score, Question};
use crate::record::encode_record;
use crate::tokenize::HfTokenizer;
use crate::weights::to_f32;

type B = NdArray<f32>;

fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn values(tensors: &SafeTensors<'_>, name: &str) -> (Vec<f32>, Vec<usize>) {
    let view = tensors.tensor(name).unwrap_or_else(|err| panic!("{name}: {err}"));
    (to_f32(&view).expect("fixture dtype"), view.shape().to_vec())
}

fn span_of(values: &[f32]) -> (usize, usize) {
    (values[0] as usize, values[1] as usize)
}

#[test]
fn head_matches_the_reference_forward() {
    let bytes = std::fs::read(fixture_path("head_fixture.safetensors")).expect("head fixture");
    let tensors = SafeTensors::deserialize(&bytes).expect("fixture file");
    let (config_values, _) = values(&tensors, "fixture.config");
    let config = HeadConfig {
        hidden_size: config_values[0] as usize,
        width: config_values[1] as usize,
        routing_layers: config_values[2] as usize,
        layers: config_values[3] as usize,
        heads: config_values[4] as usize,
        feedforward: config_values[5] as usize,
    };
    let device = NdArrayDevice::default();
    let head = JointSchemaHead::<B>::load(&bytes, config, &device).expect("load head");

    let (hidden_values, hidden_shape) = values(&tensors, "fixture.hidden");
    let seq = hidden_shape[0];
    let hidden = Tensor::<B, 2>::from_data(
        TensorData::new(hidden_values, [seq, config.hidden_size]),
        &device,
    );
    let (id_values, _) = values(&tensors, "fixture.input_ids");
    let input_ids: Vec<u32> = id_values.iter().map(|id| *id as u32).collect();
    let (embedding, embedding_shape) = values(&tensors, "fixture.embedding");
    assert_eq!(embedding_shape[1], config.hidden_size);

    let mut questions = Vec::new();
    let mut lexical = Vec::new();
    let mut expected = Vec::new();
    for index in 0.. {
        let name = format!("fixture.question.{index}");
        if tensors.tensor(&name).is_err() {
            break;
        }
        let (meta, _) = values(&tensors, &name);
        let (spans, spans_shape) = values(&tensors, &format!("fixture.option_spans.{index}"));
        let option_spans: Vec<(usize, usize)> = (0..spans_shape[0])
            .map(|row| span_of(&spans[row * 2..row * 2 + 2]))
            .collect();
        let mut rows = Vec::with_capacity(option_spans.len() * config.hidden_size);
        for (start, end) in &option_spans {
            let mut mean = vec![0f32; config.hidden_size];
            for id in &input_ids[*start..*end] {
                let row = &embedding[*id as usize * config.hidden_size..];
                for (slot, value) in mean.iter_mut().zip(row) {
                    *slot += value;
                }
            }
            for slot in &mut mean {
                *slot /= (end - start) as f32;
            }
            rows.extend(mean);
        }
        lexical.push(Tensor::<B, 2>::from_data(
            TensorData::new(rows, [option_spans.len(), config.hidden_size]),
            &device,
        ));
        questions.push(HeadQuestion {
            question_type: meta[0] as usize,
            question_span: span_of(&meta[1..3]),
            option_spans,
        });
        let (logits, _) = values(&tensors, &format!("fixture.logits.{index}"));
        expected.push(logits);
    }
    assert_eq!(questions.len(), 3, "the fixture has three questions");

    let normalized = head.normalized(hidden);
    let memory = head.memory(normalized.clone());
    let results = head.score(memory, normalized, 0, &questions, lexical);

    for (result, want) in results.iter().zip(&expected) {
        assert_eq!(result.len(), want.len());
        for (got, want) in result.iter().zip(want) {
            assert!(
                (got - want).abs() < 3e-4,
                "logit {got} differs from the reference {want}"
            );
        }
    }
}

#[test]
fn encoding_matches_the_reference_layout() {
    let tokenizer =
        HfTokenizer::open(&fixture_path("small_tokenizer.json")).expect("small tokenizer");
    let bytes = std::fs::read(fixture_path("encoding_fixture.safetensors")).expect("encoding fixture");
    let tensors = SafeTensors::deserialize(&bytes).expect("fixture file");

    let records: Vec<(&str, Vec<Question>, Vec<String>)> = vec![
        (
            "Our checkout started returning errors and orders are blocked.",
            vec![
                choice(
                    "Which team should handle the message?",
                    &["billing", "technical"],
                    Some(&[
                        ("billing", "Payments or invoices"),
                        ("technical", "Bugs or outages"),
                    ]),
                ),
                score("urgency", &["Can wait", "This week", "Today"]),
                noul("Is a service down?"),
            ],
            vec!["department".into(), "urgency".into(), "outage".into()],
        ),
        (
            "A note with unicode: café — 15°, and \"quotes\" plus a \\ backslash.\nSecond line.",
            vec![choice("Pick a label.", &["zeta", "alpha"], None)],
            vec!["pick".into()],
        ),
    ];

    for (record_index, (state, questions, ids)) in records.iter().enumerate() {
        let layout = encode_record(
            &tokenizer,
            state,
            None,
            questions,
            ids,
            16384,
            16384,
            false,
        )
        .expect("encode");
        let prefix = format!("record{record_index}");
        let (want_ids, _) = values(&tensors, &format!("{prefix}.input_ids"));
        let got_ids: Vec<f32> = layout.ids.iter().map(|id| *id as f32).collect();
        assert_eq!(got_ids, want_ids, "{prefix} token ids");
        for (question_index, question) in layout.questions.iter().enumerate() {
            let (meta, _) = values(&tensors, &format!("{prefix}.q{question_index}.meta"));
            assert_eq!(question.question_type, meta[0] as usize, "{prefix} type");
            assert_eq!(
                question.question_span,
                span_of(&meta[1..3]),
                "{prefix} q{question_index} span"
            );
            let (spans, spans_shape) =
                values(&tensors, &format!("{prefix}.q{question_index}.option_spans"));
            let want: Vec<(usize, usize)> = (0..spans_shape[0])
                .map(|row| span_of(&spans[row * 2..row * 2 + 2]))
                .collect();
            assert_eq!(question.option_spans, want, "{prefix} q{question_index} options");
        }
    }
}
