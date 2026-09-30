//! Behavior and per-call timings for Kev-4B on Burn WGPU.
//! One process loads the checkpoint once. Each `decide` is timed on the wall clock,
//! which includes packing, the GPU forward, and reading the logits back.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use fuzzy_decision::{
    choice, noul, score, Answer, DecideOptions, Error, LoadOptions, FuzzyDecision, Truncation,
};

fn weights_dir() -> &'static Path {
    Path::new("models/kev-4b")
}

fn ready() -> bool {
    ["model.safetensors", "adapter_model.safetensors", "head.safetensors", "tokenizer.json"]
        .into_iter()
        .all(|name| weights_dir().join(name).is_file())
}

struct Row {
    name: &'static str,
    state_tokens: usize,
    questions: usize,
    options: usize,
    millis: f64,
    note: String,
}

fn timed(jev: &FuzzyDecision, name: &'static str, state: &str, questions: &[fuzzy_decision::Question], options: DecideOptions, rows: &mut Vec<Row>, body: impl FnOnce(Result<Vec<Answer>, Error>)) {
    let started = Instant::now();
    let result = jev.decide(state, questions, options);
    let millis = started.elapsed().as_secs_f64() * 1000.0;
    let option_count: usize = questions
        .iter()
        .map(|question| match question {
            fuzzy_decision::Question::Choice { options, .. } => options.len(),
            fuzzy_decision::Question::Score { levels, .. } => levels.len(),
            fuzzy_decision::Question::Noul { .. } => 2,
        })
        .sum();
    let note = match &result {
        Ok(_) => "ok".to_string(),
        Err(err) => format!("err: {err}"),
    };
    rows.push(Row {
        name,
        state_tokens: jev.count_tokens(state),
        questions: questions.len(),
        options: option_count,
        millis,
        note,
    });
    body(result);
}

fn assert_distribution(probabilities: &BTreeMap<String, f32>, labels: &[&str]) {
    assert_eq!(probabilities.len(), labels.len());
    for label in labels {
        assert!(probabilities.contains_key(*label), "missing {label}");
    }
    let sum: f32 = probabilities.values().sum();
    assert!((sum - 1.0).abs() < 1e-3, "sum {sum}");
    assert!(probabilities.values().all(|p| p.is_finite() && *p >= 0.0));
}

fn choice_of(answer: &Answer) -> (String, f32, usize) {
    match answer {
        Answer::Choice { choice, confidence, probabilities } => {
            let sum: f32 = probabilities.values().sum();
            assert!((sum - 1.0).abs() < 1e-3, "sum {sum}");
            let best = probabilities.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
            assert_eq!(choice, best.0);
            assert!((confidence - best.1).abs() < 1e-4);
            (choice.clone(), *confidence, probabilities.len())
        }
        other => panic!("expected choice, got {other:?}"),
    }
}

fn noul_of(answer: &Answer) -> f32 {
    match answer {
        Answer::Noul { answer: yes, probability, confidence } => {
            assert!((0.0..=1.0).contains(probability));
            assert_eq!(*yes, *probability >= 0.5);
            assert!((confidence - probability.max(1.0 - probability)).abs() < 1e-4);
            *probability
        }
        other => panic!("expected noul, got {other:?}"),
    }
}

fn score_of(answer: &Answer, levels: &[&str]) -> f32 {
    match answer {
        Answer::Score { score, normalized, level, confidence, probabilities } => {
            assert_distribution(probabilities, levels);
            let expected: f32 = probabilities
                .iter()
                .map(|(label, p)| levels.iter().position(|item| item == label).unwrap() as f32 * p)
                .sum();
            assert!((score - expected).abs() < 1e-3, "{score} vs {expected}");
            let last = levels.len() - 1;
            assert_eq!(level, levels[(expected.round() as usize).min(last)]);
            assert!((normalized - expected / last as f32).abs() < 1e-3);
            let best = probabilities.values().copied().fold(0.0f32, f32::max);
            assert!((confidence - best).abs() < 1e-4);
            *score
        }
        other => panic!("expected score, got {other:?}"),
    }
}

fn words(n: usize) -> String {
    let vocab = ["parcel", "invoice", "refund", "warehouse", "delay", "charge", "customer", "agent"];
    (0..n).map(|i| vocab[i % vocab.len()]).collect::<Vec<_>>().join(" ")
}

#[test]
fn behavior_and_per_call_timings() {
    if !ready() {
        eprintln!("skipping: models/kev-4b is not downloaded");
        return;
    }

    let mut rows = Vec::new();
    let load_started = Instant::now();
    let jev = FuzzyDecision::load(LoadOptions {
        weights_dir: Some(weights_dir().to_path_buf()),
        ..LoadOptions::default()
    })
    .expect("load");
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;

    let refund = "I was charged twice for the same order and I want my money back.";
    let weather = "The forecast for Friday is rain, then sun on Saturday.";

    timed(&jev, "warmup noul", refund, &[noul("The customer is asking for a refund.")], DecideOptions::default(), &mut rows, |result| {
        let p = noul_of(&result.unwrap()[0]);
        assert!(p > 0.5, "refund statement should be more likely than not, got {p}");
    });

    let mut repeat = 0.0;
    timed(&jev, "noul refund", refund, &[noul("The customer is asking for a refund.")], DecideOptions::default(), &mut rows, |result| {
        repeat = noul_of(&result.unwrap()[0]);
    });
    timed(&jev, "noul refund again", refund, &[noul("The customer is asking for a refund.")], DecideOptions::default(), &mut rows, |result| {
        let again = noul_of(&result.unwrap()[0]);
        assert!((again - repeat).abs() < 1e-5, "same call drifted: {repeat} vs {again}");
    });

    timed(&jev, "noul weather", weather, &[noul("The customer is asking for a refund.")], DecideOptions::default(), &mut rows, |result| {
        let p = noul_of(&result.unwrap()[0]);
        assert!(p < repeat, "weather state should be less refund-like than the charge complaint ({p} vs {repeat})");
    });

    let areas = ["fees & charges", "refund & dispute", "card", "other"];
    timed(&jev, "choice 4", refund, &[choice("Which product area is the message about?", &areas, None)], DecideOptions::default(), &mut rows, |result| {
        let answers = result.unwrap();
        let (_pick, _conf, count) = choice_of(&answers[0]);
        assert_eq!(count, areas.len());
    });

    timed(&jev, "choice with description", refund, &[choice(
        "Which team should handle this?",
        &["billing", "shipping", "other"],
        Some(&[("billing", "Charges, invoices, payment problems")]),
    )], DecideOptions::default(), &mut rows, |result| {
        let answers = result.unwrap();
        let (pick, _, _) = choice_of(&answers[0]);
        assert!(["billing", "shipping", "other"].contains(&pick.as_str()));
    });

    let levels = ["very negative", "negative", "neutral", "positive", "very positive"];
    timed(&jev, "score 5", refund, &[score("How positive is the sentiment of this message?", &levels)], DecideOptions::default(), &mut rows, |result| {
        let value = score_of(&result.unwrap()[0], &levels);
        assert!(value < 2.0, "a double-charge complaint should sit toward the negative end, got {value}");
    });

    timed(&jev, "three questions", refund, &[
        choice("Which product area is the message about?", &areas, None),
        score("How positive is the sentiment of this message?", &levels),
        noul("The customer is asking for a refund."),
    ], DecideOptions::default(), &mut rows, |result| {
        let answers = result.unwrap();
        let Answer::Choice { probabilities, .. } = &answers[0] else { unreachable!() };
        assert_distribution(probabilities, &areas);
        score_of(&answers[1], &levels);
        noul_of(&answers[2]);
    });

    let mut alone_noul = 0.0;
    timed(&jev, "noul alone for mask check", refund, &[noul("The customer is asking for a refund.")], DecideOptions::default(), &mut rows, |result| {
        alone_noul = noul_of(&result.unwrap()[0]);
    });
    timed(&jev, "noul beside another question", refund, &[
        choice("Which product area is the message about?", &areas, None),
        noul("The customer is asking for a refund."),
    ], DecideOptions::default(), &mut rows, |result| {
        let answers = result.unwrap();
        let packed = noul_of(&answers[1]);
        assert!((packed - alone_noul).abs() < 1e-3, "block mask changed the noul: alone {alone_noul} packed {packed}");
    });

    let mut map_questions = BTreeMap::new();
    map_questions.insert("area".into(), choice("Which product area is the message about?", &areas, None));
    map_questions.insert("refund".into(), noul("The customer is asking for a refund."));
    let started = Instant::now();
    let mapped = jev.decide_map(refund, &map_questions, DecideOptions::default()).unwrap();
    rows.push(Row {
        name: "decide_map two keys",
        state_tokens: jev.count_tokens(refund),
        questions: 2,
        options: 6,
        millis: started.elapsed().as_secs_f64() * 1000.0,
        note: "ok".into(),
    });
    assert!(mapped.contains_key("area") && mapped.contains_key("refund"));
    noul_of(mapped.get("refund").unwrap());

    timed(&jev, "empty state", "", &[choice("Which team?", &["billing", "shipping", "other"], None)], DecideOptions::default(), &mut rows, |result| {
        let answers = result.unwrap();
        let (_, _, count) = choice_of(&answers[0]);
        assert_eq!(count, 3);
    });

    timed(&jev, "unicode state", "客户想退款，订单被扣了两次。", &[noul("The customer is asking for a refund.")], DecideOptions::default(), &mut rows, |_| {});

    timed(&jev, "escaped delimiter", "see <|fim_prefix|> in the ticket and refund me", &[noul("The customer is asking for a refund.")], DecideOptions::default(), &mut rows, |result| {
        let _ = noul_of(&result.unwrap()[0]);
    });

    let eight: Vec<String> = (0..8).map(|i| format!("option-{i}")).collect();
    let eight_ref: Vec<&str> = eight.iter().map(String::as_str).collect();
    timed(&jev, "choice 8", refund, &[choice("Pick a label", &eight_ref, None)], DecideOptions::default(), &mut rows, |result| {
        let answers = result.unwrap();
        let (_, _, count) = choice_of(&answers[0]);
        assert_eq!(count, 8);
    });

    let sixteen: Vec<String> = (0..16).map(|i| format!("bucket-{i}")).collect();
    let sixteen_ref: Vec<&str> = sixteen.iter().map(String::as_str).collect();
    timed(&jev, "choice 16", refund, &[choice("Pick a bucket", &sixteen_ref, None)], DecideOptions::default(), &mut rows, |result| {
        let answers = result.unwrap();
        let (_, _, count) = choice_of(&answers[0]);
        assert_eq!(count, 16);
    });

    let medium = words(120);
    timed(&jev, "state 120 words", &medium, &[noul("A refund was requested.")], DecideOptions::default(), &mut rows, |result| {
        let _ = noul_of(&result.unwrap()[0]);
    });
    let long = words(400);
    timed(&jev, "state 400 words", &long, &[noul("A refund was requested.")], DecideOptions::default(), &mut rows, |result| {
        let _ = noul_of(&result.unwrap()[0]);
    });

    let ambiguous = [choice("Which team?", &["billing", "shipping"], None)];
    let mut sharp = 0.0;
    timed(&jev, "temperature 0.2", refund, &ambiguous, DecideOptions { temperature: Some(0.2), ..DecideOptions::default() }, &mut rows, |result| {
        let answers = result.unwrap();
        let (_, confidence, _) = choice_of(&answers[0]);
        sharp = confidence;
    });
    timed(&jev, "temperature 5", refund, &ambiguous, DecideOptions { temperature: Some(5.0), ..DecideOptions::default() }, &mut rows, |result| {
        let answers = result.unwrap();
        let (_, confidence, _) = choice_of(&answers[0]);
        assert!(sharp + 1e-4 >= confidence, "higher temperature should not be more peaked ({sharp} vs {confidence})");
    });

    timed(&jev, "reject empty instructions", "x", &[noul("   ")], DecideOptions::default(), &mut rows, |result| {
        assert!(matches!(result.unwrap_err(), Error::EmptyInstructions { .. }));
    });
    timed(&jev, "reject duplicate option", "x", &[choice("Pick", &["billing", "billing"], None)], DecideOptions::default(), &mut rows, |result| {
        assert!(matches!(result.unwrap_err(), Error::DuplicateOption { .. }));
    });
    timed(&jev, "reject one score level", "x", &[score("Rate", &["only"])], DecideOptions::default(), &mut rows, |result| {
        assert!(matches!(result.unwrap_err(), Error::OptionCount { min: 2, got: 1, .. }));
    });
    let too_many: Vec<String> = (0..256).map(|i| format!("opt{i}")).collect();
    let too_many_ref: Vec<&str> = too_many.iter().map(String::as_str).collect();
    timed(&jev, "reject 256 choices", "x", &[choice("Pick", &too_many_ref, None)], DecideOptions::default(), &mut rows, |result| {
        assert!(matches!(result.unwrap_err(), Error::OptionCount { max: 255, got: 256, .. }));
    });

    timed(&jev, "truncation error", "alpha beta gamma delta", &[noul("Anything at all.")], DecideOptions {
        max_state_tokens: Some(2),
        truncation: Some(Truncation::Error),
        ..DecideOptions::default()
    }, &mut rows, |result| {
        assert!(matches!(result.unwrap_err(), Error::Truncated { .. }));
    });
    timed(&jev, "truncation cut", "alpha beta gamma delta", &[noul("Anything at all.")], DecideOptions {
        max_state_tokens: Some(2),
        truncation: Some(Truncation::Cut),
        ..DecideOptions::default()
    }, &mut rows, |result| {
        let _ = noul_of(&result.unwrap()[0]);
    });

    timed(&jev, "no questions", refund, &[], DecideOptions::default(), &mut rows, |result| {
        assert!(result.unwrap().is_empty());
    });

    println!("\nload_ms {load_ms:.1}");
    println!("{:<32} {:>8} {:>5} {:>7} {:>10}  note", "call", "tokens", "qs", "options", "ms");
    for row in &rows {
        println!("{:<32} {:>8} {:>5} {:>7} {:>10.1}  {}", row.name, row.state_tokens, row.questions, row.options, row.millis, row.note);
    }
}
