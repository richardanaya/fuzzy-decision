//! Professional and common classification questions. One load of models/kev-4b.
//! Writes text_eval.html next to Cargo.toml.
//!
//! ```text
//! cargo run --release --example domain_eval
//! ```

use std::collections::BTreeMap;
use std::fs;
use std::process::{Command, ExitCode};

use fuzzy_decision::{FuzzyDecision, LoadOptions};

#[path = "corpus.rs"]
mod corpus;

#[derive(Clone)]
pub enum Ask {
    Choice {
        instructions: String,
        options: Vec<String>,
        gold: String,
    },
    Noul {
        statement: String,
        gold: bool,
    },
    Score {
        instructions: String,
        levels: Vec<String>,
        gold: String,
    },
}

#[derive(Clone)]
pub struct Item {
    pub domain: String,
    pub state: String,
    pub ask: Ask,
}

struct Outcome {
    domain: String,
    state: String,
    question: String,
    gold: String,
    got: String,
    confidence: f32,
    hit: bool,
}

fn esc(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn write_report(outcomes: &[Outcome]) {
    let mut by_domain: BTreeMap<&str, (usize, usize, f32)> = BTreeMap::new();
    for outcome in outcomes {
        let slot = by_domain.entry(outcome.domain.as_str()).or_insert((0, 0, 0.0));
        slot.0 += 1;
        if outcome.hit {
            slot.1 += 1;
        }
        slot.2 += outcome.confidence;
    }
    let total_n = outcomes.len();
    let total_hit = outcomes.iter().filter(|outcome| outcome.hit).count();
    let date = Command::new("date")
        .arg("+%Y-%m-%d")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .unwrap_or_default()
        .trim()
        .to_string();

    let mut domain_rows = String::new();
    for (domain, (n, hit, conf_sum)) in &by_domain {
        let acc = *hit as f32 / *n as f32;
        domain_rows.push_str(&format!(
            "<tr><td>{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{:.2}</td><td><div class=\"bar\"><span style=\"width:{:.0}%\"></span></div></td><td class=\"num\">{:.2}</td></tr>\n",
            esc(domain),
            n,
            hit,
            acc,
            acc * 100.0,
            conf_sum / *n as f32
        ));
    }

    let mut miss_rows = String::new();
    let mut miss_count = 0;
    for outcome in outcomes.iter().filter(|outcome| !outcome.hit) {
        miss_count += 1;
        miss_rows.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td class=\"miss\">{}</td><td class=\"num\">{:.2}</td></tr>\n",
            esc(&outcome.domain),
            esc(&outcome.state),
            esc(&outcome.question),
            esc(&outcome.gold),
            esc(&outcome.got),
            outcome.confidence
        ));
    }
    if miss_count == 0 {
        miss_rows.push_str("<tr><td colspan=\"6\">Every item matched its label.</td></tr>\n");
    }

    let mut item_rows = String::new();
    for outcome in outcomes {
        let mark = if outcome.hit { "hit" } else { "miss" };
        item_rows.push_str(&format!(
            "<tr><td class=\"{mark}\">{mark}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td class=\"num\">{:.2}</td></tr>\n",
            esc(&outcome.domain),
            esc(&outcome.state),
            esc(&outcome.question),
            esc(&outcome.gold),
            esc(&outcome.got),
            outcome.confidence
        ));
    }

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>fuzzy-decision classification report</title>
  <style>
    :root {{ color-scheme: light; }}
    body {{ font-family: Georgia, "Iowan Old Style", serif; max-width: 58rem; margin: 2.5rem auto 4rem; padding: 0 1.4rem; color: #1c1a17; background: #f6f3ed; line-height: 1.5; }}
    header {{ border-bottom: 1px solid #d9d0c3; padding-bottom: 1rem; margin-bottom: 1.5rem; }}
    h1 {{ font-size: 1.9rem; font-weight: 600; margin: 0 0 0.3rem; }}
    h2 {{ font-size: 1.25rem; margin: 2rem 0 0.6rem; }}
    p {{ margin: 0.6rem 0; }}
    .meta {{ color: #5c564c; }}
    .figure {{ font-family: ui-sans-serif, system-ui, sans-serif; font-size: 2.4rem; font-weight: 600; letter-spacing: -0.03em; margin: 0.2rem 0; }}
    table {{ width: 100%; border-collapse: collapse; margin: 0.8rem 0 1rem; background: #fff; }}
    th, td {{ text-align: left; padding: 0.45rem 0.55rem; border-bottom: 1px solid #e4ddd2; vertical-align: top; }}
    th {{ font-family: ui-sans-serif, system-ui, sans-serif; font-size: 0.75rem; letter-spacing: 0.04em; text-transform: uppercase; color: #5c564c; }}
    td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .hit {{ color: #1d6b3a; }}
    .miss {{ color: #9a3412; font-weight: 600; }}
    code {{ font-family: ui-monospace, "Cascadia Code", monospace; font-size: 0.92em; }}
    .bar {{ height: 0.5rem; background: #efe8dc; border-radius: 99px; overflow: hidden; min-width: 5.5rem; }}
    .bar > span {{ display: block; height: 100%; background: #2f5d50; }}
  </style>
</head>
<body>
  <header>
    <p class="meta">fuzzy-decision &nbsp;·&nbsp; classification report &nbsp;·&nbsp; {date}</p>
    <h1>How Kev-4B does on ordinary classification</h1>
    <p class="figure">{total_hit} / {total_n} &nbsp; <span class="meta">{pct:.0}%</span></p>
    <p>Each item is a short workplace or everyday text, one question, and a closed set of answers. The model must pick from that set. It does not write a free-form answer.</p>
  </header>

  <h2>What was measured</h2>
  <p>The checkpoint is the Kev-4B Qwen3 revision, loaded from <code>models/kev-4b</code>. {domain_count} domains, {total_n} items. Choice and yes/no items count as a hit only when the labeled label is the one selected. Ordered items count as a hit only when the nearest level is the labeled level. Confidence is the probability of the selected answer. The same inputs return the same answer. Temperature stayed at 1.</p>
  <p>Domains: {domain_list}. Some situations are repeated with a short filing prefix so the set reaches 1000 items. The label does not change.</p>

  <h2>Results by domain</h2>
  <table>
    <thead>
      <tr><th>Domain</th><th class="num">Items</th><th class="num">Hits</th><th class="num">Accuracy</th><th></th><th class="num">Mean confidence</th></tr>
    </thead>
    <tbody>
      {domain_rows}
      <tr><td><strong>Overall</strong></td><td class="num"><strong>{total_n}</strong></td><td class="num"><strong>{total_hit}</strong></td><td class="num"><strong>{acc:.2}</strong></td><td><div class="bar"><span style="width:{overall_width:.0}%"></span></div></td><td class="num"></td></tr>
    </tbody>
  </table>

  <h2>Where it missed</h2>
  <table>
    <thead>
      <tr><th>Domain</th><th>Text</th><th>Question</th><th>Labeled</th><th>Selected</th><th class="num">Confidence</th></tr>
    </thead>
    <tbody>
      {miss_rows}
    </tbody>
  </table>

  <h2>Every item</h2>
  <table>
    <thead>
      <tr><th></th><th>Domain</th><th>Text</th><th>Question</th><th>Labeled</th><th>Selected</th><th class="num">Confidence</th></tr>
    </thead>
    <tbody>
      {item_rows}
    </tbody>
  </table>
</body>
</html>
"#,
        date = esc(&date),
        total_hit = total_hit,
        total_n = total_n,
        acc = total_hit as f32 / total_n as f32,
        pct = total_hit as f32 / total_n as f32 * 100.0,
        overall_width = total_hit as f32 / total_n as f32 * 100.0,
        domain_count = by_domain.len(),
        domain_list = esc(&by_domain.keys().copied().collect::<Vec<_>>().join(", ")),
        domain_rows = domain_rows,
        miss_rows = miss_rows,
        item_rows = item_rows,
    );
    fs::write("text_eval.html", html).expect("write text_eval.html");
}

fn main() -> ExitCode {
    let options = LoadOptions {
        weights_dir: Some("models/kev-4b".into()),
        ..LoadOptions::default()
    };
    let info = FuzzyDecision::info(&options);
    if !info.ready {
        eprintln!(
            "weights are not ready in {} (need model.safetensors, adapter_model.safetensors, head.safetensors, tokenizer.json)",
            info.weights_dir.display()
        );
        return ExitCode::from(2);
    }

    let decider = match FuzzyDecision::load(options) {
        Ok(decider) => decider,
        Err(err) => {
            eprintln!("load failed: {err}");
            return ExitCode::from(2);
        }
    };

    let mut outcomes = Vec::new();
    let bank = corpus::build();
    eprintln!("items {}", bank.len());
    for (index, item) in bank.into_iter().enumerate() {
        let (question, gold, got, confidence, hit) = match &item.ask {
            Ask::Choice { instructions, options, gold } => {
                let opts: Vec<&str> = options.iter().map(String::as_str).collect();
                let answer = decider.choice(&item.state, instructions, &opts).unwrap_or_else(|err| panic!("{err}"));
                (
                    instructions.clone(),
                    gold.clone(),
                    answer.choice.clone(),
                    answer.confidence,
                    answer.choice == *gold,
                )
            }
            Ask::Noul { statement, gold } => {
                let answer = decider.noul(&item.state, statement).unwrap_or_else(|err| panic!("{err}"));
                (
                    statement.clone(),
                    gold.to_string(),
                    answer.answer.to_string(),
                    answer.confidence,
                    answer.answer == *gold,
                )
            }
            Ask::Score { instructions, levels, gold } => {
                let levels_ref: Vec<&str> = levels.iter().map(String::as_str).collect();
                let answer = decider.score(&item.state, instructions, &levels_ref).unwrap_or_else(|err| panic!("{err}"));
                (
                    instructions.clone(),
                    gold.clone(),
                    format!("{} ({:.2})", answer.level, answer.score),
                    answer.confidence,
                    answer.level == *gold,
                )
            }
        };
        if index % 50 == 0 || !hit {
            println!(
                "{index:>4} {}  {}  gold={}  got={}  conf={:.2}",
                if hit { "hit " } else { "miss" },
                item.domain,
                gold,
                got,
                confidence
            );
        }
        outcomes.push(Outcome {
            domain: item.domain,
            state: item.state,
            question,
            gold,
            got,
            confidence,
            hit,
        });
    }

    write_report(&outcomes);
    println!("wrote text_eval.html");
    ExitCode::SUCCESS
}
