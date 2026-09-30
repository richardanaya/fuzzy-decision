//! Professional and common classification questions. One load of models/kev-4b.
//! Writes index.html next to Cargo.toml.
//!
//! ```text
//! cargo run --release --example domain_eval
//! ```

use std::collections::BTreeMap;
use std::fs;
use std::process::{Command, ExitCode};

use fuzzy_decision::{FuzzyDecision, LoadOptions};

enum Ask {
    Choice {
        instructions: &'static str,
        options: &'static [&'static str],
        gold: &'static str,
    },
    Noul {
        statement: &'static str,
        gold: bool,
    },
    Score {
        instructions: &'static str,
        levels: &'static [&'static str],
        gold: &'static str,
    },
}

struct Item {
    domain: &'static str,
    state: &'static str,
    ask: Ask,
}

struct Outcome {
    domain: &'static str,
    state: &'static str,
    question: String,
    gold: String,
    got: String,
    confidence: f32,
    hit: bool,
}

fn items() -> Vec<Item> {
    vec![
        Item {
            domain: "intent",
            state: "Can we move Thursday's budget review to Friday at 2pm? I have a client call.",
            ask: Ask::Choice {
                instructions: "What is the sender trying to do?",
                options: &["reschedule a meeting", "request a refund", "share a status update", "none"],
                gold: "reschedule a meeting",
            },
        },
        Item {
            domain: "intent",
            state: "The staging deploy finished. Error rate is flat and the new login page is up.",
            ask: Ask::Choice {
                instructions: "What is the sender trying to do?",
                options: &["reschedule a meeting", "request a refund", "share a status update", "none"],
                gold: "share a status update",
            },
        },
        Item {
            domain: "intent",
            state: "Please send me the signed statement of work before the kickoff.",
            ask: Ask::Choice {
                instructions: "What is the sender trying to do?",
                options: &["request a document", "cancel a subscription", "report an outage", "none"],
                gold: "request a document",
            },
        },
        Item {
            domain: "intent",
            state: "Thanks for the intro. I will read the brief this afternoon.",
            ask: Ask::Choice {
                instructions: "What is the sender trying to do?",
                options: &["request a document", "acknowledge a message", "escalate a complaint", "none"],
                gold: "acknowledge a message",
            },
        },
        Item {
            domain: "topic",
            state: "Q3 revenue was $4.2 million, up 6% from Q2. Gross margin held at 41%.",
            ask: Ask::Choice {
                instructions: "Which subject is this paragraph about?",
                options: &["finance", "hiring", "facilities", "product support"],
                gold: "finance",
            },
        },
        Item {
            domain: "topic",
            state: "We are opening a senior accountant role. Applications close on the 15th.",
            ask: Ask::Choice {
                instructions: "Which subject is this paragraph about?",
                options: &["finance", "hiring", "facilities", "product support"],
                gold: "hiring",
            },
        },
        Item {
            domain: "topic",
            state: "The HVAC unit on the third floor failed overnight. The floor is closed until noon.",
            ask: Ask::Choice {
                instructions: "Which subject is this paragraph about?",
                options: &["finance", "hiring", "facilities", "product support"],
                gold: "facilities",
            },
        },
        Item {
            domain: "topic",
            state: "Customers cannot export CSV from the reports page after yesterday's release.",
            ask: Ask::Choice {
                instructions: "Which subject is this paragraph about?",
                options: &["finance", "hiring", "facilities", "product support"],
                gold: "product support",
            },
        },
        Item {
            domain: "entailment",
            state: "The contract renews on June 1 unless either party gives 30 days' written notice. No notice has been sent.",
            ask: Ask::Noul {
                statement: "The contract is set to renew on June 1.",
                gold: true,
            },
        },
        Item {
            domain: "entailment",
            state: "The contract renews on June 1 unless either party gives 30 days' written notice. No notice has been sent.",
            ask: Ask::Noul {
                statement: "A party has already cancelled the renewal.",
                gold: false,
            },
        },
        Item {
            domain: "entailment",
            state: "Only managers in the payroll group can approve overtime. Jordan is a designer and is not in that group.",
            ask: Ask::Noul {
                statement: "Jordan can approve overtime.",
                gold: false,
            },
        },
        Item {
            domain: "entailment",
            state: "The vendor delivered 80 of the 100 chairs. The remaining 20 ship next Tuesday.",
            ask: Ask::Noul {
                statement: "Some of the chairs have not arrived yet.",
                gold: true,
            },
        },
        Item {
            domain: "abstain",
            state: "Please book a conference room for six people on Monday morning.",
            ask: Ask::Choice {
                instructions: "Which payroll action is requested? Choose none if the text is not a payroll action.",
                options: &["run payroll", "correct a tax withholding", "issue a bonus", "none"],
                gold: "none",
            },
        },
        Item {
            domain: "abstain",
            state: "The cafeteria serves soup on Wednesdays.",
            ask: Ask::Choice {
                instructions: "Which legal filing does the text request? Choose none if it requests none.",
                options: &["file a trademark", "send a cease and desist", "none"],
                gold: "none",
            },
        },
        Item {
            domain: "abstain",
            state: "Please correct the tax withholding on my August paycheck.",
            ask: Ask::Choice {
                instructions: "Which payroll action is requested? Choose none if the text is not a payroll action.",
                options: &["run payroll", "correct a tax withholding", "issue a bonus", "none"],
                gold: "correct a tax withholding",
            },
        },
        Item {
            domain: "abstain",
            state: "Attached is the agenda for the design critique.",
            ask: Ask::Choice {
                instructions: "Which facilities request is this? Choose none if it is not a facilities request.",
                options: &["repair the HVAC", "replace a badge", "none"],
                gold: "none",
            },
        },
        Item {
            domain: "routing",
            state: "I was billed twice for the April invoice and I want the duplicate charge returned.",
            ask: Ask::Choice {
                instructions: "Which team should own this message?",
                options: &["billing", "account access", "shipping", "none"],
                gold: "billing",
            },
        },
        Item {
            domain: "routing",
            state: "I cannot sign in. The password reset email never arrives.",
            ask: Ask::Choice {
                instructions: "Which team should own this message?",
                options: &["billing", "account access", "shipping", "none"],
                gold: "account access",
            },
        },
        Item {
            domain: "routing",
            state: "The tracking page still shows my order sitting at the warehouse from last week.",
            ask: Ask::Choice {
                instructions: "Which team should own this message?",
                options: &["billing", "account access", "shipping", "none"],
                gold: "shipping",
            },
        },
        Item {
            domain: "routing",
            state: "What time does the downtown shop close on Sundays?",
            ask: Ask::Choice {
                instructions: "Which team should own this message? Choose none if none of these teams own it.",
                options: &["billing", "account access", "shipping", "none"],
                gold: "none",
            },
        },
        Item {
            domain: "expense",
            state: "Uber from the airport to the client office, $46, March 2.",
            ask: Ask::Choice {
                instructions: "Which expense category is this?",
                options: &["ground transport", "lodging", "meals", "software", "none"],
                gold: "ground transport",
            },
        },
        Item {
            domain: "expense",
            state: "Two nights at the Harbor Hotel during the Seattle offsite, $380.",
            ask: Ask::Choice {
                instructions: "Which expense category is this?",
                options: &["ground transport", "lodging", "meals", "software", "none"],
                gold: "lodging",
            },
        },
        Item {
            domain: "expense",
            state: "Annual seat for the design tool, billed to the company card.",
            ask: Ask::Choice {
                instructions: "Which expense category is this?",
                options: &["ground transport", "lodging", "meals", "software", "none"],
                gold: "software",
            },
        },
        Item {
            domain: "expense",
            state: "Team lunch after the customer workshop, $96 including tip.",
            ask: Ask::Choice {
                instructions: "Which expense category is this?",
                options: &["ground transport", "lodging", "meals", "software", "none"],
                gold: "meals",
            },
        },
        Item {
            domain: "priority",
            state: "A typo in the footer of the monthly newsletter. The send already went out.",
            ask: Ask::Score {
                instructions: "How urgent is this for the on-call team?",
                levels: &["low", "medium", "high", "urgent"],
                gold: "low",
            },
        },
        Item {
            domain: "priority",
            state: "Checkout has failed for every customer for the last 40 minutes. No orders are completing.",
            ask: Ask::Score {
                instructions: "How urgent is this for the on-call team?",
                levels: &["low", "medium", "high", "urgent"],
                gold: "urgent",
            },
        },
        Item {
            domain: "priority",
            state: "Search is slow for some users, about three seconds, and results are still correct.",
            ask: Ask::Score {
                instructions: "How urgent is this for the on-call team?",
                levels: &["low", "medium", "high", "urgent"],
                gold: "medium",
            },
        },
        Item {
            domain: "priority",
            state: "New signups in one region are erroring. Other regions are fine. Support volume is rising.",
            ask: Ask::Score {
                instructions: "How urgent is this for the on-call team?",
                levels: &["low", "medium", "high", "urgent"],
                gold: "high",
            },
        },
        Item {
            domain: "sentiment",
            state: "This is the clearest onboarding I have used. I had my team invited the same day.",
            ask: Ask::Score {
                instructions: "How positive is this message?",
                levels: &["very negative", "negative", "neutral", "positive", "very positive"],
                gold: "very positive",
            },
        },
        Item {
            domain: "sentiment",
            state: "Support never replied, and the invoice was wrong for the second month in a row.",
            ask: Ask::Score {
                instructions: "How positive is this message?",
                levels: &["very negative", "negative", "neutral", "positive", "very positive"],
                gold: "very negative",
            },
        },
        Item {
            domain: "sentiment",
            state: "The office address is 400 Market Street. Reception is on the second floor.",
            ask: Ask::Score {
                instructions: "How positive is this message?",
                levels: &["very negative", "negative", "neutral", "positive", "very positive"],
                gold: "neutral",
            },
        },
        Item {
            domain: "sentiment",
            state: "The workshop was useful. The room was crowded, but the material was solid.",
            ask: Ask::Score {
                instructions: "How positive is this message?",
                levels: &["very negative", "negative", "neutral", "positive", "very positive"],
                gold: "positive",
            },
        },
    ]
}

fn esc(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn write_report(outcomes: &[Outcome]) {
    let mut by_domain: BTreeMap<&str, (usize, usize, f32)> = BTreeMap::new();
    for outcome in outcomes {
        let slot = by_domain.entry(outcome.domain).or_insert((0, 0, 0.0));
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
            esc(outcome.domain),
            esc(outcome.state),
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
            esc(outcome.domain),
            esc(outcome.state),
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
  <p>The checkpoint is the Kev-4B Qwen3 revision, loaded from <code>models/kev-4b</code>. Eight domains, four items each. Choice and yes/no items count as a hit only when the labeled label is the one selected. Ordered items (priority and sentiment) count as a hit only when the nearest level is the labeled level. Confidence is the probability of the selected answer. The same inputs return the same answer. Temperature stayed at 1.</p>
  <p>The domains are message intent, document topic, whether a statement follows from a memo, refusing a label set that does not apply, which team owns a request, expense category, on-call urgency, and how positive a message is.</p>

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
        domain_rows = domain_rows,
        miss_rows = miss_rows,
        item_rows = item_rows,
    );
    fs::write("index.html", html).expect("write index.html");
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
    for item in items() {
        let (question, gold, got, confidence, hit) = match &item.ask {
            Ask::Choice { instructions, options, gold } => {
                let answer = decider.choice(item.state, *instructions, options).unwrap_or_else(|err| panic!("{err}"));
                (
                    (*instructions).to_string(),
                    (*gold).to_string(),
                    answer.choice.clone(),
                    answer.confidence,
                    answer.choice == *gold,
                )
            }
            Ask::Noul { statement, gold } => {
                let answer = decider.noul(item.state, *statement).unwrap_or_else(|err| panic!("{err}"));
                (
                    (*statement).to_string(),
                    gold.to_string(),
                    answer.answer.to_string(),
                    answer.confidence,
                    answer.answer == *gold,
                )
            }
            Ask::Score { instructions, levels, gold } => {
                let answer = decider.score(item.state, *instructions, levels).unwrap_or_else(|err| panic!("{err}"));
                (
                    (*instructions).to_string(),
                    (*gold).to_string(),
                    format!("{} ({:.2})", answer.level, answer.score),
                    answer.confidence,
                    answer.level == *gold,
                )
            }
        };
        println!(
            "{}  {}  gold={}  got={}  conf={:.2}",
            if hit { "hit " } else { "miss" },
            item.domain,
            gold,
            got,
            confidence
        );
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
    println!("wrote index.html");
    ExitCode::SUCCESS
}
