//! Score the packed pictures in image_eval/ and write image_eval.html.
//!
//! The manifest is one line per picture:
//! section, question, gold, width, height, rgb path, jpg path, then options
//! separated by tabs.
//!
//! ```text
//! cargo run --release --example image_eval -- models/d1-omni-600M /tmp/d1-image-manifest.tsv
//! ```

use std::collections::BTreeMap;
use std::fs;
use std::time::Instant;

use fuzzy_decision::{RgbImage, VisionDecision};

struct Row {
    section: String,
    question: String,
    gold: String,
    width: u32,
    height: u32,
    rgb: String,
    jpg: String,
    options: Vec<String>,
}

struct Scored {
    row: Row,
    choice: String,
    confidence: f32,
    probabilities: Vec<(String, f32)>,
    ms: u128,
}

fn esc(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn main() -> Result<(), fuzzy_decision::Error> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("snapshot directory");
    let manifest = args.next().expect("manifest");
    let text = fs::read_to_string(&manifest).unwrap_or_else(|err| panic!("read {manifest}: {err}"));
    let mut rows = Vec::new();
    for line in text
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
    {
        let mut parts = line.split('\t');
        let section = parts.next().expect("section").to_string();
        let question = parts.next().expect("question").to_string();
        let gold = parts.next().expect("gold").to_string();
        let width: u32 = parts.next().expect("width").parse().expect("width");
        let height: u32 = parts.next().expect("height").parse().expect("height");
        let rgb = parts.next().expect("rgb").to_string();
        let jpg = parts.next().expect("jpg").to_string();
        let options: Vec<String> = parts.map(str::to_string).collect();
        rows.push(Row {
            section,
            question,
            gold,
            width,
            height,
            rgb,
            jpg,
            options,
        });
    }

    let started = Instant::now();
    let decider = VisionDecision::load(&dir)?;
    let load_ms = started.elapsed().as_millis();
    println!("load_ms\t{load_ms}");

    let mut scored = Vec::new();
    for row in rows {
        let data = fs::read(&row.rgb).unwrap_or_else(|err| panic!("read {}: {err}", row.rgb));
        let image = RgbImage {
            width: row.width,
            height: row.height,
            data,
        };
        let options: Vec<&str> = row.options.iter().map(String::as_str).collect();
        let started = Instant::now();
        let answer = decider.choice(&image, "", &row.question, &options)?;
        let ms = started.elapsed().as_millis();
        println!(
            "item\t{}\t{ms}\t{}\t{:.4}",
            row.jpg, answer.choice, answer.confidence
        );
        let probabilities = answer.probabilities.into_iter().collect();
        scored.push(Scored {
            row,
            choice: answer.choice,
            confidence: answer.confidence,
            probabilities,
            ms,
        });
    }

    write_html(&scored, load_ms);
    println!("wrote image_eval.html");
    Ok(())
}

fn write_html(scored: &[Scored], load_ms: u128) {
    let hits = scored
        .iter()
        .filter(|item| item.choice == item.row.gold)
        .count();
    let n = scored.len();
    let total_ms: u128 = scored.iter().map(|item| item.ms).sum();
    let mut sections: BTreeMap<&str, Vec<&Scored>> = BTreeMap::new();
    let mut order = Vec::new();
    for item in scored {
        if !sections.contains_key(item.row.section.as_str()) {
            order.push(item.row.section.as_str());
        }
        sections
            .entry(item.row.section.as_str())
            .or_default()
            .push(item);
    }

    let mut body = String::new();
    for section in order {
        let items = &sections[section];
        let section_hits = items
            .iter()
            .filter(|item| item.choice == item.row.gold)
            .count();
        let mean = items.iter().map(|item| item.ms).sum::<u128>() / items.len() as u128;
        body.push_str(&format!(
            "<h2>{}</h2>\n<p>{} of {}. Mean time {:.1}s per picture, after the model was already loaded.</p>\n",
            esc(section),
            section_hits,
            items.len(),
            mean as f32 / 1000.0
        ));
        for item in items {
            let hit = item.choice == item.row.gold;
            let mark = if hit { "hit" } else { "miss" };
            let class = if hit { "hit" } else { "miss" };
            let mut bars = String::new();
            for (label, probability) in &item.probabilities {
                let chosen = label == &item.choice;
                let pct = probability * 100.0;
                bars.push_str(&format!(
                    r#"<div class="opt{}"><div class="lab"><span>{}</span><span>{:.1}%</span></div><div class="bar"><span style="width:{:.1}%"></span></div></div>"#,
                    if chosen { " chosen" } else { "" },
                    esc(label),
                    pct,
                    pct
                ));
            }
            body.push_str(&format!(
                r#"<article>
  <img src="{jpg}" alt="">
  <div>
    <p class="q">{q}</p>
    <p class="{class}">{mark} · {secs:.1}s · {conf:.1}% on “{choice}”</p>
    <p class="exp">Labeled answer: {gold}</p>
    {bars}
  </div>
</article>
"#,
                jpg = esc(&item.row.jpg),
                q = esc(&item.row.question),
                class = class,
                mark = mark,
                secs = item.ms as f32 / 1000.0,
                conf = item.confidence * 100.0,
                choice = esc(&item.choice),
                gold = esc(&item.row.gold),
                bars = bars
            ));
        }
    }

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>fuzzy-decision image eval</title>
  <style>
    :root {{ color-scheme: light; }}
    body {{ font-family: Georgia, "Iowan Old Style", serif; max-width: 58rem; margin: 2.5rem auto 4rem; padding: 0 1.4rem; color: #1c1a17; background: #f6f3ed; line-height: 1.5; }}
    header {{ border-bottom: 1px solid #d9d0c3; padding-bottom: 1rem; margin-bottom: 1.5rem; }}
    h1 {{ font-size: 1.9rem; font-weight: 600; margin: 0 0 0.3rem; }}
    h2 {{ font-size: 1.25rem; margin: 2rem 0 0.6rem; }}
    p {{ margin: 0.45rem 0; }}
    .meta {{ color: #5c564c; }}
    .figure {{ font-family: ui-sans-serif, system-ui, sans-serif; font-size: 2.4rem; font-weight: 600; letter-spacing: -0.03em; margin: 0.2rem 0; }}
    article {{ display: grid; grid-template-columns: 220px 1fr; gap: 1rem; background: #fff; border: 1px solid #e4ddd2; padding: 0.8rem; margin: 0.8rem 0; }}
    img {{ width: 220px; height: 220px; object-fit: cover; background: #efe8dc; }}
    .q {{ font-family: ui-sans-serif, system-ui, sans-serif; font-size: 0.95rem; }}
    .hit {{ color: #1d6b3a; font-family: ui-sans-serif, system-ui, sans-serif; }}
    .miss {{ color: #9a3412; font-weight: 600; font-family: ui-sans-serif, system-ui, sans-serif; }}
    .exp {{ color: #5c564c; font-size: 0.95rem; }}
    .opt {{ margin: 0.28rem 0; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 0.82rem; }}
    .opt .lab {{ display: flex; justify-content: space-between; gap: 1rem; }}
    .opt.chosen .lab span:first-child {{ font-weight: 650; }}
    .bar {{ height: 0.45rem; background: #efe8dc; border-radius: 99px; overflow: hidden; }}
    .bar > span {{ display: block; height: 100%; background: #2f5d50; }}
    .opt.chosen .bar > span {{ background: #1d6b3a; }}
    @media (max-width: 640px) {{
      article {{ grid-template-columns: 1fr; }}
      img {{ width: 100%; height: auto; }}
    }}
  </style>
</head>
<body>
  <header>
    <p class="meta">fuzzy-decision &nbsp;·&nbsp; vision mode &nbsp;·&nbsp; LiquidAI/d1-omni-600M &nbsp;·&nbsp; WGPU</p>
    <h1>d1-omni-600M on the picture set</h1>
    <p class="figure">{hits} / {n}</p>
    <p>One forward scores every option. The percentage on a line is that option’s share of the list. Loading the checkpoint took {load:.1}s. Scoring itself was {score:.0}s across {n} questions.</p>
  </header>
  {body}
</body>
</html>
"#,
        hits = hits,
        n = n,
        load = load_ms as f32 / 1000.0,
        score = total_ms as f32 / 1000.0,
        body = body
    );
    fs::write("image_eval.html", html).expect("write image_eval.html");
}
