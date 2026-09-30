//! Time a folder of packed RGB images against one shared label list.
//!
//! Each line of the manifest is `id<TAB>expected<TAB>width<TAB>height<TAB>rgb-path`.
//! The label list is every expected value, in first-seen order.
//!
//! ```text
//! cargo run --release --example vision_suite -- <snapshot> <manifest>
//! ```

use std::time::Instant;

use fuzzy_decision::{RgbImage, VisionDecision};

fn main() -> Result<(), fuzzy_decision::Error> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("snapshot directory");
    let manifest = args.next().expect("manifest");
    let text = std::fs::read_to_string(&manifest).unwrap_or_else(|err| panic!("read {manifest}: {err}"));
    let mut rows = Vec::new();
    let mut labels = Vec::new();
    for line in text.lines().filter(|line| !line.is_empty()) {
        let mut parts = line.split('\t');
        let id = parts.next().expect("id").to_string();
        let expected = parts.next().expect("expected").to_string();
        let width: u32 = parts.next().expect("width").parse().expect("width");
        let height: u32 = parts.next().expect("height").parse().expect("height");
        let path = parts.next().expect("path").to_string();
        if !labels.iter().any(|label: &String| label == &expected) {
            labels.push(expected.clone());
        }
        rows.push((id, expected, width, height, path));
    }
    let options: Vec<&str> = labels.iter().map(String::as_str).collect();
    let started = Instant::now();
    let decider = VisionDecision::load(&dir)?;
    let load_ms = started.elapsed().as_millis();
    println!("load_ms\t{load_ms}");
    let mut hits = 0usize;
    let mut total_ms = 0u128;
    for (id, expected, width, height, path) in &rows {
        let data = std::fs::read(path).unwrap_or_else(|err| panic!("read {path}: {err}"));
        let image = RgbImage {
            width: *width,
            height: *height,
            data,
        };
        let started = Instant::now();
        let answer = decider.choice(
            &image,
            "",
            "What is the main subject? Answer with one of the options.",
            &options,
        )?;
        let ms = started.elapsed().as_millis();
        total_ms += ms;
        let hit = answer.choice == *expected;
        if hit {
            hits += 1;
        }
        let breakdown = answer
            .probabilities
            .iter()
            .map(|(label, probability)| format!("{label}={probability:.4}"))
            .collect::<Vec<_>>()
            .join("|");
        println!(
            "item\t{id}\t{ms}\t{}\t{}\t{:.4}\t{breakdown}",
            if hit { "hit" } else { "miss" },
            answer.choice,
            answer.confidence
        );
    }
    let n = rows.len();
    println!(
        "summary\t{hits}\t{n}\t{total_ms}\t{}",
        if n == 0 { 0 } else { total_ms / n as u128 }
    );
    Ok(())
}
