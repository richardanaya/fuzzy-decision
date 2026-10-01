//! Score a local RGB image with `VisionDecision`.
//!
//! ```text
//! cargo run --release --example vision_choice -- <snapshot> <rgb> <width> <height>
//! ```
//!
//! `rgb` is tightly packed RGB bytes. The snapshot is a local
//! `Cloudflare/clef-flash` directory. This example does not download it.

use fuzzy_decision::{RgbImage, VisionDecision};

fn main() -> Result<(), fuzzy_decision::Error> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("snapshot directory");
    let rgb_path = args.next().expect("rgb file");
    let width: u32 = args.next().expect("width").parse().expect("width");
    let height: u32 = args.next().expect("height").parse().expect("height");
    let data = std::fs::read(&rgb_path).unwrap_or_else(|err| panic!("read {rgb_path}: {err}"));
    let image = RgbImage { width, height, data };
    let decider = VisionDecision::load(dir)?;
    let answer = decider.choice(
        &image,
        "",
        "What is in the picture? Answer with one of the options.",
        &["a red circle", "a blue square", "a green triangle", "nothing"],
    )?;
    println!("{}", answer.choice);
    for (label, probability) in &answer.probabilities {
        println!("{label}\t{probability:.4}");
    }
    Ok(())
}
