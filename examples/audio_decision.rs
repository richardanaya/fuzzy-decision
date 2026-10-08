//! Score a WAV clip with d1-omni-600M.
//!
//! ```text
//! cargo run --release --example audio_decision -- <snapshot> <clip.wav>
//! ```
//!
//! The snapshot is a local `LiquidAI/d1-omni-600M` directory. This example
//! does not download it. `"{}"` is the state the audio questions were trained
//! with when the clip is the whole input. Non-16 kHz files are linearly
//! resampled; stereo is averaged.

use fuzzy_decision::{AudioClip, FuzzyDecision};

fn main() -> Result<(), fuzzy_decision::Error> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| "models/d1-omni-600M".into());
    let wav = args.next().expect("wav path");
    let decider = FuzzyDecision::open(dir)?;
    let clip = AudioClip::from_wav(wav)?;
    let answer = decider.choice_audio(
        &clip,
        "{}",
        "What is in this clip?",
        &["speech", "music", "noise", "silence"],
        None,
    )?;
    println!("{} ({:.3})", answer.choice, answer.confidence);
    for (label, probability) in &answer.probabilities {
        println!("{label}\t{probability:.4}");
    }
    Ok(())
}
