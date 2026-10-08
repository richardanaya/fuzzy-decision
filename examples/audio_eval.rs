//! Score a fixed set of clips with d1-omni-600M and write audio_eval.html.
//!
//! Generated clips are written under audio_eval/. The speech and silence
//! files are the fixtures in tests/data/.
//!
//! ```text
//! cargo run --release --example audio_eval
//! ```

use std::f32::consts::PI;
use std::fs;
use std::path::Path;
use std::time::Instant;

use fuzzy_decision::{AudioClip, FuzzyDecision};

struct Item {
    section: &'static str,
    name: &'static str,
    note: &'static str,
    /// Path written or copied next to the report, for the audio element.
    file: &'static str,
    question: &'static str,
    gold: &'static str,
    options: &'static [&'static str],
    clip: AudioClip,
}

struct Scored {
    item: Item,
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

fn tone(freq: f32, seconds: f32, amp: f32) -> Vec<f32> {
    let n = (16_000.0 * seconds) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            amp * (2.0 * PI * freq * t).sin()
        })
        .collect()
}

fn noise(n: usize, amp: f32) -> Vec<f32> {
    let mut state: u32 = 0xA5A5_1234;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let unit = (state >> 8) as f32 / (1u32 << 24) as f32;
            amp * (2.0 * unit - 1.0)
        })
        .collect()
}

fn write_wav(path: &str, samples: &[f32]) {
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent).expect("audio_eval dir");
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("create wav");
    for sample in samples {
        let clipped = sample.clamp(-1.0, 1.0);
        writer
            .write_sample((clipped * 32767.0) as i16)
            .expect("write sample");
    }
    writer.finalize().expect("finalize wav");
}

fn items() -> Vec<Item> {
    let kind = &[
        "speech",
        "music",
        "noise",
        "silence",
        "a pure tone",
    ][..];
    let yes_no = &["yes", "no"][..];

    let silence = vec![0.0; 16_000];
    let hiss = noise(16_000, 0.25);
    let pure = tone(440.0, 1.0, 0.25);
    let chord: Vec<f32> = (0..(16_000 * 3 / 2))
        .map(|i| {
            let t = i as f32 / 16_000.0;
            0.15
                * ((2.0 * PI * 261.63 * t).sin()
                    + (2.0 * PI * 329.63 * t).sin()
                    + (2.0 * PI * 392.00 * t).sin())
        })
        .collect();
    let speech = AudioClip::from_wav("tests/data/speech.wav").expect("speech fixture");
    let fixture_silence = AudioClip::from_wav("tests/data/silence.wav").expect("silence fixture");

    write_wav("audio_eval/silence.wav", &silence);
    write_wav("audio_eval/noise.wav", &hiss);
    write_wav("audio_eval/tone.wav", &pure);
    write_wav("audio_eval/chord.wav", &chord);
    fs::copy("tests/data/speech.wav", "audio_eval/speech.wav").expect("copy speech");
    fs::copy("tests/data/silence.wav", "audio_eval/fixture-silence.wav").expect("copy silence");

    let speech_len = speech.samples.len();
    let mut speech_plus_noise = speech.samples.clone();
    let hiss_over = noise(speech_len, 0.08);
    for (sample, hiss) in speech_plus_noise.iter_mut().zip(hiss_over) {
        *sample = (*sample + hiss).clamp(-1.0, 1.0);
    }
    write_wav("audio_eval/speech-plus-noise.wav", &speech_plus_noise);

    let clip = |samples: Vec<f32>| AudioClip::from_samples(samples, 16_000).expect("clip");

    vec![
        Item {
            section: "Kind of sound",
            name: "silence",
            note: "One second of zeros.",
            file: "audio_eval/silence.wav",
            question: "What is in this clip?",
            gold: "silence",
            options: kind,
            clip: clip(silence),
        },
        Item {
            section: "Kind of sound",
            name: "fixture silence",
            note: "tests/data/silence.wav, one second.",
            file: "audio_eval/fixture-silence.wav",
            question: "What is in this clip?",
            gold: "silence",
            options: kind,
            clip: fixture_silence,
        },
        Item {
            section: "Kind of sound",
            name: "noise",
            note: "One second of uniform noise at amplitude 0.25.",
            file: "audio_eval/noise.wav",
            question: "What is in this clip?",
            gold: "noise",
            options: kind,
            clip: clip(hiss),
        },
        Item {
            section: "Kind of sound",
            name: "pure tone",
            note: "One second of a 440 Hz sine at amplitude 0.25.",
            file: "audio_eval/tone.wav",
            question: "What is in this clip?",
            gold: "a pure tone",
            options: kind,
            clip: clip(pure),
        },
        Item {
            section: "Kind of sound",
            name: "chord",
            note: "1.5 seconds of C4, E4, and G4 added together.",
            file: "audio_eval/chord.wav",
            question: "What is in this clip?",
            gold: "music",
            options: kind,
            clip: clip(chord),
        },
        Item {
            section: "Kind of sound",
            name: "speech",
            note: "tests/data/speech.wav, about half a second.",
            file: "audio_eval/speech.wav",
            question: "What is in this clip?",
            gold: "speech",
            options: kind,
            clip: speech,
        },
        Item {
            section: "Kind of sound",
            name: "speech plus noise",
            note: "The speech fixture with a quieter noise layer added.",
            file: "audio_eval/speech-plus-noise.wav",
            question: "What is in this clip?",
            gold: "speech",
            options: kind,
            clip: clip(speech_plus_noise),
        },
        Item {
            section: "Is anyone speaking?",
            name: "silence",
            note: "One second of zeros.",
            file: "audio_eval/silence.wav",
            question: "Does this clip contain speech?",
            gold: "no",
            options: yes_no,
            clip: clip(vec![0.0; 16_000]),
        },
        Item {
            section: "Is anyone speaking?",
            name: "noise",
            note: "One second of uniform noise.",
            file: "audio_eval/noise.wav",
            question: "Does this clip contain speech?",
            gold: "no",
            options: yes_no,
            clip: clip(noise(16_000, 0.25)),
        },
        Item {
            section: "Is anyone speaking?",
            name: "chord",
            note: "The three-note chord.",
            file: "audio_eval/chord.wav",
            question: "Does this clip contain speech?",
            gold: "no",
            options: yes_no,
            clip: clip(
                (0..(16_000 * 3 / 2))
                    .map(|i| {
                        let t = i as f32 / 16_000.0;
                        0.15
                            * ((2.0 * PI * 261.63 * t).sin()
                                + (2.0 * PI * 329.63 * t).sin()
                                + (2.0 * PI * 392.00 * t).sin())
                    })
                    .collect(),
            ),
        },
        Item {
            section: "Is anyone speaking?",
            name: "speech",
            note: "tests/data/speech.wav.",
            file: "audio_eval/speech.wav",
            question: "Does this clip contain speech?",
            gold: "yes",
            options: yes_no,
            clip: AudioClip::from_wav("tests/data/speech.wav").expect("speech"),
        },
        Item {
            section: "Is anyone speaking?",
            name: "speech plus noise",
            note: "Speech with a quieter noise layer.",
            file: "audio_eval/speech-plus-noise.wav",
            question: "Does this clip contain speech?",
            gold: "yes",
            options: yes_no,
            clip: AudioClip::from_wav("audio_eval/speech-plus-noise.wav").expect("mix"),
        },
    ]
}

fn main() -> Result<(), fuzzy_decision::Error> {
    let started = Instant::now();
    let decider = FuzzyDecision::open("models/d1-omni-600M")?;
    let load_ms = started.elapsed().as_millis();
    println!("load_ms\t{load_ms}");

    let mut scored = Vec::new();
    for item in items() {
        let options = item.options;
        let started = Instant::now();
        let answer = decider.choice_audio(&item.clip, "{}", item.question, options, None)?;
        let ms = started.elapsed().as_millis();
        println!(
            "item\t{}\t{}\t{ms}\t{}\t{:.4}",
            item.section, item.name, answer.choice, answer.confidence
        );
        let probabilities = answer.probabilities.into_iter().collect();
        scored.push(Scored {
            item,
            choice: answer.choice,
            confidence: answer.confidence,
            probabilities,
            ms,
        });
    }
    write_html(&scored, load_ms);
    println!("wrote audio_eval.html");
    Ok(())
}

fn write_html(scored: &[Scored], load_ms: u128) {
    let hits = scored
        .iter()
        .filter(|item| item.choice == item.item.gold)
        .count();
    let n = scored.len();
    let total_ms: u128 = scored.iter().map(|item| item.ms).sum();
    let mut body = String::new();
    let mut current = "";
    for item in scored {
        if item.item.section != current {
            current = item.item.section;
            let items: Vec<_> = scored
                .iter()
                .filter(|other| other.item.section == current)
                .collect();
            let section_hits = items
                .iter()
                .filter(|other| other.choice == other.item.gold)
                .count();
            let mean = items.iter().map(|other| other.ms).sum::<u128>() / items.len() as u128;
            body.push_str(&format!(
                "<h2>{}</h2>\n<p>{} of {}. Mean time {:.1}s per clip, after the model was already loaded.</p>\n",
                esc(current),
                section_hits,
                items.len(),
                mean as f32 / 1000.0
            ));
        }
        let hit = item.choice == item.item.gold;
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
  <div class="clip">
    <p class="name">{name}</p>
    <audio controls src="{file}"></audio>
    <p class="exp">{note}</p>
  </div>
  <div>
    <p class="q">{q}</p>
    <p class="{class}">{mark} · {secs:.1}s · {conf:.1}% on “{choice}”</p>
    <p class="exp">Labeled answer: {gold}</p>
    {bars}
  </div>
</article>
"#,
            name = esc(item.item.name),
            file = esc(item.item.file),
            note = esc(item.item.note),
            q = esc(item.item.question),
            class = class,
            mark = mark,
            secs = item.ms as f32 / 1000.0,
            conf = item.confidence * 100.0,
            choice = esc(&item.choice),
            gold = esc(item.item.gold),
            bars = bars
        ));
    }

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>fuzzy-decision audio eval</title>
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
    .clip audio {{ width: 220px; }}
    .name {{ font-family: ui-sans-serif, system-ui, sans-serif; font-weight: 650; margin-top: 0; }}
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
      .clip audio {{ width: 100%; }}
    }}
  </style>
</head>
<body>
  <header>
    <p class="meta">fuzzy-decision &nbsp;·&nbsp; audio mode &nbsp;·&nbsp; LiquidAI/d1-omni-600M &nbsp;·&nbsp; WGPU</p>
    <h1>d1-omni-600M on a small clip set</h1>
    <p class="figure">{hits} / {n}</p>
    <p>Each clip is the whole input. The state string is <code>{{}}</code>, which is what the audio questions were trained with in that case. One forward scores every option. The percentage on a line is that option’s share of the list. Loading the checkpoint took {load:.1}s. Scoring itself was {score:.0}s across {n} questions.</p>
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
    fs::write("audio_eval.html", html).expect("write audio_eval.html");
}
