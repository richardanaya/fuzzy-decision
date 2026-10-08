# fuzzy-decision

`fuzzy-decision` scores questions you write. Each answer is a probability over the options you supplied. The model does not write new text. Text, audio, and image questions use [LiquidAI/d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M). The forward pass runs on [Burn](https://burn.dev) 0.21 with the WGPU backend. This crate reads weights from a directory you pass. It does not download them.

The crate is MIT. The checkpoint is separate and uses the [LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-omni-600M). That license restricts commercial use to entities under $10M annual revenue.

**d1-omni-600M** is Liquid AI's small omni decision model. A bidirectional encoder reads a state, and a two-layer decision head scores the options of one question in a single forward pass. Nothing is generated. A request is text, text plus one image, or text plus one audio clip.

One process holds one loaded model. Later calls reuse it.

## Add the crate

```toml
[dependencies]
fuzzy-decision = "0.6"
```

A call needs a GPU that WGPU can see (Vulkan, Metal, or DX12). The checkpoint is one f32 `model.safetensors` of about 2.35 GB. The token embedding table (about 256 MB) stays on the CPU. The encoder plus the audio tower is about 1.8 GB of weights on the device, plus activations. `VisionDecision` loads the encoder and the SigLIP2 vision tower instead of the audio tower.

## Weights

Put a local `LiquidAI/d1-omni-600M` snapshot in one directory, then pass that directory to `FuzzyDecision::open`. `FuzzyDecision::load(LoadOptions::default())` looks for `models/d1-omni-600M` relative to the process working directory. A missing file returns `Error::MissingFile` with the directory and the file name.

| File in that directory | Purpose |
| --- | --- |
| `tokenizer.json` | the tokenizer |
| `config.json` | architecture and text temperatures |
| `model.safetensors` | encoder, decision head, audio tower, and vision tower (f32) |

Fetch them from the application that embeds this library, once, before the first `open`. The library never calls the network.

```bash
DIR=models/d1-omni-600M
mkdir -p "$DIR"
for f in tokenizer.json config.json model.safetensors; do
  curl -L --fail -o "$DIR/$f" \
    "https://huggingface.co/LiquidAI/d1-omni-600M/resolve/main/$f"
done
```

No token is required; the repository is public. If you mirror it behind authentication, pass your own token to `curl` (for example `-H "Authorization: Bearer $HF_TOKEN"`) — this crate never reads one.

Then `FuzzyDecision::open(DIR)`. Keep the directory next to the application, or set `LoadOptions { weights_dir: Some(path), .. }`. Do not commit the files.

Only `d1-omni-600M` loads. Any other `LoadOptions.model` returns `Error::UnsupportedModel`.

The `cpu` feature loads that same snapshot with [`CpuDecision`] on Burn's NdArray backend, for a machine without a GPU. The default API stays on WGPU. `CpuDecision` keeps the checkpoint in host memory.

## One question

```rust
use fuzzy_decision::FuzzyDecision;

fn main() -> Result<(), fuzzy_decision::Error> {
    let decider = FuzzyDecision::open("models/d1-omni-600M")?;
    let state = "I was charged twice and I want my money back.";

    let refund = decider.noul(state, "The customer is asking for a refund.")?;
    if refund.answer {
        println!("refund, p = {:.3}", refund.probability);
    }

    let area = decider.choice(
        state,
        "Which area?",
        &["fees & charges", "refund & dispute", "other"],
    )?;
    println!("{} ({:.3})", area.choice, area.confidence);
    println!("fees: {:.3}", area.probability("fees & charges")?);

    let tone = decider.score(
        state,
        "How positive is this message?",
        &["very negative", "negative", "neutral", "positive", "very positive"],
    )?;
    println!("score {:.2} -> {}", tone.score, tone.level);

    Ok(())
}
```

`state` is the document. The second string is the question. Options and levels are the only answers the model can return.

- `noul` is yes or no. `answer` is `true` when the yes probability is at least `0.5`. `probability` is the yes probability. `confidence` is the larger of yes and no.
- `choice` picks one label. `choice` is that label. `confidence` is its probability. `probability("fees & charges")` reads any label you passed. An unknown label returns `Error::UnknownOption`.
- `score` treats `levels` as ordered from low to high. `score` is the expected index (`0.0` is the first level, `4.0` would be the last in the example). `normalized` divides that by the last index, so it sits in `0.0..=1.0`. `level` is the level nearest the expected index. `confidence` is the probability of the single most likely level.

The same state and the same options return the same answer. The library does not sample.

Text questions use the per-type temperatures in `config.json`. A temperature you set on `LoadOptions` or `DecideOptions` replaces that value. Image and audio questions stay at temperature `1.0` unless you set one.

## Several questions

`decide` scores each question on its own forward pass. Questions do not see each other, so an answer does not move when you add another question. Map keys from `decide_map` are not written into the prompt.

```rust
use fuzzy_decision::{choice, noul, score, Answer, DecideOptions, FuzzyDecision};

let decider = FuzzyDecision::open("models/d1-omni-600M")?;
let state = "I was charged twice and I want my money back.";
let answers = decider.decide(
    state,
    &[
        choice("Which area?", &["fees & charges", "refund & dispute", "other"], None),
        score("How positive is this message?", &["negative", "neutral", "positive"]),
        noul("The customer is asking for a refund."),
    ],
    DecideOptions::default(),
)?;

for answer in &answers {
    match answer {
        Answer::Choice { choice, confidence, .. } => {
            println!("choice {choice} ({confidence:.3})");
        }
        Answer::Score { score, level, .. } => {
            println!("score {score:.2} ({level})");
        }
        Answer::Noul { answer, probability, .. } => {
            println!("noul {answer} (yes = {probability:.3})");
        }
    }
}
```

Answers come back in the same order as the questions. A description can be attached to a choice option. For text and images the model sees `key: description` when a description is set, and the bare key otherwise. The returned label stays the name. Options stay in the order you passed them.

```rust
choice(
    "Which area?",
    &["fees", "refund"],
    Some(&[("fees", "billing mistakes and extra charges")]),
)
```

`decide_map` takes a `BTreeMap<String, Question>` and returns a `BTreeMap<String, Answer>` with the same keys.

## Audio

`FuzzyDecision` also scores a clip. Pass samples or a WAV path. There is no microphone capture.

```rust
use fuzzy_decision::{AudioClip, FuzzyDecision};

let decider = FuzzyDecision::open("models/d1-omni-600M")?;
let clip = AudioClip::from_wav("clip.wav")?;
let answer = decider.choice_audio(
    &clip,
    "{}",
    "What is in this clip?",
    &["speech", "music", "noise", "silence"],
    None,
)?;
println!("{} ({:.3})", answer.choice, answer.confidence);
```

`from_samples` and `from_pcm16` build a clip without a file. `decide_audio`, `noul_audio`, and `score_audio` cover the other question types. `cargo run --release --example audio_decision -- models/d1-omni-600M clip.wav` does the same choice.

The clip is prepared on the CPU, then the audio tower runs on the device:

1. Stereo and other multi-channel WAVs are averaged to mono. Integer samples are scaled by `2^(bits-1)` (16-bit by 32768). Float WAVs are used as stored.
2. Anything that is not 16 kHz is linearly resampled to 16 kHz.
3. The waveform is cut to 30 seconds and padded to 0.5 seconds when it is shorter.
4. A 128-bin log-mel frontend matches the reference: pre-emphasis 0.97, a 512-point FFT, a 400-point Hann window, hop 160, a Slaney mel filterbank, and per-feature normalization.
5. An 8× convolutional subsampler and a 17-layer FastConformer (width 512) produce one embedding per 80 ms. An adapter maps those to the encoder width of 1024 and adds a residual.
6. Those embeddings are prepended to the text. The text questions then run as usual.

When the clip is the whole state, pass `"{}"`. An empty string is not rewritten. A decision takes an image or a clip, not both.

## Vision

[`VisionDecision`] loads the same snapshot and the SigLIP2 vision tower, and skips the audio tower. `choice`, `noul`, and `score` return the same answer types as text mode. Each call is one question. A later call with the same image bytes reuses the vision embeddings and runs the new question.

```rust
use fuzzy_decision::{RgbImage, VisionDecision};

fn main() -> Result<(), fuzzy_decision::Error> {
    let decider = VisionDecision::load("models/d1-omni-600M")?;
    let image = RgbImage {
        width: 320,
        height: 320,
        data: std::fs::read("picture.rgb").expect("rgb bytes"),
    };
    let answer = decider.choice(
        &image,
        "",
        "What is the main subject?",
        &["a red circle", "a blue square"],
    )?;
    println!("{}", answer.choice);
    Ok(())
}
```

Images follow the reference layout: sides snap with a factor of 32, a tile is 512 pixels, and large pictures are tiled. Resize is antialiased bilinear in floating point, then rounded to bytes. That rounding can differ by about one level from torchvision's integer resize.

## Evals

`cargo run --release --example domain_eval` loads `models/d1-omni-600M` and scores 1000 professional and everyday classification questions across 21 domains, then writes the report to `text_eval.html`. `cargo run --release --example vision_suite`, `vision_judge`, and `image_eval` score packed RGB images from a manifest. `cargo run --release --example audio_decision` scores one WAV. The examples do not download weights.

## Limits

Text context is **16384** tokens. An image leaves room for 896 text tokens, and a clip leaves room for 15360, after the media prefix is subtracted. If fewer than 64 text tokens remain, the call returns `Error::Context`. `max_length` on `LoadOptions` can only lower that cap.

Count tokens before you call, with the same tokenizer the model uses:

```rust
let n = decider.count_tokens("alpha beta gamma delta");
```

Choice allows 2 to 255 options. Score allows 2 to 10 levels, in order. Instructions and options must be non-empty, and option labels must be unique. An empty question list is valid and returns no answers without running the model.

By default a state that is too long is cut to the limit (`Truncation::Cut`). `Truncation::Error` fails instead:

```rust
use fuzzy_decision::{DecideOptions, Truncation};

let answers = decider.decide(
    state,
    &questions,
    DecideOptions::default()
        .truncation(Truncation::Error)
        .max_state_tokens(256),
)?;
```

`max_state_tokens` and `truncation` on `DecideOptions` apply to that call. The same methods on `LoadOptions` set the default for every later call.

A state that does not fit in strict mode returns `Error::Truncated` with `state_tokens` and `kept`.

## Temperature

Temperature must be finite and greater than zero. Text questions otherwise use the matching entry in `config.json` (`choice:2`, `choice:3-5`, `choice:6-10`, `choice:11+`, and the same pattern for `score` and `noul`). `1.0` leaves those logits at the calibrated scale only when the config entry is `1.0`. Values you pass replace the config value; they are not multiplied by it. Values below `1.0` make the top option sharper. Values above `1.0` flatten the distribution. The chosen label can change. The call still does not sample.

```rust
let sharp = decider.choice_with(
    state,
    "Which area?",
    &["fees & charges", "refund & dispute", "other"],
    None,
    DecideOptions::default().temperature(0.2),
)?;
```

`noul_with` and `score_with` take the same `DecideOptions`. A bad temperature returns `Error::Temperature` before the forward pass. Setting it on `LoadOptions` rejects it at `load`. `DecideOptions` wins over `LoadOptions` when both are set.

## Errors

`open`, `noul`, `choice`, `score`, and `decide` return `Result<_, fuzzy_decision::Error>`. The text from `Display` is the message you can show.

| Error | When |
| --- | --- |
| `MissingFile` | `tokenizer.json`, `config.json`, or `model.safetensors` is not in the directory |
| `UnsupportedModel` | `model` is not `d1-omni-600M` |
| `Weights` | a file is present but the tensors cannot be read |
| `Audio` | a clip is empty, non-finite, or the WAV cannot be read |
| `EmptyInstructions` | instructions are blank |
| `BadOption` | an option or level is blank |
| `DuplicateOption` | the same label appears twice |
| `OptionCount` | choice or score has too few or too many entries |
| `UnknownOption` | `probability` was asked for a label that was not passed |
| `Temperature` | temperature is not finite or is not greater than zero |
| `Truncated` | strict truncation and the state does not fit |
| `Context` | the options, or the media prefix, do not leave a usable text budget |
| `Row` | kept for callers matching on it; this checkpoint does not produce it |

Drop the `FuzzyDecision` value when you are finished. That releases the model.

## What the model sees

One question is one sequence:

`<bos> <state> state <q> instructions <opt> <mask> option </opt> ... <decide>`

The decision head reads the hidden state at each `<mask>`. The delimiters are the tokenizer's reserved special tokens. User text is encoded without adding another start token. Media embeddings, when present, sit in front of that sequence and attend only to themselves. Text attends to the media and to the text.

Yes/no options are `false` then `true`, which this crate reports as no then yes. `probability` is the yes probability.

Text that contains a delimiter spelling such as `<|reserved_7|>` is escaped before it is tokenized, so user text cannot close the state early.

## Breaking changes from 0.5

- The checkpoint moved from `Cloudflare/clef-flash` to `LiquidAI/d1-omni-600M`. `DEFAULT_MODEL` is `"d1-omni-600M"` and `MODEL_REPO` is `"LiquidAI/d1-omni-600M"`.
- The default weights directory is `models/d1-omni-600M`. The snapshot is `tokenizer.json`, `config.json`, and one f32 `model.safetensors`.
- Each question is its own forward pass. Questions in one `decide` call do not influence each other. `decide_map` keys are not part of the prompt.
- Choice requires at least 2 options. Score allows at most 10 levels.
- Text questions use the temperatures in `config.json`. Image and audio questions use temperature 1 unless you set one. A temperature you set replaces the config value.
- Audio input is `AudioClip` plus `decide_audio`, `choice_audio`, `noul_audio`, and `score_audio`. When the clip is the whole state, pass `"{}"`.
- Vision still uses `VisionDecision` on the same snapshot. Image resize can differ by about one byte level from torchvision.
