# fuzzy-decision

`fuzzy-decision` scores questions you write. Each answer is a probability over the options you supplied. The model does not write new text. Text and image questions both use [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash). The forward pass runs on [Burn](https://burn.dev) 0.21 with the WGPU backend. This crate reads weights from a directory you pass. It does not download them. The crate is MIT. The checkpoint is separate and Apache-2.0.

**Clef-Flash** is Cloudflare's 9B multimodal decision model, post-trained from `Qwen/Qwen3.5-9B`. It takes a state (text, with an optional image) plus a schema of typed questions — yes/no (`noul`), `choice`, and ordered `score` — and returns logits for every option of every question in one forward pass, through a joint schema head that routes evidence from the state to each question. Text mode skips the vision tower; vision mode loads it. Both modes read the same snapshot directory.

One process holds one loaded model. Later calls reuse it.

## Add the crate

```toml
[dependencies]
fuzzy-decision = "0.5"
```

A call needs a GPU that WGPU can see (Vulkan, Metal, or DX12). The checkpoint is stored as bf16 (about 19 GB on disk) and this crate runs it as f32 on the device, so plan for roughly 36 GB of GPU memory plus activations. The two token tables (embedding and `lm_head`) stay on the CPU in bf16 to halve host RAM.

## Weights

Put a local `Cloudflare/clef-flash` snapshot in one directory, then pass that directory to `FuzzyDecision::open`. `FuzzyDecision::load(LoadOptions::default())` looks for `models/clef-flash` relative to the process working directory. A missing file returns `Error::MissingFile` with the directory and the file name.

| File in that directory | Purpose |
| --- | --- |
| `tokenizer.json` | the tokenizer |
| `model.safetensors.index.json` | maps tensor names to shards |
| `model-00001-of-00004.safetensors` … `model-00004-of-00004.safetensors` | the backbone and vision tower, bf16 |
| `joint_head.safetensors` | the joint schema head |
| `joint_head_config.json` | head dimensions |

Fetch them from the application that embeds this library, once, before the first `open`. The library never calls the network.

```bash
DIR=models/clef-flash
mkdir -p "$DIR"
for f in tokenizer.json model.safetensors.index.json \
         model-00001-of-00004.safetensors model-00002-of-00004.safetensors \
         model-00003-of-00004.safetensors model-00004-of-00004.safetensors \
         joint_head.safetensors joint_head_config.json; do
  curl -L --fail -o "$DIR/$f" \
    "https://huggingface.co/Cloudflare/clef-flash/resolve/main/$f"
done
```

No token is required; the repository is public. If you mirror it behind authentication, pass your own token to `curl` (for example `-H "Authorization: Bearer $HF_TOKEN"`) — this crate never reads one.

Then `FuzzyDecision::open(DIR)`. Keep the directory next to the application, or set `LoadOptions { weights_dir: Some(path), .. }`. Do not commit the files.

Only `clef-flash` loads. Any other `LoadOptions.model` returns `Error::UnsupportedModel`. Weights are `f32` on the default WGPU device.

## One question

```rust
use fuzzy_decision::FuzzyDecision;

fn main() -> Result<(), fuzzy_decision::Error> {
    let decider = FuzzyDecision::open("models/clef-flash")?;
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

The same state and the same options return the same answer. The library does not sample. The default temperature is `1.0`, and a temperature only rescales the logits before the same softmax.

## Several questions in one pass

`decide` packs every question into one forward pass. Clef-Flash scores the schema jointly: every question sees the state and the other questions, and the head reads all of them at once. Asking related questions together is the intended use; an answer can shift slightly when the surrounding questions change.

```rust
use fuzzy_decision::{choice, noul, score, Answer, DecideOptions, FuzzyDecision};

let decider = FuzzyDecision::open("models/clef-flash")?;
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

Answers come back in the same order as the questions. A description can be attached to a choice option; the model sees the description in the option's JSON while the returned label stays the name:

```rust
choice(
    "Which area?",
    &["fees", "refund"],
    Some(&[("fees", "billing mistakes and extra charges")]),
)
```

`decide_map` takes a `BTreeMap<String, Question>` and returns a `BTreeMap<String, Answer>` with the same keys. Map order is sorted by key, and that sorted order is the order the questions are packed. The keys are used as the schema's field identifiers, so the model can see them; pick short descriptive keys.

## Modes

Text is the default. [`FuzzyDecision`] loads Clef-Flash without the vision tower and scores typed questions with the joint schema head. There is no image input.

Vision is [`VisionDecision`]. It loads the same `Cloudflare/clef-flash` snapshot from a directory you prepare (the library does not download it) and adds the vision tower. `choice`, `noul`, and `score` return the same answer types as the text mode. Each call is still one question. A later call with the same image bytes and the same state reuses the picture encoding and the language-model state up to that state, and runs only the new question.

```rust
use fuzzy_decision::{RgbImage, VisionDecision};

fn main() -> Result<(), fuzzy_decision::Error> {
    let decider = VisionDecision::load("models/clef-flash")?;
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

Images are resized with the checkpoint's smart-resize rule (sides snap to multiples of 32) and capped so one picture stays near a thousand language-model tokens.

## Evals

`cargo run --release --example domain_eval` loads `models/clef-flash` and scores 1000 professional and everyday classification questions across 21 domains, then writes the report to `text_eval.html`. `cargo run --release --example vision_suite` and `cargo run --release --example vision_judge` score packed RGB images from a manifest. The examples do not download weights. Reports from the previous checkpoint were removed; rerun the examples to produce fresh ones.

## Limits

The packed record — the state, the schema with every question, and the template tokens around them — can be at most **16384** tokens. The state portion is capped separately by `max_state_tokens` (also 16384 by default; the template and schema take tokens out of what fits).

Count tokens before you call, with the same tokenizer the model uses:

```rust
let n = decider.count_tokens("alpha beta gamma delta");
```

Choice allows 1 to 255 options. Score allows 2 to 255 levels, in order. Instructions and options must be non-empty, and option labels must be unique. An empty question list is valid and returns no answers without running the model.

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

`max_state_tokens` and `truncation` on `DecideOptions` apply to that call. The same methods on `LoadOptions` set the default for every later call. `max_length` on `LoadOptions` is the whole-record cap. Setting either cap above 16384 is allowed by the checkpoint's rotary embedding but is untested territory.

A state that does not fit in strict mode returns `Error::Truncated` with `state_tokens` and `kept`.

## Temperature

Temperature must be finite and greater than zero. `1.0` leaves the logits alone. Values below `1.0` make the top option sharper. Values above `1.0` flatten the distribution. The chosen label can change. The call still does not sample.

```rust
let sharp = decider.choice_with(
    state,
    "Which area?",
    &["fees & charges", "refund & dispute", "other"],
    None,
    DecideOptions::default().temperature(0.2),
)?;
```

`noul_with` and `score_with` take the same `DecideOptions`. A bad temperature returns `Error::Temperature` before the forward pass. Setting it on `LoadOptions` rejects it at `load`.

## Errors

`open`, `noul`, `choice`, `score`, and `decide` return `Result<_, fuzzy_decision::Error>`. The text from `Display` is the message you can show.

| Error | When |
| --- | --- |
| `MissingFile` | `tokenizer.json`, `model.safetensors.index.json`, a shard it names, `joint_head.safetensors`, or `joint_head_config.json` is not in the directory |
| `UnsupportedModel` | `model` is not `clef-flash` |
| `Weights` | a file is present but the tensors cannot be read |
| `EmptyInstructions` | instructions are blank |
| `BadOption` | an option or level is blank |
| `DuplicateOption` | the same label appears twice |
| `OptionCount` | choice or score has too few or too many entries |
| `UnknownOption` | `probability` was asked for a label that was not passed |
| `Temperature` | temperature is not finite or is not greater than zero |
| `Truncated` | strict truncation and the state does not fit |
| `Row` | kept for callers matching on it; Clef-Flash packs one record, so it is no longer produced |

Drop the `FuzzyDecision` value when you are finished. That releases the model.

## What the model sees

The packer renders the state and the schema into one chat-template record. The state sits in a `STATE:` section; each question becomes a `FIELD` block with its instructions and one JSON line per option. The joint schema head pools the hidden states over each span, routes evidence from the state to every question, and combines a lexical prior with a learned joint score per option. The softmax is over each question's options.

Text that contains a template spelling such as `<|im_start|>` is escaped before it is tokenized, so user text cannot close the state early.

## Breaking changes from 0.4

- The checkpoint moved from Kev-4B (`jaredpalmer/kev-4b` + `Qwen/Qwen3-4B-Base`) to `Cloudflare/clef-flash`. The constants `BASE_REPO` and `ADAPTER_REPO` were replaced by `MODEL_REPO`, and `DEFAULT_MODEL` is now `"clef-flash"`.
- The default weights directory is `models/clef-flash` and the file set changed (see Weights above).
- Questions in one `decide` call are scored jointly and can influence one another; in 0.4 each question was isolated by the attention mask.
- `decide_map` keys are now visible to the model as field identifiers.
- The token limits rose from 8192 to 16384, and the whole record shares one budget instead of a per-question row limit, so `Error::Row` is no longer produced.
- `VisionDecision` uses the same clef-flash snapshot instead of `yah01/vjev-vision`; vision `confidence` is now the chosen option's probability and vision `noul` uses a softmax over yes/no.
