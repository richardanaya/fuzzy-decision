# fuzzy-decision

`fuzzy-decision` scores a piece of text against questions you write. Each answer is a probability over the options you supplied. The model does not write new text.

The loaded checkpoint is **Kev-0.6B**: `Qwen/Qwen3-0.6B-Base`, the LoRA from `jaredpalmer/kev-0.6b` merged in at load, and that repo's pointer head. The forward pass runs on [Burn](https://burn.dev) 0.21 with the WGPU backend. The library reads weights from disk. It does not download them.

One process holds one loaded model. Later calls reuse it.

## Add the crate

```toml
[dependencies]
fuzzy-decision = "0.1"
```

A call needs a GPU that WGPU can see (Vulkan, Metal, or DX12). The first `open` compiles shaders and loads about 1.2 GB of weights. On the machine this was timed on, that load was about two seconds, and a short question after warmup was about 160 ms.

## Weights

Create `models/kev-0.6b/` and put these four files in it:

| File | Where it comes from |
| --- | --- |
| `model.safetensors` | [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) |
| `adapter_model.safetensors` | [jaredpalmer/kev-0.6b](https://huggingface.co/jaredpalmer/kev-0.6b) |
| `tokenizer.json` | the same Kev repo |
| `head.safetensors` | the pointer head from that repo's `head.pt` |

`head.pt` is a PyTorch zip. This crate reads safetensors, so convert the four float32 tensors (`q.weight`, `q.bias`, `k.weight`, `k.bias`) into `head.safetensors` before you call `open`. Shapes are `q`/`k` weight `[256, 1024]` and bias `[256]`.

`FuzzyDecision::open("models/kev-0.6b")` reads that directory. `FuzzyDecision::load(LoadOptions::default())` looks for `models/kev-0.6b` relative to the process working directory. A missing file returns `Error::MissingFile` with the directory and the file name.

Kev-4B and the other published Kev sizes are not loaded. `LoadOptions { model: "kev-4b", .. }` returns `Error::UnsupportedModel`.

## One question

```rust
use fuzzy_decision::FuzzyDecision;

fn main() -> Result<(), fuzzy_decision::Error> {
    let decider = FuzzyDecision::open("models/kev-0.6b")?;
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

`decide` packs every question into one forward pass. Each question sees the state and itself, and does not see the other questions.

```rust
use fuzzy_decision::{choice, noul, score, Answer, DecideOptions, FuzzyDecision};

let decider = FuzzyDecision::open("models/kev-0.6b")?;
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

Answers come back in the same order as the questions. A description can be attached to a choice option; the model sees `name: description` while the returned label stays the name:

```rust
choice(
    "Which area?",
    &["fees", "refund"],
    Some(&[("fees", "billing mistakes and extra charges")]),
)
```

`decide_map` takes a `BTreeMap<String, Question>` and returns a `BTreeMap<String, Answer>` with the same keys. Map order is sorted by key, and that sorted order is the order the questions are packed.

## Limits

The state, including its delimiter token, can be at most **8192** tokens. The state plus any one question (instructions, options, and the decide token) can also be at most **8192** tokens. Positions start again at the beginning of each question, after the shared state.

Count tokens before you call, with the same tokenizer the model uses:

```rust
let n = decider.count_tokens("alpha beta gamma delta");
```

`count_tokens` counts the text. The packer also inserts one delimiter in front of the state, so a limit check is `count_tokens(state) + 1`.

Choice allows 1 to 255 options. Score allows 2 to 255 levels, in order. Instructions and options must be non-empty, and option labels must be unique. An empty question list is valid and still runs the model on the state, then returns no answers.

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

`max_state_tokens` and `truncation` on `DecideOptions` apply to that call. The same methods on `LoadOptions` set the default for every later call. `max_length` on `LoadOptions` is the state-plus-one-question cap. Setting either cap above 8192 walks off the end of the position table.

A question that still does not fit next to the state returns `Error::Row` with `state_tokens`, `question_tokens`, and `limit`. A state that does not fit in strict mode returns `Error::Truncated` with `state_tokens` and `kept`.

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
| `MissingFile` | `model.safetensors`, `adapter_model.safetensors`, `head.safetensors`, or `tokenizer.json` is not in the directory |
| `UnsupportedModel` | `model` is not `kev-0.6b` |
| `Weights` | a file is present but the tensors cannot be read |
| `EmptyInstructions` | instructions are blank |
| `BadOption` | an option or level is blank |
| `DuplicateOption` | the same label appears twice |
| `OptionCount` | choice or score has too few or too many entries |
| `UnknownOption` | `probability` was asked for a label that was not passed |
| `Temperature` | temperature is not finite or is not greater than zero |
| `Truncated` | strict truncation and the state does not fit |
| `Row` | one question plus the state exceeds the row limit |
| `Disposed` | `dispose` was called and a later call used the same value |

`dispose` marks the value so later calls fail. Dropping it is enough if you are finished.

## What the model sees

The packer turns the state and each question into one token sequence. Special tokens separate the state, the question, each option, and the point where the decision is read. A question attends to the state and to its own tokens. The pointer head compares the hidden state at the decision token with the hidden state at the end of each option, and the softmax is over those scores.

Text that contains a delimiter spelling such as `<|fim_prefix|>` is escaped before it is tokenized, so user text cannot close the state early.
