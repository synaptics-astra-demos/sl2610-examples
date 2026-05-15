# v10 training-data generation spec

## Goals (concrete + measurable)

1. **100% routing on the 5 voice failures from the May 14 log.** Every one of "Something exciting on the new pixels", "Turn the Neo pixels on", "Light up the Neo Pixels", "Set the Neo Pixels on", "Turn the led strip" must produce a sane `set_lights(...)` call, not a wrong tool / garbage args / fallback-respond.
2. **≥95% routing on the 12 chip prompts** in `chat_window.PROMPT_CATEGORIES` (mix of HAT colors, strip effects, system status, alarms, buzzer).
3. **Robust to Moonshine ASR noise** on light-related prompts (article drops, capitalization, common substitutions).
4. **Non-light tools (buzzer/alarms/status/respond) retain v9 quality** — they shouldn't regress just because we reshuffled the LED surface.

## Schema reminder

- 6 tools: `set_lights`, `play_buzzer`, `set_alarm`, `cancel_alarm`, `get_system_status`, `respond`
- `set_lights(color?, effect?, state?)` — all optional named args
- Output: `<tool_N>(name1="value1", name2="value2")<end>`
- Prompt: `<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n` (Octopus-v2 pure, no schema)

## Volume target

- **~7,000 rows total**, **~14 rows per template** (v9 ratio).
- Stratified eval holdout: 15% (~1,050 rows), per-intent stratification so each category has eval coverage.

Allocation per tool:

| Tool | Rows | Notes |
|---|---|---|
| `set_lights` | 4,800 | Absorbs v9's three LED tools. See intent breakdown below. |
| `play_buzzer` | 400 | 7 patterns × phrasings |
| `set_alarm` | 500 | duration + time + label combinations |
| `cancel_alarm` | 200 | with/without label |
| `get_system_status` | 400 | per-metric + "all" phrasings |
| `respond` | 700 | conversational + out-of-scope + ambiguous |
| **Total** | **7,000** | |

## `set_lights` intent breakdown (4,800 rows)

| Intent | Rows | Target shape | Example |
|---|---|---|---|
| Color only | 800 | `set_lights(color="red")` | "lights to red", "make the LEDs blue" |
| Effect only | 1,400 | `set_lights(effect="rainbow")` | "rainbow", "police effect", "fireworks" |
| Color + effect | 1,000 | `set_lights(color="blue", effect="pulse")` | "pulse blue", "blue rainbow", "red fire" |
| State (on/off) | 400 | `set_lights(state="off")` | "lights off", "turn them on" |
| Color + state | 600 | `set_lights(color="red", state="on")` | "turn the red light on" |
| Implicit / vague effects | 400 | `set_lights(effect="aurora")` for "soothing", etc. | "something soothing", "make it exciting" |
| ASR-variant `neopixels` phrasings | 200 | maps to whichever `set_lights(...)` is right | "Neo pixels on" → `set_lights(state="on")` |

The 200 ASR-variant rows are the bridge to the demo's voice reality: we don't *promote* "neopixels" vocabulary (per Google), but the model will see "neo pixels" / "new pixels" / "led strip" enough to route them correctly. They're a small fraction so they don't dominate.

## Entity pools (`coral_v10_entities.py`)

```python
COLORS = ["red", "green", "blue", "white", "yellow",
          "purple", "orange", "pink", "cyan", "magenta"]

# All 19 effects from KNOWN_EFFECTS in lights.py, minus "solid" and "off"
# (which are handled via state semantics or naturally)
EFFECTS = ["blink", "pulse", "fade", "rainbow", "fire", "plasma",
           "aurora", "police", "fireworks", "sparkle", "twinkle",
           "chase", "comet", "heartbeat", "lightning", "glitter",
           "loading", "sunrise"]

LIGHT_NOUNS = [
    "light", "lights", "LED", "LEDs",
    "indicator", "indicators", "status light", "status lights",
    "strip", "led strip", "light strip",
]
# Deliberately NOT in main rotation: "neopixels", "neo pixels", "new pixels"
# — those live only in the 200-row ASR-variant slice.

VERBS_ON = ["turn on", "switch on", "light up", "activate"]
VERBS_OFF = ["turn off", "switch off", "kill", "deactivate", "stop"]
VERBS_SET = ["set", "make", "change"]

VAGUE_TO_EFFECT = {
    "soothing": "aurora",     "calming": "aurora",
    "exciting": "fireworks",  "wild": "fireworks",
    "fun": "rainbow",         "party": "rainbow",
    "chill": "fade",          "intense": "fire",
    "dramatic": "lightning",  "alert": "police",
    "festive": "twinkle",     "subtle": "fade",
    "gentle": "fade",         "energetic": "rainbow",
}
```

## Template format (per-intent JSON files)

Same as v9 (`data/v9_templates/*.json`):

```json
{
  "intent": "set_lights_color_effect",
  "templates": [
    "{verb} the {light_noun} {color} {effect}",
    "{effect} the {light_noun} in {color}",
    "{color} {effect} on the {light_noun}",
    "{effect} {color}"
  ],
  "target": "set_lights(color=\"{color}\", effect=\"{effect}\")"
}
```

Generator cross-products each template against the entity pools. Each combination becomes one training row.

## ASR-noise augmentation

For ~10% of rows, apply 1-2 random transforms from this set:

- **Drop articles**: "turn the lights on" → "turn lights on"
- **Drop trailing punctuation**: "rainbow." → "rainbow"
- **Case noise**: random title-case / lowercase / "Sentence case"
- **Auxiliary verb prefix**: "rainbow" → "can you do rainbow"
- **Filler suffix**: "lights off" → "lights off please"
- **Trailing period removal/addition**

Noise is sampled per row. Goal: model learns to ignore surface variation, not to overfit to clean training English.

## Failure-mode coverage (hand-curated)

A small set (~50 rows) of *exactly* the prompts we've observed failing, with the target call we want. These are NOT noise-augmented — they're verbatim.

```
"Something exciting on the new pixels." → set_lights(effect="fireworks")
"Turn the Neo pixels on."                → set_lights(state="on")
"Light up the Neo Pixels."               → set_lights(state="on")
"Set the Neo Pixels on."                 → set_lights(state="on")
"Turn the led strip."                    → set_lights(state="on")
```

Plus the v9 chip-strip prompts (verbatim from `PROMPT_CATEGORIES`) with v10-correct targets, so chip clicks stay green.

## Output format

JSONL, one record per row, matching SmartPanel v15 shape:

```json
{
  "messages": [{"role": "user", "content": "rainbow on the lights"}],
  "completion": "<tool_0>(effect=\"rainbow\")<end>"
}
```

Trainer reads with `completion_only_collator` (mask everything before `<start_of_turn>model\n`).

## Implementation plan

1. **`scripts/coral_v10_entities.py`** — entity pools + vague→effect map (above).
2. **`data/v10_templates/*.json`** — one file per intent. Templates seeded by hand for structure (~10 patterns each), then expanded via Haiku agents in parallel ("pane 0") to ~40-60 patterns per intent.
3. **`scripts/generate_v10.py`** — generator: load templates, cross-product with entities, apply noise, format as JSONL, stratified 85/15 split.
4. **Spot-check 50 random rows + check class balance** (per-intent counts close to targets).
5. **Hand-curate `data/v10_failure_modes.jsonl`** — the 5 voice failures + chip prompts.

## Open questions before I start writing code

1. **Vague descriptor mappings.** Is my `VAGUE_TO_EFFECT` table sensible? Specifically: "exciting" → "fireworks", "soothing" → "aurora", "party" → "rainbow" — these are my judgment calls and the model will *commit* to whatever we train. Worth a 30-second review.
2. **Should the 200-row "neopixels-variant" slice exist at all, or is the risk that the model learns to expect that vocabulary?** My read: include it, because Moonshine reliably produces these phrasings on voice input and the model has to handle them. The user wanting to drop "neopixels" was about marketing/UX, not about discarding observed inputs.
3. **Class weight in training, or rely on row counts alone?** SmartPanel v15 used vanilla `SFTTrainer` with no class weighting. I'd mirror that. Worth confirming.
4. **Volume — 7k vs more.** v9 was 7.5k. We could push to 10k for more robustness. Cost: ~5 extra minutes of Haiku-template generation, training time unchanged. Inclined to start at 7k and only expand if eval shows weakness.

## What this spec does NOT cover (separately)

- v10 trainer script (`train_coral_v10.py`) — derives from `train_smartpanel_v15.py` verbatim, swaps token map + max_length=256.
- GGUF conversion + quantization (existing recipe in CLAUDE.md works).
- On-board deploy (atomic tools.json + token_map.json + GGUF swap).
