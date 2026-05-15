# HAT-mode effect approximations (draft)

The v10 dispatcher receives the same `set_lights(color?, effect?, state?)` call
regardless of which LED hardware is connected. In **strip mode** (WLED detected),
effects map directly to native WLED palettes/effects. In **HAT mode** (no WLED),
the dispatcher approximates the same intent on the 3 discrete LEDs of the Grinn
HAT (red, green, blue, individually addressable, brightness 0-100 each).

Approximations are *charming, not literal* — the goal is for the take-home build
to feel responsive and visible even on minimal hardware, not to claim parity with
the strip.

The v10 schema dropped `brightness`, `count`, and `speed` from `set_lights` for
robustness on the 270M. The dispatcher uses the following defaults, baked in per
effect (not user-controllable):

- brightness: 100% always (HAT) / WLED's default for the chosen effect (strip)
- blink/pulse count: 3 repetitions
- speed: "normal" (per-effect periods in the table below)

## Effect mapping table

| effect       | HAT approximation                                                                  |
|--------------|------------------------------------------------------------------------------------|
| `solid`      | Color → that LED on full. No color → all 3 on.                                     |
| `blink`      | Color → blink that LED 3 times. No color → blink all together.                     |
| `pulse`      | Smooth brightness ramp up→down on specified LED (or all). Loop until canceled.     |
| `fade`       | Slow brightness ramp in then out, single cycle. Color → that LED.                  |
| `rainbow`    | R→G→B→R cycle, one LED at a time.                                                  |
| `fire`       | Red LED with rapid random brightness flicker (60-100%, jittered).                  |
| `plasma`     | Red and blue alternating soft pulses, slow drift.                                  |
| `aurora`     | Green LED slow soft pulse (5-30% range, ~3s period).                               |
| `police`     | Red and blue alternation at ~4 Hz. The signature HAT-approximation moment.         |
| `fireworks`  | All three LEDs flash random colors in bursts, with decaying brightness.            |
| `sparkle`    | Random LED short flashes (single-blink) at random intervals.                       |
| `twinkle`    | Slow gentle random LED pulses, low intensity.                                      |
| `chase`      | Sequential R→G→B→R single-LED-at-a-time, brightness 100%.                          |
| `comet`      | Brightness wave R→G→B→R, each LED ramps up then down before the next starts.       |
| `heartbeat`  | Double-pulse pattern on red (bright-dim-bright-rest, ~1Hz outer rhythm).           |
| `lightning`  | All LEDs sudden bright flash (100%), rapid fade to 0, occasional repeat strike.    |
| `glitter`    | Dense rapid sparkle on all three LEDs simultaneously.                              |
| `loading`    | Rotating dot R→G→B→R indefinitely.                                                  |
| `sunrise`    | Slow gradient: red dim → red bright → red+green → all on. ~15s total ramp.         |
| `off`        | All LEDs off.                                                                       |

## Color mapping in HAT mode

User-specified colors map to the closest physical LED combo at full brightness:

| color     | HAT LEDs lit                          |
|-----------|---------------------------------------|
| `red`     | red                                   |
| `green`   | green                                 |
| `blue`    | blue                                  |
| `yellow`  | red + green                           |
| `cyan`    | green + blue                          |
| `purple`  | red + blue                            |
| `pink`    | red 100% + blue 30%                   |
| `orange`  | red 100% + green 30%                  |
| `white`   | all three                             |
| (unknown) | all three (closest "neutral" fallback) |

Combo colors (`yellow`, `cyan`, etc.) are obviously non-physically-accurate but
visibly distinct from the single-LED colors, which is what matters for the demo.

## Effect default periods (HAT mode, baked-in)

Now that `speed` is gone from the user-facing schema, the dispatcher uses these
periods as constants (in ms):

| effect       | period |
|--------------|--------|
| `blink`      | 400    |
| `pulse`      | 1000   |
| `rainbow`    | 800    |
| `chase`      | 300    |
| `comet`      | 600    |
| `police`     | 250    |
| (others)     | varies per effect, tuned for visible animation |

## Open questions

1. **Threading model.** Many effects loop indefinitely (rainbow, fire, aurora,
   police, loading, plasma, twinkle, sparkle, glitter). HAT mode needs a
   background loop thread per active effect, cancelable on the next
   `set_lights` call. Strip mode delegates to WLED firmware. The dispatcher
   needs to own this lifecycle.

2. **Brightness floor.** A few approximations (sunrise, aurora) go to very low
   brightness (5-10%). Some LEDs are noticeably non-linear at that range. May
   need per-LED gamma correction in `hardware.py`.

3. **Two-LED combo colors at full brightness.** Hardware-wise, red 100% + green
   100% draws more current than a single LED at 100%. Should be fine on the
   HAT's GPIO drivers but worth confirming on board.
