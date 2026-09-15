# Vision-Configured Analog Oscillator

The current analog hardware build and LTspice model are complete. A manual ten-photo computer-vision capture preflight is also complete; model training, potentiometer calibration, physical waveform measurements, and automated LTspice control remain in progress.

## Demonstration

![Completed breadboard](hardware/media/completed-breadboard.JPG)

The completed 5 V breadboard oscillator provides three persistent controls: one potentiometer and two independently switched timing capacitors. A momentary button discharges an RC envelope, gradually reducing the buzzer-driver current while the LED fades. Releasing the button allows both to recover.

- [Watch the complete hardware demonstration](hardware/media/hardware-demo.mp4)
- [Watch the envelope and LED close-up](hardware/media/envelope-fade-closeup.mp4)

## Circuit overview

![Final LTspice schematic](ltspice/final-schematic.png)

The core is a 5 V two-transistor astable multivibrator. A nominal 10 kΩ potentiometer changes the shared timing resistance, while each maintained switch adds a 100 nF capacitor to one side of the timing network. Together, the switches provide four timing configurations, including symmetric and asymmetric waveforms.

The `O1` output passes through a 2 kΩ/100 nF low-pass filter with a nominal cutoff near 796 Hz. Q3 isolates this filtered signal from the output stage. Q5 drives the passive buzzer, while Q4 controls Q5’s emitter-return path from the `ENV` voltage.

Holding the momentary button discharges the 300 µF envelope through 1 kΩ, reducing buzzer current and fading the LED. After release, the envelope recharges through 10 kΩ and both outputs recover.

![Physical controls](hardware/media/controls-closeup.JPG)

## LTspice simulation

The final circuit was simulated in LTspice 26.0.2. The reference run uses `Rp = 5.1 kΩ`, with both additional timing capacitors disconnected; `1 pF` represents each open switch. A 12-second transient simulation applies a one-second button hold from 6–7 seconds.

- [Open the LTspice schematic](ltspice/analog-oscillator-v8.asc)
- [View the simulation results](ltspice/simulation-results.txt)

### Filter response

![Raw and filtered oscillator waveforms](ltspice/filter-response.png)

The `FILT` waveform has slower edges, lower amplitude, and phase delay relative to the raw `O1` waveform.

### Envelope response

![Simulated envelope response](ltspice/envelope-response.png)

The button discharges `ENV`, reducing both LED current and modeled buzzer current. After release, the 300 µF envelope recharges gradually.

## Results

The final reference simulation produced:

| Measurement | Result |
|---|---:|
| Normal buzzer RMS current | 5.441 mA |
| Held-button buzzer RMS current | 3.094 µA |
| Recovered buzzer RMS current | 5.362 mA |
| Normal oscillator frequency | 389.428 Hz |
| Held-button oscillator frequency | 389.508 Hz |
| Simulated frequency change | 0.0204% |

The model predicts approximately a 99.94% reduction in buzzer RMS current while the button is held, without materially changing oscillator frequency. This supports the intended separation between the timing oscillator and the output envelope.

The physical breadboard reproduces the control behavior qualitatively: the potentiometer and both timing switches change the tone, while the button fades the LED and suppresses the buzzer before gradual recovery. Physical frequency and waveform measurements have not yet been collected with an oscilloscope.

## Design history

This project began as a physics-class demonstration of a two-transistor astable multivibrator. The potentiometer replaced one timing resistor, while two momentary buttons independently connected additional capacitors. The oscillator drove both a passive buzzer and two alternating LEDs. Holding either button lowered the frequency, and the added capacitance slowed the frequency enough for the LEDs’ alternating blinking to become directly visible.

![Original physics-class prototype](hardware/history/original-prototype.jpg)

- [Watch the original prototype demonstration](hardware/history/original-prototype-demo.mp4)

| Stage | Main change | Result |
|---|---|---|
| Physics-class prototype | Used a potentiometer in one timing branch, two button-switched capacitors, two alternating LEDs, and a passive buzzer | Demonstrated how resistance and capacitance change audible pitch and visible oscillation rate |
| Rebuilt oscillator | Changed to a shared timing potentiometer, two maintained capacitor switches, and a fixed RC-filtered buzzer path | Produced four persistent timing configurations suitable for simulation and later visual recognition |
| Output-control experiments | Tested raw/filtered signal mixing, a volume-accent path, a button-controlled pitch drop, and several RC-envelope arrangements | Raw/filter mixing and the volume accent did not create a distinct effect; the pitch drop worked but repeated the function of the timing capacitors; early envelope circuits loaded or altered the oscillator instead of independently controlling the output |
| Final v8 | Isolated the filtered signal with Q3, used Q4 and Q5 for envelope-controlled buzzer current, increased the envelope to 300 µF, and added an LED indicator | Kept the oscillator frequency nearly unchanged during the button action while making the envelope directly visible |

The current build keeps the original project’s oscillator, passive buzzer, and adjustable timing concept, but separates oscillator timing from the momentary output effect.

## Limitations

- The physical circuit has been tested by sound and visible LED behavior, but its frequency, duty cycle, and node voltages have not yet been measured with an oscilloscope.
- The potentiometer is nominally 10 kΩ but has not yet been calibrated from physical angle to measured resistance.
- LTspice uses generic NPN models and represents the passive buzzer as a 16 Ω resistive load. The simulation therefore predicts electrical behavior, not exact loudness or acoustic response.
- The LED shows the envelope more clearly than the passive buzzer reproduces it acoustically.
- The current breadboard is a single hand-wired prototype, so component tolerances and wiring parasitics are not characterized.
- The computer-vision model has not been trained or evaluated. The manual capture preflight establishes only that the marked controls remain identifiable after the planned input resize.

## Computer-vision phase

The next phase will test whether a compact CNN can recover the persistent control configuration of this fixed breadboard from a saved photograph. It will estimate board rotation, potentiometer center and pointer direction, and the independent states of the two maintained switches. The momentary button, LED brightness, and buzzer state are not targets.

The primary implementation will use a small PyTorch CNN trained from scratch. A separate Python controller will combine accepted predictions with measured potentiometer calibration and a verified switch-to-capacitor mapping, then run LTspice to estimate frequency and waveform behavior.

A manual ten-photo capture preflight confirmed that the orange potentiometer pointer and blue switch tabs remained readable at the planned `576 x 768` input size across moderate changes in rotation, perspective, lighting, and framing. This was not model evaluation; no model has been trained.

See the [computer-vision implementation plan](docs/computer-vision-plan.md) and [dataset protocol](data/README.md) for the detailed conventions, data design, labels, and evaluation approach.

## Repository contents

- `hardware/media/`: current breadboard photographs and demonstrations.
- `hardware/history/`: the original physics-class prototype.
- `ltspice/`: the editable schematic, simulation results, and waveform images.
- `docs/` and `data/`: the computer-vision implementation plan and dataset protocol.

Generated LTspice `.raw`, `.db`, `.log`, and operating-point files are excluded because they are not needed to inspect or rerun the schematic. Full-resolution datasets, processed tensors, training runs, and model checkpoints are also excluded from ordinary Git history; reviewed samples and final release artifacts can be published separately.
