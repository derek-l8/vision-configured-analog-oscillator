# Circuit design and experiments

This project began as a physics-class demonstration of a two-transistor astable multivibrator. The potentiometer replaced one timing resistor, while two momentary buttons independently connected additional capacitors. The oscillator drove both a passive buzzer and two alternating LEDs. Holding either button lowered the frequency, and the added capacitance slowed the frequency enough for the LEDs’ alternating blinking to become directly visible.

[![Original physics-class prototype](../hardware/history/original-prototype-web.jpg)](../hardware/history/original-prototype.jpg)

| Stage | Main change | Result |
|---|---|---|
| Physics-class prototype | Used a potentiometer in one timing branch, two button-switched capacitors, two alternating LEDs, and a passive buzzer | Demonstrated how resistance and capacitance change audible pitch and visible oscillation rate |
| Rebuilt oscillator | Changed to a shared timing potentiometer, two maintained capacitor switches, and a fixed RC-filtered buzzer path | Produced four persistent timing configurations suitable for simulation and later visual recognition |
| Output-control experiments | Tested raw/filtered signal mixing, a volume-accent path, a button-controlled pitch drop, and several RC-envelope arrangements | Raw/filter mixing and the volume accent did not create a distinct effect; the pitch drop worked but repeated the function of the timing capacitors; early envelope circuits loaded or altered the oscillator instead of independently controlling the output |
| Final v8 | Isolated the filtered signal with Q3, used Q4 and Q5 for envelope-controlled buzzer current, increased the envelope to 300 µF, and added an LED indicator | Kept the oscillator frequency nearly unchanged during the button action while making the envelope directly visible |

The current build keeps the original project’s oscillator, passive buzzer, and adjustable timing concept, but separates oscillator timing from the momentary output effect.

## Interpreting the simulation

The LTspice circuit uses generic NPN models and a 16 Ω resistive substitute for
the passive buzzer. It predicts circuit voltages and currents, rather than exact
loudness or acoustic response. The LED makes the envelope easier to see than
the buzzer makes it audible.

The reference run uses `Rp = 5.1 kΩ`, with both added timing capacitors represented
by `1 pF` for open switches. A 12-second transient includes a button hold from
6–7 seconds. Physical frequency, duty cycle, and node-voltage measurements are
still needed; component tolerances and wiring parasitics have not been characterized.

See the [schematic and reference results](../README.md#simulation-results) and
[verification instructions](verification.md#ltspice-reference-run).
