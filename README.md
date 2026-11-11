# OmegaMind — From Question to Answer

OmegaMind is an educational walkthrough of how a text request moves through a small language-model inference pipeline: from the question entered in the website, to token IDs, CPU-side model processing, next-token selection, and the final readable answer.

## Animated walkthrough

![OmegaMind animated walkthrough](assets/omegamind-walkthrough.gif)

> **Note:** This is a guided concept demonstration, not a live model execution. The displayed token IDs, next-token scores, timings, CPU activity, and sample answer are illustrative. They should not be read as measured benchmark results.

## What the walkthrough explains

1. **Enter a question** — the website prepares the user's message.
2. **Tokenize the text** — the tokenizer breaks text into pieces and maps each piece to an ID.
3. **Send token IDs** — the service passes the numbered request to the C++ inference program.
4. **Process the input** — the model looks up learned values and processes them through its layers.
5. **Use the CPU and RAM** — the visualization illustrates CPU work, parallel tasks, and memory use.
6. **Choose the next token** — the model scores candidate pieces and selects one.
7. **Reuse the KV cache** — attention information from earlier pieces is retained for subsequent generation steps.
8. **Return readable text** — token IDs are decoded into text and displayed in the website.

## Run the interactive HTML demo locally

The GIF above is for the GitHub README preview. To use the original Play/Pause control, playback-speed selector, progress slider, question input, and theme switch, open the HTML demo locally:

```bash
python -m http.server 8000
```

Then open [http://localhost:8000/demo/](http://localhost:8000/demo/) in your browser from the repository root.

The demo is self-contained apart from its optional Google Fonts import. It does not start or query a live model process; its moving metrics and token-selection visuals are illustrative.

## Repository files

- `demo/index.html` — original interactive HTML walkthrough.
- `assets/omegamind-walkthrough.gif` — animated preview embedded above.
