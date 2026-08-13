# Result images (per model)

Input and output for every model used in the project, one folder per model, in
project order. Each output shows the model's single highest-confidence box
(red, with its confidence). Output boxes were drawn with the multi-scale WBF
inference pipeline (tiles 320/512/768).

| Folder | Model | Input page | Top-1 confidence | Note |
|---|---|---|---|---|
| 1_yolov8n | waldo_yolov8n | amusement park | 0.03 | earliest baseline, essentially noise |
| 2_waldo_synth_a | waldo_synth_A | ski slope (b09_p013) | 0.50 | synthetic model: fires many boxes (41), low precision |
| 3_waldo_book_a | waldo_book_A | town (b10_p004) | 0.66 | real-tile model, a confident correct-looking box |
| 4_waldo_book_decoy | waldo_book_decoy | sea (b16_p025) | 0.25 | reported model |
| 5_waldo_book_ms | waldo_book_ms | maze (b12_p019) | 0.78 | multi-scale-trained model, confident box |

Each folder contains `input_*.jpg` and `output_*.jpg`.

Reading the progression: the earliest models (yolov8n, synth) either barely
fire or fire everywhere with low precision; the book-tile models (book_a,
book_decoy) put a confident box on a real Waldo-like figure. These are single
illustrative pages, not the held-out evaluation (see METRICS.md for the
measured numbers).
