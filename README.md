# LLM Attribution Bias — Quick Test

Does telling a judge model *who wrote* a response change how it scores that response, independent of actual quality?

This is a self-contained OpenAI-only experiment testing **attribution bias** in LLM-as-judge evaluation pipelines.

→ [Experiment Design](experiment_design.html)

---

## Models

Four OpenAI models with a defined prestige gradient:

| Model | Prestige |
|---|---|
| GPT-3.5 Turbo | 1 |
| GPT-4o-mini | 2 |
| GPT-4 Turbo | 3 |
| GPT-4o | 4 |

Same models act as both response generators and judges. Self-judging excluded.

---

## Prompts

6 prompts, 2 per task type:

| ID | Type | Topic |
|---|---|---|
| A1 | Analytical | Snail wall climbing problem |
| A2 | Analytical | Pirate gold coin game theory |
| C1 | Creative | Programmer breakup letter |
| C2 | Creative | Traffic jam — three perspectives |
| F1 | Factual | Bayes' theorem / rare disease test |
| F2 | Factual | Antibiotic resistance |

---

## Design

Each response is judged under 4 attribution conditions:

| Condition | What the judge is told |
|---|---|
| Blind | Nothing |
| True | Actual model that wrote the response |
| Upward | Nearest-prestige model above the true author (prestige + 1) |
| Downward | Nearest-prestige model below the true author (prestige − 1) |

Primary metric: **attribution delta** = `score(condition) − score(blind)`

- 4 models × 6 prompts × 2 reps = **48 responses**
- 3 judges per response × 4 conditions = **~480 judgments**

---

## Usage

```bash
cd quick_test
export OPENAI_API_KEY=your_key_here
python quick_test.py

# dry run (no API calls)
python quick_test.py --dry-run
```

---

## Output files

| File | Contents |
|---|---|
| `data/test_responses.json` | Generated responses |
| `data/test_judgments.json` | Scores + rationales |
| `plots/` | Delta charts by condition, task type, judge model |