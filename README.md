# sftlint

[![tests](https://github.com/ngo-binh0108/sftlint/actions/workflows/tests.yml/badge.svg)](https://github.com/ngo-binh0108/sftlint/actions/workflows/tests.yml)

Check a fine-tuning dataset before you spend GPU hours on it. sftlint reads chat datasets in JSONL and finds the
records that quietly make a fine-tune worse:

- lines that are not valid JSON, empty answers, unknown roles, conversations with no assistant turn
- turns out of order: two user turns in a row, a system prompt in the middle, a tool result with no tool call
- chat-template tokens such as `<|im_start|>` or `[INST]` pasted into the text, which the template then adds a second time
- "As an AI language model" boilerplate and refusals in the answers
- exact and near-duplicate records
- eval or test records that also appear in the training data
- records longer than your training context

It needs only the Python standard library. numpy makes duplicate finding about three times faster, and
[`tokenizers`](https://github.com/huggingface/tokenizers) gives exact token counts.

## Install

```sh
pip install git+https://github.com/ngo-binh0108/sftlint
pip install "sftlint[fast,tokenizer] @ git+https://github.com/ngo-binh0108/sftlint"   # with numpy and tokenizers
```

## Usage

```sh
sftlint train.jsonl                                  # check one file
sftlint train.jsonl --eval test.jsonl                # also look for test records in the training data
sftlint train.jsonl --max-tokens 4096 --tokenizer Qwen/Qwen3-8B
sftlint train.jsonl --write-clean train.clean.jsonl  # keep only records without errors or repeats
sftlint train.jsonl --json                           # machine-readable report
```

Running it on the [example dataset](examples) in this repository:

```text
$ sftlint train.jsonl --eval eval.jsonl --write-clean clean.jsonl
train.jsonl: 12 records (9 messages, 1 sharegpt, 1 alpaca)
  turns per record: 2-3, mean 2.8   tokens per record: p50 ~33, p95 71, max 71
  system prompt in 9 records (1 distinct)
eval.jsonl (eval): 2 records (2 messages)
  turns per record: 3-3, mean 3.0   tokens per record: p50 ~38, p95 38, max 38
  system prompt in 2 records (1 distinct)

ERROR   bad-json: the line is not valid JSON (1)
  train.jsonl:8  not valid JSON (Expecting ',' delimiter at column 60)

ERROR   empty-turn: a turn has no text (1)
  train.jsonl:7  turn 3 (assistant) is empty

ERROR   unknown-role: a turn has a role chat templates do not know (1)
  train.jsonl:12  turn 2 has role 'chef'

ERROR   no-assistant: there is no assistant turn to learn from (1)
  train.jsonl:9  no assistant turn

ERROR   eval-leak: an eval record also appears in the training data (1)
  eval.jsonl:1  the same as training record train.jsonl:2

WARNING template-token: the text contains chat-template tokens (1)
  train.jsonl:6  turn 3 (assistant) contains <|im_start|>; the chat template adds these itself

WARNING boilerplate: an assistant turn has AI boilerplate or a refusal (1)
  train.jsonl:5  turn 3 says 'As an AI language model'

WARNING duplicate: the record repeats an earlier one (1)
  train.jsonl:3  repeats train.jsonl:1

WARNING near-duplicate: the record is almost the same as an earlier one (1)
  train.jsonl:4  ~0.94 similar to train.jsonl:1

5 errors and 4 warnings in 14 records.
Wrote 6 records to clean.jsonl, leaving out records with errors and repeats.
```

The exit code is 1 when there are errors, so it can stop a training script or a CI job. With `--strict`,
warnings count too.

| Option | Default | Meaning |
| --- | --- | --- |
| `--eval FILE` | none | eval or test set to check for leaks; repeat it for several files |
| `--max-tokens N` | off | warn about records longer than N tokens |
| `--tokenizer NAME` | estimate | count tokens with a Hugging Face tokenizer: a repo id or a `tokenizer.json` path |
| `--similarity X` | `0.8` | how similar two records must be to count as near-duplicates |
| `--skip CODES` | none | checks to turn off, comma-separated, e.g. `boilerplate,near-duplicate` |
| `--write-clean OUT` | off | write the training records without errors, duplicates or near-duplicates to OUT |
| `--strict` | off | exit with 1 on warnings as well |
| `--all` | off | list every issue instead of the first five of each kind |
| `--json` | off | print the report as JSON |

## Formats

Each record is detected on its own, so one file can mix formats.

| Format | Fields |
| --- | --- |
| messages (OpenAI, TRL, Axolotl `chat_template`) | `messages`: a list of `role` and `content`; content can be a string or a list of parts |
| ShareGPT | `conversations`: a list of `from` (`human`, `gpt`, `system`, ...) and `value` |
| Alpaca | `instruction`, optional `input`, `output`, optional `system` |
| prompt-completion | `prompt` and `completion`, as strings or as message lists |
| text | `text`, already formatted with a chat template; only empty and too-long records are checked |

Files ending in `.json` may hold one array of records; anything else is read as JSON Lines.

## Checks

| Code | Level | Finds |
| --- | --- | --- |
| `bad-json`, `bad-encoding`, `not-object` | error | lines that cannot be read as a JSON object |
| `unknown-format` | error | records in none of the formats above |
| `bad-turn` | error | turns that are not objects, or whose content is not text |
| `empty-turn` | error | turns with no text |
| `unknown-role` | error | roles other than system, developer, user, assistant, tool, function and ipython |
| `no-assistant` | error | records with nothing for the model to learn |
| `eval-leak` | error | eval records that are the same as, or nearly the same as, a training record |
| `role-order` | warning | a conversation starting with the assistant, two user or assistant turns in a row, a system turn after the start, a tool result with no assistant turn before it |
| `ends-with-user` | warning | a last turn that is not from the assistant, so nothing after it is learned |
| `template-token` | warning | template tokens from ChatML, Llama 3, Gemma, Phi, Mistral and Llama 2 inside the text |
| `boilerplate` | warning | "As an AI language model", "my knowledge cutoff", stock refusals and similar phrases in assistant turns |
| `too-long` | warning | records over `--max-tokens` |
| `duplicate` | warning | records that repeat an earlier one, ignoring case, spacing and punctuation |
| `near-duplicate` | warning | records at least `--similarity` similar to an earlier one |

## How duplicates are found

The system prompt is left out, since many records share one. The rest of the record is lower-cased and split
into words, with each Chinese, Japanese or Korean character counted as a word. Two records are exact duplicates
when those words are the same.

For near-duplicates, each record becomes a set of 3-word shingles, and similarity is the
[Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index) of two sets. Comparing every pair would be too slow,
so sftlint uses MinHash with 64 hashes and locality-sensitive hashing in 16 bands: each record is only compared
with records that share a band. The reported similarity is an estimate, usually within 0.05 of the true value.
Records under three words are only checked for exact duplicates.

The first record of a group is kept and the later ones are reported. On 20,000 records of 150 words it takes
about 10 seconds with numpy and 30 without it.

## Limits

- Without `--tokenizer`, token counts are an estimate: four characters per token, one per CJK character.
- Token counts cover the text of the turns, not the tokens the chat template adds around them.
- The boilerplate check is a list of English phrases. It finds the common ones, not every refusal.
- `--write-clean` keeps records that only have warnings other than duplicates. Use `--skip` and `--strict` to
  decide which warnings matter for your dataset.
- Preference datasets (`chosen` and `rejected`) are not supported yet.

## Development

```sh
pip install -e ".[test,fast]" ruff
ruff check src tests
pytest -q
```

## License

MIT. See [LICENSE](LICENSE).
