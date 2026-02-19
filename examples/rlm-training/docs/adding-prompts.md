# Adding a New System Prompt

This walkthrough shows how to create and test a new system prompt variant for the training pipeline.

## What you need to change

| File | Change | Required? |
|------|--------|-----------|
| `prompts.py` | Add your new prompt function | Yes |
| `config.py` | No changes needed (just assign at runtime) | No |
| `train.py` / `debug_rollout.py` | Set `config.system_prompt_fn = your_function` | Yes |

## Step 1: Write a prompt function in `prompts.py`

Every prompt function has the same signature:

```python
def my_prompt() -> str:
```

It returns a plain string that becomes the system message in the conversation. The prompt should describe the available tools to the model, since the REPL namespace defines what's actually callable.

### Available tools (defined in `repl.py`)

The model's code runs in a persistent Python REPL with these globals:

| Tool | Signature | Description |
|------|-----------|-------------|
| `ls` | `ls(path=".", max_depth=2) -> str` | List files |
| `read` | `read(path, start=None, end=None) -> str` | Read file or line range |
| `grep` | `grep(pattern, path=".", file_pattern="*.py") -> str` | Search for pattern |
| `apply_patch` | `apply_patch(patch: str) -> str` | Apply unified diff |
| `bash` | `bash(cmd, timeout=90) -> str` | Run shell command |
| `run_tests` | `run_tests(timeout=300) -> str` | Run test suite |
| `finish` | `finish() -> str` | Signal completion |
| `llm_query` | `llm_query(prompt) -> str` | Sub-agent (multi-step REPL) |
| `llm_query_batched` | `llm_query_batched(prompts) -> list[str]` | Parallel sub-agents |

Your prompt should mention whichever tools you want the model to use. If you omit a tool from the prompt, the model likely won't use it (but it's still callable).

### Example: Chain-of-thought prompt

```python
def cot_system_prompt() -> str:
    """Prompt that explicitly asks for chain-of-thought reasoning."""
    return """\
You are an expert software engineer. Fix the bug in the repository.

Before writing any code, think through your approach step by step:
1. What does the bug report describe?
2. Where in the codebase is the bug likely located?
3. What is the root cause?
4. What is the minimal fix?

Available tools (persistent Python REPL):
- ls(path=".", max_depth=2) -> str
- read(path, start=None, end=None) -> str
- grep(pattern, path=".", file_pattern="*.py") -> str
- apply_patch(patch: str) -> str
- bash(cmd, timeout=90) -> str
- llm_query(prompt) -> str
- llm_query_batched(prompts) -> list[str]
- run_tests(timeout=300) -> str
- finish() -> str

Write Python code in ```python or ```repl blocks. Always call finish() when done.

IMPORTANT: Before each code block, explain your reasoning in plain text."""
```

## Step 2: Also update the sub-agent prompt (optional)

If your prompt changes how sub-agents should behave, also create a matching `sub_agent_system_prompt_fn`:

```python
def cot_sub_agent_prompt() -> str:
    return """\
You are helping analyze code to answer a question.
Think step by step before writing code.

Tools: ls(), read(), grep(), bash(), llm_query()
Write code in ```python blocks. Answer in plain text when done."""
```

Then set both in config:

```python
config.system_prompt_fn = cot_system_prompt
config.sub_agent_system_prompt_fn = cot_sub_agent_prompt
```

## Step 3: Test with debug_rollout.py

```bash
# Quick test against one task
python debug_rollout.py \
    --instance_id django__django-16229 \
    --max_steps 5 \
    --system_prompt cot_system_prompt
```

Then inspect the saved `.jsonl` trajectory in the RLM visualizer to see how the model responded to your new prompt.

## Step 4: Run a full experiment

```python
from config import ExperimentConfig
from prompts import cot_system_prompt

config = ExperimentConfig(
    experiment_name="cot-prompt-v1",
    system_prompt_fn=cot_system_prompt,
)
```

## Existing prompt variants

| Function | Description | Use case |
|----------|-------------|----------|
| `r2e_rlm_system_prompt` | Full RLM-style with examples and sub-LLM guidance | Default / production |
| `r2e_minimal_system_prompt` | Just the tool list, no examples | Ablation: does more prompting help? |
| `r2e_original_system_prompt` | The existing R2E-Gym rollout prompt | Baseline comparison |
| `sub_agent_system_prompt` | Default sub-agent prompt | Used by `llm_query` recursive sub-agents |

## Tips for prompt experiments

- **A/B test**: Run two training runs with different `experiment_name` values and compare reward curves in W&B.
- **Inspect trajectories**: Use `debug_rollout.py` + the RLM visualizer to qualitatively check behavior before committing to a full training run.
- **Start minimal**: Begin with `r2e_minimal_system_prompt` and add instructions incrementally to isolate what helps.
- **Sub-LLM emphasis**: The original RLM paper found that encouraging sub-LLM usage via explicit examples significantly improved performance. The `r2e_rlm_system_prompt` includes this.
