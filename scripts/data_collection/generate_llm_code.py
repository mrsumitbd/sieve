"""
scripts/data_collection/generate_llm_code.py

Language-agnostic LLM code generation via Together.ai.
Adapted from the original Python-only generation script to support
Python, Java, JavaScript, and C++ in a single unified interface.

Usage:
    python generate_llm_code.py \\
        --input   data/java_functions.csv \\
        --output  data/generated/deepseek_java_func.csv \\
        --model   deepseek-ai/DeepSeek-V3 \\
        --language Java \\
        --granularity function

    python generate_llm_code.py \\
        --input   data/js_classes.csv \\
        --output  data/generated/qwen_js_class.csv \\
        --model   Qwen/Qwen2.5-Coder-7B-Instruct \\
        --language JavaScript \\
        --granularity class

Input CSV must have:
    - Function level: a 'func_signature' column
    - Class level:    a 'class_skeleton' column

Output CSV: input columns + 'generated_code' column.
Run clean_generated_code.py afterwards to produce 'generated_code_cleaned'.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Prompt templates ────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "Python":     "You are an expert Python programmer. Implement complete, correct Python code.",
    "Java":       "You are an expert Java programmer. Implement complete, correct Java code.",
    "JavaScript": "You are an expert JavaScript programmer. Implement complete, correct JavaScript code.",
    "C++":        "You are an expert C++ programmer. Implement complete, correct C++ code.",
}

USER_PROMPT_TEMPLATES = {
    "function": (
        "Implement the following {language} function. "
        "Return only the code with no explanation or markdown fences. "
        "The function signature is:\n{skeleton}"
    ),
    "class": (
        "Implement the following {language} class. "
        "Return only the code with no explanation or markdown fences. "
        "The class skeleton is:\n{skeleton}"
    ),
}

INPUT_COLUMNS = {
    "function": "func_signature",
    "class":    "class_skeleton",
}


# ─── Core generation ─────────────────────────────────────────────────────────

def _make_client():
    import together
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "TOGETHER_API_KEY not found. Set it in your .env file or environment."
        )
    return together.Together(api_key=api_key)


def generate_batch(
    skeletons: list[str],
    model: str,
    language: str,
    granularity: str,
    temperature: float = 0.0,
) -> list[str | None]:
    """
    Generate code for a list of skeletons using Together.ai batch API.
    Returns a list of generated strings (None for failed requests).
    Preferred for large datasets (>500 items) — lower cost, async.
    """
    import together

    client = _make_client()
    system_prompt = SYSTEM_PROMPTS[language]
    user_tmpl = USER_PROMPT_TEMPLATES[granularity]

    batch_file = "batch_requests_tmp.jsonl"
    output_file = "batch_outputs_tmp.jsonl"

    # Write JSONL request file
    with open(batch_file, "w", encoding="utf-8") as f:
        for i, skeleton in enumerate(skeletons):
            user_prompt = user_tmpl.format(language=language, skeleton=skeleton)
            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                "temperature": temperature,
            }
            f.write(json.dumps({"custom_id": f"req-{i}", "body": body}) + "\n")
    logger.info(f"Wrote {len(skeletons)} requests to {batch_file}")

    # Upload
    file_resp = client.files.upload(file=batch_file, purpose="batch-api")
    file_id = file_resp.id
    logger.info(f"Uploaded batch file: {file_id}")

    # Create batch
    batch = client.batches.create_batch(
        file_id=file_id,
        endpoint="/v1/chat/completions",
    )
    batch_id = batch.id
    logger.info(f"Created batch job: {batch_id}")

    # Poll
    while True:
        status = client.batches.get_batch(batch_id)
        logger.info(f"Batch status: {status.status}")
        if status.status == "COMPLETED":
            output_file_id = status.output_file_id
            break
        elif status.status in ("FAILED", "EXPIRED", "CANCELLED"):
            raise RuntimeError(f"Batch job failed: {status.status}")
        time.sleep(30)

    # Retrieve output
    client.files.retrieve_content(id=output_file_id, output=output_file)

    # Parse results
    results: dict[str, str] = {}
    with open(output_file, "r") as f:
        for line in f:
            data = json.loads(line)
            if data["response"]["status_code"] == 200:
                content = data["response"]["body"]["choices"][0]["message"]["content"]
                results[data["custom_id"]] = content
            else:
                logger.warning(f"Request {data['custom_id']} failed: {data['response']['status_code']}")

    # Cleanup
    for tmp in (batch_file, output_file):
        try:
            os.remove(tmp)
        except OSError:
            pass

    return [results.get(f"req-{i}") for i in range(len(skeletons))]


def generate_sequential(
    skeletons: list[str],
    model: str,
    language: str,
    granularity: str,
    temperature: float = 0.0,
) -> list[str | None]:
    """
    Generate code one request at a time via Together.ai chat completions.
    Preferred for small datasets (<500 items) or debugging.
    """
    client = _make_client()
    system_prompt = SYSTEM_PROMPTS[language]
    user_tmpl = USER_PROMPT_TEMPLATES[granularity]

    results = []
    for i, skeleton in enumerate(skeletons):
        user_prompt = user_tmpl.format(language=language, skeleton=skeleton)
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            code = response.choices[0].message.content
            results.append(code)
            logger.info(f"Generated {i + 1}/{len(skeletons)}")
        except Exception as e:
            logger.error(f"Failed at index {i}: {e}")
            results.append(None)

    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM code via Together.ai for Python/Java/JavaScript/C++"
    )
    parser.add_argument("--input",       required=True, help="Input CSV file")
    parser.add_argument("--output",      required=True, help="Output CSV file")
    parser.add_argument("--model",       required=True, help="Together.ai model ID")
    parser.add_argument("--language",    required=True,
                        choices=["Python", "Java", "JavaScript", "C++"],
                        help="Programming language")
    parser.add_argument("--granularity", required=True,
                        choices=["function", "class"],
                        help="Granularity level")
    parser.add_argument("--mode",        default="auto",
                        choices=["auto", "batch", "sequential"],
                        help="Generation mode (auto: batch if >500 rows, else sequential)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default 0.0 for reproducibility)")
    parser.add_argument("--start",       type=int, default=0,
                        help="Start row index (inclusive)")
    parser.add_argument("--end",         type=int, default=None,
                        help="End row index (exclusive). Default: all rows.")
    args = parser.parse_args()

    # Load input
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows from {args.input}")

    # Slice
    df = df.iloc[args.start:args.end].copy()
    logger.info(f"Processing rows {args.start} to {args.start + len(df) - 1} ({len(df)} rows)")

    # Get input column
    input_col = INPUT_COLUMNS[args.granularity]
    if input_col not in df.columns:
        raise ValueError(
            f"Input column '{input_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    df.dropna(subset=[input_col], inplace=True)
    skeletons = df[input_col].tolist()

    # Choose mode
    mode = args.mode
    if mode == "auto":
        mode = "batch" if len(skeletons) > 500 else "sequential"
    logger.info(f"Using {mode} mode for {len(skeletons)} items with {args.model}")

    # Generate
    if mode == "batch":
        generated = generate_batch(
            skeletons, args.model, args.language, args.granularity, args.temperature
        )
    else:
        generated = generate_sequential(
            skeletons, args.model, args.language, args.granularity, args.temperature
        )

    df["generated_code"] = generated

    # Clean up index column name if present
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    n_success = sum(1 for g in generated if g is not None)
    logger.info(f"Saved {len(df)} rows ({n_success} successful) to {args.output}")


if __name__ == "__main__":
    main()