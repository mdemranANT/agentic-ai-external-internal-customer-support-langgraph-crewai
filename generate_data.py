"""
Synthetic FAQ generator.
Uses GPT-4o to produce 20-25 QA pairs per department and saves them as JSON.

Usage:  python generate_data.py
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from config import DEPARTMENTS, DATA_DIR

load_dotenv()

DATA_PATH = Path(DATA_DIR)
DATA_PATH.mkdir(exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_qa_for_department(dept_key: str, num_pairs: int = 25) -> dict:
    """Calls GPT-4o to create realistic FAQ pairs for one department."""
    dept = DEPARTMENTS[dept_key]

    audience_label = "External customers" if dept['audience'] == 'external' else "Internal employees"

    prompt = f"""You are a data-generation assistant for Shop4You, a UK-based online retail company.

Generate exactly {num_pairs} realistic frequently-asked-question (FAQ) pairs for the **{dept['name']}** department.

Department details:
- Audience: {audience_label}
- Scope: {dept['description']}
- Tone: {dept['tone']}
- Example topics: {', '.join(dept['example_topics'])}

Requirements:
1. Each QA pair must have: "question", "answer", "tags" (list of 2-4 keyword tags).
2. Answers should be 3-6 sentences, helpful, detailed, and realistic for a UK retail company.
3. Mix of question types:
   - Simple factual ("What is...?", "How do I...?")
   - Scenario-based ("I ordered X but received Y, what do I do?")
   - Policy clarification ("What happens if...?", "Am I eligible for...?")
   - Process/procedural ("What are the steps to...?", "Who do I contact for...?")
   - Edge cases and follow-ups ("What if I miss the deadline?", "Can I change my request after submission?")
4. Use British English spelling (e.g., "colour", "programme", "organisation").
5. Reference Shop4You by name where appropriate.
6. Make sure all {num_pairs} questions are distinct  --  no duplicate or near-duplicate questions.
7. Cover the full breadth of the department's scope, not just the obvious topics.

Return ONLY a valid JSON array of objects, no markdown fences, no extra text:
[
  {{"question": "...", "answer": "...", "tags": ["tag1", "tag2"]}},
  ...
]"""

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    entries = json.loads(raw)

    now = datetime.now(timezone.utc).isoformat()
    for entry in entries:
        entry["last_updated"] = now
        entry["confidence_score"] = 0.95

    return {
        "department": dept["name"],
        "department_key": dept_key,
        "audience": dept["audience"],
        "doc_id": f"{dept_key}_faq_v1",
        "generated_date": now,
        "num_entries": len(entries),
        "entries": entries,
    }


def save_department_data(doc: dict, dept_key: str) -> Path:
    """Writes a department QA doc to disk as JSON."""
    filepath = DATA_PATH / f"{dept_key}_faq.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    return filepath


def generate_all(num_pairs: int = 25) -> dict:
    """Loops through every department, generates QA data, and saves the files."""
    results = {}
    for dept_key in DEPARTMENTS:
        print(f"  Generating {num_pairs} QA pairs for [{dept_key}]...", end=" ", flush=True)
        try:
            doc = generate_qa_for_department(dept_key, num_pairs)
            filepath = save_department_data(doc, dept_key)
            results[dept_key] = {"status": "ok", "file": str(filepath), "count": len(doc["entries"])}
            print(f"[PASS] ({len(doc['entries'])} pairs -> {filepath.name})")
        except Exception as e:
            results[dept_key] = {"status": "error", "error": str(e)}
            print(f"[FAIL] Error: {e}")
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Shop4You  --  Synthetic QA Data Generation")
    print("=" * 60)
    results = generate_all(num_pairs=25)
    print("\n" + "=" * 60)
    ok = sum(1 for r in results.values() if r["status"] == "ok")
    print(f"Done: {ok}/{len(results)} departments generated successfully.")
