import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "Patient_Psi_CM_Dataset.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
PROMPT_DIR = ROOT / "prompts"
UNKNOWN = "Unknown"


CCD_ELEMENTS = [
    {
        "id": "situation",
        "label": "Situation",
        "fields": ["situation"],
    },
    {
        "id": "auto_thought",
        "label": "Automatic Thoughts",
        "fields": ["auto_thought"],
    },
    {
        "id": "emotion",
        "label": "Emotions",
        "fields": ["emotion"],
    },
    {
        "id": "behavior",
        "label": "Behavior",
        "fields": ["behavior"],
    },
    {
        "id": "history",
        "label": "Relevant History",
        "fields": ["history"],
    },
    {
        "id": "intermediate_belief",
        "label": "Intermediate Beliefs",
        "fields": ["intermediate_belief"],
    },
    {
        "id": "core_beliefs",
        "label": "Core Beliefs",
        "fields": ["helpless_belief", "unlovable_belief", "worthless_belief"],
    },
    {
        "id": "coping_strategies",
        "label": "Coping Strategies",
        "fields": ["coping_strategies"],
    },
]


class SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_prompts(prompt_dir: Path) -> dict[str, str]:
    return {
        "client_system": read_text(prompt_dir / "client_system.txt"),
        "client_user": read_text(prompt_dir / "client_user.txt"),
        "therapist_system": read_text(prompt_dir / "therapist_system.txt"),
        "therapist_user": read_text(prompt_dir / "therapist_user.txt"),
        "reveal_critic": read_text(prompt_dir / "reveal-critic.txt"),
    }


def load_cases(dataset_path: Path) -> list[dict[str, Any]]:
    with dataset_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of cases in {dataset_path}")
    return data


def find_case(cases: list[dict[str, Any]], case_id: str | None) -> dict[str, Any]:
    if case_id is None:
        return cases[0]
    for case in cases:
        if str(case.get("id")) == case_id:
            return case
    raise ValueError(f"No case found with id={case_id!r}")


def format_value(value: Any) -> str:
    if value in (None, "", []):
        return UNKNOWN
    if isinstance(value, list):
        return "; ".join(str(item) for item in value) if value else UNKNOWN
    return str(value)


def render(template: str, values: dict[str, Any]) -> str:
    formatted = {key: format_value(value) for key, value in values.items()}
    return template.format_map(SafeDict(formatted))


def initial_revealed(reveal_surface: bool) -> dict[str, bool]:
    revealed = {element["id"]: False for element in CCD_ELEMENTS}
    if reveal_surface:
        for element_id in ["situation", "auto_thought", "emotion", "behavior"]:
            revealed[element_id] = True
    return revealed


def build_client_case(case: dict[str, Any], revealed: dict[str, bool]) -> dict[str, Any]:
    client_case = dict(case)
    for element in CCD_ELEMENTS:
        if not revealed[element["id"]]:
            for field in element["fields"]:
                client_case[field] = UNKNOWN
    return client_case


def ccd_target_text(case: dict[str, Any], element: dict[str, Any]) -> str:
    lines = []
    for field in element["fields"]:
        lines.append(f"{field}: {format_value(case.get(field))}")
    return f"{element['label']}\n" + "\n".join(lines)


def dialogue_to_text(dialogue: list[dict[str, str]]) -> str:
    if not dialogue:
        return "(No dialogue yet.)"
    return "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in dialogue)


def recent_context(dialogue: list[dict[str, str]], max_turns: int = 6) -> str:
    return dialogue_to_text(dialogue[-max_turns:])


def parse_yes_no(critic_output: str) -> bool:
    match = re.search(r"Answer:\s*(YES|NO)\b", critic_output, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper() == "YES"
    return critic_output.strip().upper().startswith("YES")


def revealed_labels(revealed: dict[str, bool]) -> list[str]:
    return [element["label"] for element in CCD_ELEMENTS if revealed[element["id"]]]


def print_turn_progress(
    turn_index: int,
    therapist_text: str,
    client_text: str,
    evaluations: list[dict[str, Any]],
    revealed: dict[str, bool],
) -> None:
    revealed = revealed_labels(revealed)
    critic_runs = [evaluation["ccd_label"] for evaluation in evaluations]

    print(f"\nTurn {turn_index}", flush=True)
    print(f"Therapist: {therapist_text}", flush=True)
    print(f"\nClient: {client_text}", flush=True)
    print(f"\nrevealed elements: {', '.join(revealed) if revealed else 'None'}", flush=True)
    print(f"Critic: {', '.join(critic_runs) if critic_runs else 'None'}", flush=True)


def next_therapist_utterance(
    prompts: dict[str, str],
    dialogue: list[dict[str, str]],
    model: str,
    temperature: float,
) -> str:
    from llm import call_llm_messages

    messages = [{"role": "system", "content": prompts["therapist_system"]}]
    if not dialogue:
        messages.append(
            {
                "role": "user",
                "content": "Begin the session. Write only the therapist's first utterance.",
            }
        )
        return call_llm_messages(messages, temperature=temperature, model=model).strip()

    messages.append(
        {
            "role": "user",
            "content": render(
                prompts["therapist_user"],
                {"dialogue_history": dialogue_to_text(dialogue)},
            ),
        }
    )
    return call_llm_messages(messages, temperature=temperature, model=model).strip()


def evaluate_reveals(
    prompts: dict[str, str],
    case: dict[str, Any],
    dialogue: list[dict[str, str]],
    therapist_utterance: str,
    revealed: dict[str, bool],
    model: str,
) -> list[dict[str, Any]]:
    from llm import call_llm

    evaluations = []
    last_client_utterance = recent_context(dialogue)

    for element in CCD_ELEMENTS:
        if revealed[element["id"]]:
            continue

        critic_prompt = render(
            prompts["reveal_critic"],
            {
                "ccd_element": ccd_target_text(case, element),
                "last_client_utterance": last_client_utterance,
                "last_therapist_utterance": therapist_utterance,
            },
        )
        output = call_llm(
            system_prompt="You are a strict evaluator. Follow the requested output format exactly.",
            user_prompt=critic_prompt,
            temperature=0.0,
            model=model,
        ).strip()
        answer = parse_yes_no(output)
        if answer:
            revealed[element["id"]] = True

        evaluations.append(
            {
                "ccd_id": element["id"],
                "ccd_label": element["label"],
                "was_revealed_before_turn": False,
                "revealed_by_this_evaluation": answer,
                "answer": "YES" if answer else "NO",
                "raw_output": output,
            }
        )

    return evaluations


def next_client_utterance(
    prompts: dict[str, str],
    case: dict[str, Any],
    revealed: dict[str, bool],
    dialogue: list[dict[str, str]],
    model: str,
    temperature: float,
) -> str:
    from llm import call_llm

    client_case = build_client_case(case, revealed)
    system_prompt = render(prompts["client_system"], client_case)

    user_prompt = render(
        prompts["client_user"],
        {"dialogue_history": dialogue_to_text(dialogue)},
    )
    return call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        model=model,
    ).strip()


def simulate(args: argparse.Namespace) -> dict[str, Any]:
    prompts = load_prompts(args.prompt_dir)
    case = find_case(load_cases(args.dataset), args.case_id)
    revealed = initial_revealed(args.reveal_surface)
    dialogue: list[dict[str, str]] = []
    turn_records = []

    for turn_index in range(1, args.turns + 1):
        therapist_text = next_therapist_utterance(
            prompts=prompts,
            dialogue=dialogue,
            model=args.therapist_model,
            temperature=args.therapist_temperature,
        )
        evaluations = evaluate_reveals(
            prompts=prompts,
            case=case,
            dialogue=dialogue,
            therapist_utterance=therapist_text,
            revealed=revealed,
            model=args.critic_model,
        )
        dialogue.append({"speaker": "Therapist", "text": therapist_text})

        client_text = next_client_utterance(
            prompts=prompts,
            case=case,
            revealed=revealed,
            dialogue=dialogue,
            model=args.client_model,
            temperature=args.client_temperature,
        )
        dialogue.append({"speaker": "Client", "text": client_text})

        print_turn_progress(
            turn_index=turn_index,
            therapist_text=therapist_text,
            client_text=client_text,
            evaluations=evaluations,
            revealed=revealed,
        )

        turn_records.append(
            {
                "turn": turn_index,
                "therapist": therapist_text,
                "client": client_text,
                "reveal_evaluations": evaluations,
                "revealed_after_turn": dict(revealed),
            }
        )

    return {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "case_id": case.get("id"),
            "client_name": case.get("name"),
            "turns": args.turns,
            "client_model": args.client_model,
            "therapist_model": args.therapist_model,
            "critic_model": args.critic_model,
            "reveal_surface": args.reveal_surface,
        },
        "case": case,
        "ccd_elements": CCD_ELEMENTS,
        "final_revealed": revealed,
        "dialogue": dialogue,
        "turns": turn_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a multi-turn CBT client/therapist conversation with per-turn CCD reveal critics."
    )
    parser.add_argument("--case-id", help="Case id from the dataset, e.g. 1-1. Defaults to the first case.")
    parser.add_argument("--turns", type=int, default=10, help="Number of therapist/client turns to simulate.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--prompt-dir", type=Path, default=PROMPT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, help="Output JSON path. Defaults to outputs/session_<case-id>.json.")
    parser.add_argument("--client-model", default="gpt-4o-mini")
    parser.add_argument("--therapist-model", default="gpt-4o-mini")
    parser.add_argument("--critic-model", default="gpt-4o")
    parser.add_argument("--client-temperature", type=float, default=0.7)
    parser.add_argument("--therapist-temperature", type=float, default=0.7)
    parser.add_argument(
        "--reveal-surface",
        action="store_true",
        help="Start with situation, automatic thoughts, emotions, and behavior already visible.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.turns < 1:
        raise ValueError("--turns must be at least 1")

    result = simulate(args)
    case_id = str(result["metadata"]["case_id"]).replace("/", "_")
    output_path = args.output or args.output_dir / f"session_{case_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {output_path}")
    print(f"Final revealed CCDs: {json.dumps(result['final_revealed'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
