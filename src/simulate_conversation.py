import argparse
import json
from pathlib import Path
from typing import Any

from llm import call_llm


ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "prompts"
DATASET_PATH = ROOT / "data" / "Patient_Psi_CM_Dataset.json"
OUTPUTS_DIR = ROOT / "outputs"

CCD_FIELDS = [
    "situation",
    "automatic_thoughts",
    "emotions",
    "behavior",
    "intermediate_beliefs",
    "coping_strategies",
    "core_beliefs",
    "relevant_history",
]

FIELD_LABELS = {
    "situation": "Situation",
    "automatic_thoughts": "Automatic Thoughts",
    "emotions": "Emotions",
    "behavior": "Behavior",
    "intermediate_beliefs": "Intermediate Beliefs",
    "coping_strategies": "Coping Strategies",
    "core_beliefs": "Core Beliefs",
    "relevant_history": "Relevant History",
}


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_case(case_id: str) -> dict[str, Any]:
    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    for case in cases:
        if case.get("id") == case_id:
            return case
    available = ", ".join(case.get("id", "<missing>") for case in cases[:20])
    raise ValueError(f"Case id {case_id!r} was not found. First available ids: {available}")


def join_values(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item) for item in value if str(item).strip())
    return str(value).strip()


def build_core_beliefs(case: dict[str, Any]) -> str:
    sections = [
        ("Helpless", case.get("helpless_belief")),
        ("Unlovable", case.get("unlovable_belief")),
        ("Worthless", case.get("worthless_belief")),
    ]
    parts = []
    for label, value in sections:
        text = join_values(value)
        if text:
            parts.append(f"{label}: {text}")
    return "\n".join(parts)


def case_to_ccd(case: dict[str, Any]) -> dict[str, str]:
    return {
        "situation": join_values(case.get("situation")),
        "automatic_thoughts": join_values(case.get("auto_thought")),
        "emotions": join_values(case.get("emotion")),
        "behavior": join_values(case.get("behavior")),
        "intermediate_beliefs": join_values(case.get("intermediate_belief")),
        "coping_strategies": join_values(case.get("coping_strategies")),
        "core_beliefs": build_core_beliefs(case),
        "relevant_history": join_values(case.get("history")),
    }


def visible_ccd(ccd: dict[str, str], revealed: set[str]) -> dict[str, str]:
    return {
        field: ccd[field] if field in revealed and ccd[field] else "unknown"
        for field in CCD_FIELDS
    }


def build_client_system(
    template: str,
    case: dict[str, Any],
    ccd: dict[str, str],
    revealed: set[str],
) -> str:
    visible = visible_ccd(ccd, revealed)
    return template.format(
        name=case["name"],
        traits=", ".join(case.get("type", [])),
        situation=visible["situation"],
        automatic_thoughts=visible["automatic_thoughts"],
        emotions=visible["emotions"],
        behavior=visible["behavior"],
        intermediate_beliefs=visible["intermediate_beliefs"],
        coping_strategies=visible["coping_strategies"],
        core_beliefs=visible["core_beliefs"],
        relevant_history=visible["relevant_history"],
    )


def format_dialogue(dialogue: list[dict[str, str]]) -> str:
    if not dialogue:
        return "(No dialogue yet.)"
    return "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in dialogue)


def format_recent_context(dialogue: list[dict[str, str]], max_turns: int = 8) -> str:
    recent = dialogue[-max_turns:]
    return format_dialogue(recent)


def parse_yes_no(text: str) -> str:
    first_line = text.strip().splitlines()[0].strip().upper() if text.strip() else ""
    if "YES" in first_line:
        return "YES"
    if "NO" in first_line:
        return "NO"
    upper = text.upper()
    if "ANSWER: YES" in upper:
        return "YES"
    return "NO"


def reveal_for_therapist_utterance(
    reveal_template: str,
    ccd: dict[str, str],
    revealed: set[str],
    dialogue_before_therapist: list[dict[str, str]],
    therapist_utterance: str,
    model: str,
    temperature: float,
) -> list[dict[str, str]]:
    evaluations = []
    recent_context = format_recent_context(dialogue_before_therapist)

    for field in CCD_FIELDS:
        if field in revealed:
            continue

        target = f"{FIELD_LABELS[field]}: {ccd[field] or 'unknown'}"
        prompt = reveal_template.format(
            ccd_element=target,
            last_client_utterance=recent_context,
            last_therapist_utterance=therapist_utterance,
        )
        raw = call_llm(
            system_prompt=prompt,
            user_prompt="Evaluate the therapist utterance for this target CCD element.",
            temperature=temperature,
            model=model,
        )
        evaluations.append(
            {
                "field": field,
                "label": FIELD_LABELS[field],
                "decision": parse_yes_no(raw),
                "raw_output": raw,
            }
        )

    return evaluations


def format_field_list(fields: list[str]) -> str:
    if not fields:
        return "none"
    return ", ".join(FIELD_LABELS[field] for field in fields)


def print_turn_progress(
    turn_index: int,
    therapist_utterance: str,
    client_utterance: str,
    revealed: set[str],
    reveal_evaluations: list[dict[str, str]],
) -> None:
    evaluated_fields = [evaluation["field"] for evaluation in reveal_evaluations]

    print()
    print(f"Turn {turn_index}")
    print(f"Therapist: {therapist_utterance}")
    print(f"Client: {client_utterance}")
    print()
    print(f"Already revealed: {format_field_list(sorted(revealed))}")
    print(f"Reveal Critic on: {format_field_list(evaluated_fields)}")
    print()


def simulate(
    case_id: str,
    turns: int,
    output_path: Path,
    therapist_model: str,
    client_model: str,
    reveal_model: str,
    therapist_temperature: float,
    client_temperature: float,
    reveal_temperature: float,
) -> dict[str, Any]:
    case = load_case(case_id)
    ccd = case_to_ccd(case)

    client_system_template = load_text(PROMPTS_DIR / "client_system.txt")
    client_user_template = load_text(PROMPTS_DIR / "client_user.txt")
    therapist_system = load_text(PROMPTS_DIR / "therapist_system.txt")
    therapist_user_template = load_text(PROMPTS_DIR / "therapist_user.txt")
    reveal_template = load_text(PROMPTS_DIR / "reveal_critic.txt")

    dialogue: list[dict[str, str]] = []
    transcript_turns: list[dict[str, Any]] = []
    revealed: set[str] = set()

    for turn_index in range(1, turns + 1):
        dialogue_before_therapist = list(dialogue)
        therapist_user = therapist_user_template.format(dialogue_history=format_dialogue(dialogue))
        therapist_utterance = call_llm(
            system_prompt=therapist_system,
            user_prompt=therapist_user,
            temperature=therapist_temperature,
            model=therapist_model,
        ).strip()

        reveal_evaluations = reveal_for_therapist_utterance(
            reveal_template=reveal_template,
            ccd=ccd,
            revealed=revealed,
            dialogue_before_therapist=dialogue_before_therapist,
            therapist_utterance=therapist_utterance,
            model=reveal_model,
            temperature=reveal_temperature,
        )

        newly_revealed = []
        for evaluation in reveal_evaluations:
            field = evaluation["field"]
            if evaluation["decision"] == "YES" and field not in revealed:
                revealed.add(field)
                newly_revealed.append(field)

        dialogue.append({"speaker": "Therapist", "text": therapist_utterance})

        client_system = build_client_system(client_system_template, case, ccd, revealed)
        client_user = client_user_template.format(dialogue_history=format_dialogue(dialogue))
        client_utterance = call_llm(
            system_prompt=client_system,
            user_prompt=client_user,
            temperature=client_temperature,
            model=client_model,
        ).strip()
        dialogue.append({"speaker": "Client", "text": client_utterance})

        print_turn_progress(
            turn_index=turn_index,
            therapist_utterance=therapist_utterance,
            client_utterance=client_utterance,
            revealed=revealed,
            reveal_evaluations=reveal_evaluations,
        )

        transcript_turns.append(
            {
                "turn": turn_index,
                "therapist": therapist_utterance,
                "reveal_evaluations": reveal_evaluations,
                "skipped_reveal_evaluations": sorted(revealed - set(newly_revealed)),
                "newly_revealed": newly_revealed,
                "revealed_after_turn": sorted(revealed),
                "visible_ccd_after_reveal": visible_ccd(ccd, revealed),
                "client": client_utterance,
            }
        )

    result = {
        "case_id": case_id,
        "client_name": case["name"],
        "turns_requested": turns,
        "ccd_fields": CCD_FIELDS,
        "full_ccd": ccd,
        "final_revealed_fields": sorted(revealed),
        "dialogue": dialogue,
        "turns": transcript_turns,
        "models": {
            "therapist": therapist_model,
            "client": client_model,
            "reveal": reveal_model,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a multi-turn therapist/client conversation with per-turn CCD reveal critics."
    )
    parser.add_argument("--case-id", default="1-1", help="Dataset case id to simulate.")
    parser.add_argument("--turns", type=int, default=10, help="Number of therapist/client turns.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path.")
    parser.add_argument("--model", default="gpt-4o-mini", help="Default model for all calls.")
    parser.add_argument("--therapist-model", default=None, help="Override therapist model.")
    parser.add_argument("--client-model", default=None, help="Override client model.")
    parser.add_argument("--reveal-model", default=None, help="Override reveal critic model.")
    parser.add_argument("--therapist-temperature", type=float, default=0.7)
    parser.add_argument("--client-temperature", type=float, default=0.7)
    parser.add_argument("--reveal-temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output
    if output_path is None:
        safe_case_id = args.case_id.replace("-", "_")
        output_path = OUTPUTS_DIR / f"session_{safe_case_id}.json"

    result = simulate(
        case_id=args.case_id,
        turns=args.turns,
        output_path=output_path,
        therapist_model=args.therapist_model or args.model,
        client_model=args.client_model or args.model,
        reveal_model=args.reveal_model or args.model,
        therapist_temperature=args.therapist_temperature,
        client_temperature=args.client_temperature,
        reveal_temperature=args.reveal_temperature,
    )
    print(f"Wrote {output_path}")
    print(f"Final revealed fields: {', '.join(result['final_revealed_fields']) or 'none'}")


if __name__ == "__main__":
    main()
