import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_ROOT / "prompts"
DATASET_PATH = PROJECT_ROOT / "data" / "Patient_Psi_CM_Dataset.json"
DIFFICULTY_STATE_BY_DIFFICULTY = {
    "easy": {
        "openness": "high",
        "metacognition": "high",
    },
    "normal": {
        "openness": "medium",
        "metacognition": "medium",
    },
    "hard": {
        "openness": "low",
        "metacognition": "low",
    },
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_cases(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_case(cases: list[dict[str, Any]], case_id: str | None, case_index: int) -> dict[str, Any]:
    if case_id:
        for case in cases:
            if str(case.get("id")) == case_id:
                return case
        raise ValueError(f"No case found with id {case_id!r}.")

    if case_index < 0 or case_index >= len(cases):
        raise ValueError(f"case_index must be between 0 and {len(cases) - 1}.")
    return cases[case_index]


def join_list(values: Any) -> str:
    if isinstance(values, list):
        return "; ".join(str(value) for value in values if value)
    return str(values or "")


def format_client_system_prompt(
    template: str,
    case: dict[str, Any],
    openness: str,
    metacognition: str,
) -> str:
    return template.format(
        name=case.get("name", ""),
        type=join_list(case.get("type")),
        situation=case.get("situation", ""),
        auto_thought=case.get("auto_thought", ""),
        emotion=join_list(case.get("emotion")),
        behavior=case.get("behavior", ""),
        history=case.get("history", ""),
        intermediate_belief=case.get("intermediate_belief", ""),
        helpless_belief=join_list(case.get("helpless_belief")) or "Not specified",
        unlovable_belief=join_list(case.get("unlovable_belief")) or "Not specified",
        worthless_belief=join_list(case.get("worthless_belief")) or "Not specified",
        coping_strategies=case.get("coping_strategies", ""),
        openness=openness,
        metacognition=metacognition,
    )


def render_dialogue_history(turns: list[dict[str, Any]]) -> str:
    if not turns:
        return "(No previous dialogue.)"
    return "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in turns)


def generate_next_reply(
    system_prompt: str,
    user_template: str,
    dialogue: list[dict[str, Any]],
    model: str,
    temperature: float,
) -> str:
    from llm import call_llm_messages

    dialogue_history = render_dialogue_history(dialogue)
    user_prompt = user_template.format(dialogue_history=dialogue_history)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    reply = call_llm_messages(messages, temperature=temperature, model=model)
    return reply.strip()


def simulate_conversation(
    case: dict[str, Any],
    turn_pairs: int,
    conversation_model: str,
    temperature: float,
    difficulty: str,
) -> list[dict[str, Any]]:
    client_system_template = read_text(PROMPTS_DIR / "client_system.txt")
    client_user_template = read_text(PROMPTS_DIR / "client_user.txt")
    therapist_system_prompt = read_text(PROMPTS_DIR / "therapist_system.txt")
    therapist_user_template = read_text(PROMPTS_DIR / "therapist_user.txt")
    client_state = DIFFICULTY_STATE_BY_DIFFICULTY[difficulty]
    openness = client_state["openness"]
    metacognition = client_state["metacognition"]

    dialogue: list[dict[str, Any]] = []

    for turn_number in range(1, turn_pairs + 1):
        print(f"\nTurn {turn_number}", flush=True)

        therapist_reply = generate_next_reply(
            system_prompt=therapist_system_prompt,
            user_template=therapist_user_template,
            dialogue=dialogue,
            model=conversation_model,
            temperature=temperature,
        )
        dialogue.append(
            {
                "turn": turn_number,
                "speaker": "Therapist",
                "text": therapist_reply,
            }
        )
        print(f"\nTherapist: {therapist_reply}", flush=True)

        print(
            "\n"
            f"Client state for turn {turn_number}: "
            f"openness={openness}, metacognition={metacognition}",
            flush=True,
        )
        client_system_prompt = format_client_system_prompt(
            client_system_template,
            case,
            openness,
            metacognition,
        )

        client_reply = generate_next_reply(
            system_prompt=client_system_prompt,
            user_template=client_user_template,
            dialogue=dialogue,
            model=conversation_model,
            temperature=temperature,
        )
        client_turn = {
            "turn": turn_number,
            "speaker": "Client",
            "text": client_reply,
            "openness": openness,
            "metacognition": metacognition,
        }
        dialogue.append(client_turn)
        print(f"\nClient: {client_reply}", flush=True)

    return dialogue


def write_output(path: Path, case: dict[str, Any], dialogue: list[dict[str, Any]], args: argparse.Namespace) -> None:
    payload = {
        "case_id": case.get("id"),
        "case_name": case.get("name"),
        "conversation_model": args.conversation_model,
        "temperature": args.temperature,
        "turn_pairs": args.turns,
        "total_messages": args.turns * 2,
        "difficulty": args.difficulty,
        "openness": DIFFICULTY_STATE_BY_DIFFICULTY[args.difficulty]["openness"],
        "metacognition": DIFFICULTY_STATE_BY_DIFFICULTY[args.difficulty]["metacognition"],
        "dialogue": dialogue,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a multi-turn conversation between the therapist and client agents."
    )
    parser.add_argument("--case-id", help="Patient case id from the dataset, e.g. 1-1.")
    parser.add_argument(
        "--case-index",
        type=int,
        default=0,
        help="Zero-based patient case index to use when --case-id is not provided.",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=12,
        help="Number of therapist-client turn pairs to generate. Each turn contains one therapist message and one client message.",
    )
    parser.add_argument(
        "--conversation-model",
        default="gpt-4o-mini",
        help="OpenAI model for therapist and client generation.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--difficulty",
        choices=list(DIFFICULTY_STATE_BY_DIFFICULTY.keys()),
        default="normal",
        help="Client difficulty. easy=high openness/metacognition, normal=medium, hard=low.",
    )
    parser.add_argument("--output", type=Path, help="Optional path to save transcript JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.turns < 1:
        raise ValueError("--turns must be at least 1.")

    cases = load_cases()
    case = get_case(cases, args.case_id, args.case_index)
    print(
        f"Simulating case {case.get('id')} ({case.get('name')}) "
        f"with {args.turns} therapist-client turns."
    )

    dialogue = simulate_conversation(
        case=case,
        turn_pairs=args.turns,
        conversation_model=args.conversation_model,
        temperature=args.temperature,
        difficulty=args.difficulty,
    )

    if args.output:
        write_output(args.output, case, dialogue, args)
        print(f"\nSaved transcript to {args.output}")


if __name__ == "__main__":
    main()
