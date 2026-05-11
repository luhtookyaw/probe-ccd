import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_ROOT / "prompts"
JUDGES_DIR = PROMPTS_DIR / "judges"
DATASET_PATH = PROJECT_ROOT / "data" / "Patient_Psi_CM_Dataset.json"
OPENNESS_GUIDELINE_PATH = PROJECT_ROOT / "data" / "openness_guideline.json"
OPENNESS_CADENCE_BY_DIFFICULTY = {
    "easy": 2,
    "normal": 4,
    "hard": 6,
}
EXPLORATION_CADENCE = 4
EXTERNAL_FIELD_LABELS = {
    "situation": "Situation",
    "behavior": "Behavior",
    "automatic_thoughts": "Automatic Thoughts",
    "emotions": "Emotions",
}
INTERNAL_FIELD_LABELS = {
    "intermediate_beliefs": "Intermediate Beliefs",
    "coping_strategies": "Coping Strategies",
    "core_beliefs": "Core Beliefs",
    "relevant_history": "Relevant History",
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_cases(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_guidelines(path: Path, root_key: str) -> dict[str, dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8"))[root_key]


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


def build_core_beliefs(case: dict[str, Any]) -> str:
    belief_groups = [
        ("Helpless", case.get("helpless_belief")),
        ("Unlovable", case.get("unlovable_belief")),
        ("Worthless", case.get("worthless_belief")),
    ]
    beliefs = [
        f"{label}: {join_list(values)}"
        for label, values in belief_groups
        if join_list(values)
    ]
    return "\n".join(beliefs) if beliefs else "Not specified"


def format_revealed_diagram(fields: list[tuple[str, str]]) -> str:
    visible_fields = [
        f"- {label}: {value}"
        for label, value in fields
        if value
    ]
    return "\n".join(visible_fields) if visible_fields else "- None revealed yet"


def score_to_revealed_count(score: int, total_fields: int) -> int:
    return min(max(score - 1, 0), total_fields)


def build_external_reveal_order(rng: random.Random) -> list[str]:
    remaining_fields = [
        field
        for field in EXTERNAL_FIELD_LABELS
        if field != "situation"
    ]
    rng.shuffle(remaining_fields)
    return ["situation", *remaining_fields]


def get_external_field_value(case: dict[str, Any], field: str) -> str:
    values = {
        "situation": case.get("situation", ""),
        "behavior": case.get("behavior", ""),
        "automatic_thoughts": case.get("auto_thought", ""),
        "emotions": join_list(case.get("emotion")),
    }
    return values[field]


def get_internal_field_value(case: dict[str, Any], field: str) -> str:
    values = {
        "intermediate_beliefs": case.get("intermediate_belief", ""),
        "coping_strategies": case.get("coping_strategies", ""),
        "core_beliefs": build_core_beliefs(case),
        "relevant_history": case.get("history", ""),
    }
    return values[field]


def build_external_diagram(case: dict[str, Any], revealed_fields: list[str]) -> str:
    fields = [
        (EXTERNAL_FIELD_LABELS[field], get_external_field_value(case, field))
        for field in revealed_fields
    ]
    return format_revealed_diagram(fields)


def build_internal_diagram(case: dict[str, Any], revealed_fields: list[str]) -> str:
    fields = [
        (INTERNAL_FIELD_LABELS[field], get_internal_field_value(case, field))
        for field in revealed_fields
    ]
    return format_revealed_diagram(fields)


def get_score_guideline(guidelines: dict[str, dict[str, str]], score: int) -> str:
    guideline = guidelines[str(score)]
    return guideline["guideline"]


def format_client_system_prompt(
    template: str,
    case: dict[str, Any],
    openness_score: int,
    exploration_score: int,
    external_reveal_order: list[str],
    internal_reveal_order: list[str],
    openness_guidelines: dict[str, dict[str, str]],
) -> str:
    external_count = score_to_revealed_count(openness_score, len(external_reveal_order))
    internal_count = score_to_revealed_count(exploration_score, len(internal_reveal_order))

    return template.format(
        name=case.get("name", ""),
        traits=join_list(case.get("type")),
        external_diagram=build_external_diagram(case, external_reveal_order[:external_count]),
        internal_diagram=build_internal_diagram(case, internal_reveal_order[:internal_count]),
        openness_score=openness_score,
        exploration_score=exploration_score,
        openness_guideline=get_score_guideline(openness_guidelines, openness_score),
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


def parse_rating(text: str, default: int) -> int:
    match = re.search(r"\b([1-5])\b", text)
    if not match:
        return default
    return int(match.group(1))


def judge_score(
    judge_prompt: str,
    placeholder: str,
    dialogue: list[dict[str, Any]],
    model: str,
    default: int,
) -> int:
    from llm import call_llm_messages

    if not dialogue:
        return default

    prompt = judge_prompt.format(**{placeholder: render_dialogue_history(dialogue)})
    messages = [
        {"role": "system", "content": "You are a strict evaluator. Follow the requested output format."},
        {"role": "user", "content": prompt},
    ]
    response = call_llm_messages(messages, temperature=0.0, model=model)
    return parse_rating(response, default)


def should_run_judge(client_turn_number: int, cadence: int) -> bool:
    return client_turn_number > 0 and client_turn_number % cadence == 0


def judge_after_client_turn(
    dialogue: list[dict[str, Any]],
    client_turn_number: int,
    openness_cadence: int,
    judge_model: str,
    openness_score: int,
    exploration_score: int,
) -> tuple[int, int, bool, bool]:
    openness_prompt = read_text(JUDGES_DIR / "openness_judge.txt")
    exploration_prompt = read_text(JUDGES_DIR / "exploration_judge.txt")
    openness_judge_ran = should_run_judge(client_turn_number, openness_cadence)
    exploration_judge_ran = should_run_judge(client_turn_number, EXPLORATION_CADENCE)

    if openness_judge_ran:
        openness_score = judge_score(openness_prompt, "dialogue_context", dialogue, judge_model, openness_score)

    if exploration_judge_ran:
        exploration_score = judge_score(
            exploration_prompt,
            "dialogue_history",
            dialogue,
            judge_model,
            exploration_score,
        )

    return openness_score, exploration_score, openness_judge_ran, exploration_judge_ran


def format_score_source(judge_ran: bool) -> str:
    return "judged" if judge_ran else "reused"


def get_reveal_seed(case: dict[str, Any], random_seed: int | None) -> int | str:
    if random_seed is not None:
        return random_seed
    return str(case.get("id", ""))


def simulate_conversation(
    case: dict[str, Any],
    turn_pairs: int,
    conversation_model: str,
    judge_model: str,
    temperature: float,
    difficulty: str,
    random_seed: int | None = None,
) -> list[dict[str, Any]]:
    client_system_template = read_text(PROMPTS_DIR / "client_system.txt")
    client_user_template = read_text(PROMPTS_DIR / "client_user.txt")
    therapist_system_prompt = read_text(PROMPTS_DIR / "therapist_system.txt")
    therapist_user_template = read_text(PROMPTS_DIR / "therapist_user.txt")
    openness_guidelines = load_guidelines(OPENNESS_GUIDELINE_PATH, "openness_guidelines")

    reveal_seed = get_reveal_seed(case, random_seed)
    rng = random.Random(reveal_seed)
    external_reveal_order = build_external_reveal_order(rng)
    internal_reveal_order = list(INTERNAL_FIELD_LABELS)
    rng.shuffle(internal_reveal_order)

    dialogue: list[dict[str, Any]] = []
    openness_score = 1
    exploration_score = 1
    openness_cadence = OPENNESS_CADENCE_BY_DIFFICULTY[difficulty]

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
            f"Scores used for client turn {turn_number}: "
            f"openness={openness_score}, exploration={exploration_score}",
            flush=True,
        )
        external_count = score_to_revealed_count(openness_score, len(external_reveal_order))
        internal_count = score_to_revealed_count(exploration_score, len(internal_reveal_order))
        revealed_external_fields = external_reveal_order[:external_count]
        revealed_internal_fields = internal_reveal_order[:internal_count]
        print(
            "\nRevealed External Diagram:\n"
            f"{build_external_diagram(case, revealed_external_fields)}"
            "\n\nRevealed Internal Diagram:\n"
            f"{build_internal_diagram(case, revealed_internal_fields)}",
            flush=True,
        )
        client_system_prompt = format_client_system_prompt(
            client_system_template,
            case,
            openness_score,
            exploration_score,
            external_reveal_order,
            internal_reveal_order,
            openness_guidelines,
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
            "openness_score": openness_score,
            "exploration_score": exploration_score,
        }
        dialogue.append(client_turn)
        print(f"\nClient: {client_reply}", flush=True)

        (
            openness_score,
            exploration_score,
            openness_judge_ran,
            exploration_judge_ran,
        ) = judge_after_client_turn(
            dialogue=dialogue,
            client_turn_number=turn_number,
            openness_cadence=openness_cadence,
            judge_model=judge_model,
            openness_score=openness_score,
            exploration_score=exploration_score,
        )
        client_turn["post_client_openness_score"] = openness_score
        client_turn["post_client_exploration_score"] = exploration_score
        client_turn["openness_judge_ran"] = openness_judge_ran
        client_turn["exploration_judge_ran"] = exploration_judge_ran
        print(
            "\n"
            f"Judge scores after client turn {turn_number}: "
            f"openness={openness_score} ({format_score_source(openness_judge_ran)}), "
            f"exploration={exploration_score} ({format_score_source(exploration_judge_ran)})",
            flush=True,
        )

    return dialogue


def write_output(path: Path, case: dict[str, Any], dialogue: list[dict[str, Any]], args: argparse.Namespace) -> None:
    payload = {
        "case_id": case.get("id"),
        "case_name": case.get("name"),
        "conversation_model": args.conversation_model,
        "judge_model": args.judge_model,
        "temperature": args.temperature,
        "turn_pairs": args.turns,
        "total_messages": args.turns * 2,
        "difficulty": args.difficulty,
        "openness_judge_cadence": OPENNESS_CADENCE_BY_DIFFICULTY[args.difficulty],
        "exploration_judge_cadence": EXPLORATION_CADENCE,
        "initial_openness_score": 1,
        "initial_exploration_score": 1,
        "random_seed": get_reveal_seed(case, args.random_seed),
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
    parser.add_argument(
        "--judge-model",
        default="gpt-4o",
        help="OpenAI model for openness and exploration judges.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Optional seed for reproducible random CCD reveal order. Defaults to the case id.",
    )
    parser.add_argument(
        "--difficulty",
        choices=list(OPENNESS_CADENCE_BY_DIFFICULTY.keys()),
        default="normal",
        help="Client difficulty. Openness judge cadence: easy=2 client turns, normal=4, hard=6.",
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
        judge_model=args.judge_model,
        temperature=args.temperature,
        difficulty=args.difficulty,
        random_seed=args.random_seed,
    )

    if args.output:
        write_output(args.output, case, dialogue, args)
        print(f"\nSaved transcript to {args.output}")


if __name__ == "__main__":
    main()
