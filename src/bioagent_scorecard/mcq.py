"""Multiple-choice prompt building and answer parsing.

The harness follows the LAB-Bench format: the options are the ideal answer,
the distractors and one refusal option. The option order comes from a seeded
shuffle per task, so the same seed gives the same letters on every machine.

Text normalisation (LAB-Bench issue #12): some questions and distractors have
trailing whitespace or newlines, so a plain string compare fails. The harness
compares and shows option text after `norm_ws`. Options that are the same after
normalisation are shown once. Example: one SeqQA task of the older data revision
(id 6ce2c743) lists its ideal answer a second time as a distractor. Without the dedupe the
prompt shows two identical options and only one of them is graded correct.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
import re

from .tasks import Task

# The same text as REFUSE_CHOICE in the LAB-Bench harness (labbench/utils.py).
REFUSE_TEXT = "Insufficient information to answer the question"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

INSTRUCTIONS = (
    "The following is a multiple choice question about biology. "
    "Answer with the letter of one option. "
    "If you are not sure, pick the option that says the information is insufficient. "
    "Put the letter inside [ANSWER] and [/ANSWER] tags, for example [ANSWER]B[/ANSWER]."
)


@dataclasses.dataclass(frozen=True)
class MCQ:
    task_id: str
    prompt: str
    options: tuple[str, ...]
    correct_letter: str
    refuse_letter: str
    #: Distractors removed because they are the same text (after norm_ws) as
    #: the ideal answer or as an earlier distractor.
    dropped_duplicates: int = 0

    @property
    def letters(self) -> tuple[str, ...]:
        return tuple(LETTERS[: len(self.options)])


def task_rng(seed: int, task_id: str) -> random.Random:
    """One RNG per (seed, task). Order of execution does not change the result."""
    digest = hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def norm_ws(text: str) -> str:
    """Normalise whitespace for display: strip both ends, collapse spaces and
    tabs inside each line, drop blank lines. Inner line breaks stay, because a
    table or a sequence in an option can depend on them. Case stays."""
    lines = (" ".join(line.split()) for line in text.strip().splitlines())
    return "\n".join(line for line in lines if line)


def match_key(text: str) -> str:
    """The comparison key: every whitespace run (line breaks too) is one space."""
    return " ".join(text.split())


def same_text(a: str, b: str) -> bool:
    """Exact match after whitespace normalisation (LAB-Bench issue #12)."""
    return match_key(a) == match_key(b)


def unique_options(task: Task) -> tuple[list[str], int]:
    """Return [ideal, distractors..., refusal] after norm_ws, without duplicates.

    The ideal answer always stays. A distractor that equals the ideal answer,
    the refusal text or an earlier distractor is dropped. The second value is
    the number of dropped distractors.
    """
    seen = {match_key(task.ideal), REFUSE_TEXT}
    options = [norm_ws(task.ideal)]
    dropped = 0
    for d in task.distractors:
        key = match_key(d)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        options.append(norm_ws(d))
    options.append(REFUSE_TEXT)
    return options, dropped


def build_mcq(task: Task, seed: int) -> MCQ:
    options, dropped = unique_options(task)
    if len(options) > len(LETTERS):
        raise ValueError(f"task {task.id}: too many options")
    ideal = options[0]
    task_rng(seed, task.id).shuffle(options)
    letters = LETTERS[: len(options)]
    # Only the ends of the question are stripped. Inner line breaks stay,
    # because a sequence or a table in the question can depend on them.
    lines = [INSTRUCTIONS, "", "Question: " + task.question.strip(), "", "Options:"]
    lines += [f"{letters[i]}) {opt}" for i, opt in enumerate(options)]
    return MCQ(
        task_id=task.id,
        prompt="\n".join(lines),
        options=tuple(options),
        correct_letter=letters[options.index(ideal)],
        refuse_letter=letters[options.index(REFUSE_TEXT)],
        dropped_duplicates=dropped,
    )


_TAGGED = re.compile(r"\[ANSWER\]\s*([A-Z])\s*\[/ANSWER\]", re.IGNORECASE)
_BARE = re.compile(r"^\s*\(?([A-Z])\)?[.)]?\s*$", re.IGNORECASE)


def parse_letter(text: str | None, mcq: MCQ) -> str | None:
    """Return the chosen letter, or None when the answer is not parsable.

    The parser reads the [ANSWER]X[/ANSWER] tags. If two tags name different
    letters (a hedge such as "A or C", or an echo of the format example), the
    reply is unparsable. Without a tag it accepts only a reply that is one bare
    letter. It does not guess from free text, because a guess would hide
    format failures. Upper and lower case are the same for both forms.
    """
    if not text:
        return None
    tagged = {t.upper() for t in _TAGGED.findall(text)}
    if len(tagged) > 1:
        return None
    if tagged:
        letter = tagged.pop()
    else:
        m = _BARE.match(text)
        letter = m.group(1).upper() if m else None
    if letter is None or letter not in mcq.letters:
        return None
    return letter
