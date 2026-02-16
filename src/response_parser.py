"""Robust extraction of answer letters from model output."""

import re


def parse_response(raw: str, num_choices: int = 3) -> str | None:
    """Extract answer letter from model output.

    Handles common response formats:
    - Just the letter: "A"
    - Letter with punctuation: "A)" or "A." or "(A)"
    - Sentence form: "The answer is A" / "I would choose B"
    - Fallback: first valid letter in the response

    Args:
        raw: Raw model output string.
        num_choices: Number of valid answer choices (default 3 for A/B/C).

    Returns:
        Uppercase letter ('A', 'B', or 'C') or None if unparseable.
    """
    text = raw.strip()
    valid = set("ABC"[:num_choices])

    # Case 1: Just the letter
    if text.upper() in valid:
        return text.upper()

    # Case 2: "A)" or "A." or "(A)"
    m = re.match(r"^\(?([A-C])\)?[.)\s]", text, re.IGNORECASE)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # Case 3: "The answer is A" / "I would choose B"
    m = re.search(
        r"(?:answer|choose|select|pick)\s*(?:is\s*)?\(?([A-C])\)?",
        text,
        re.IGNORECASE,
    )
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # Case 4: First capital letter that's a valid choice
    for char in text:
        if char.upper() in valid:
            return char.upper()

    return None  # Unparseable
