"""Human-readable names for the real-world manipulation tasks.

Shared by every real-world figure. It lives here rather than in each plot
script because the same task is named in more than one figure of the same
paper, and two figures calling one task different things is the sort of thing
nobody catches in review.

The CSVs key on the short ids (`task1`, …); these are the display strings.

Asterisks mark the word that distinguishes a task from its category partner --
the object for the novel-object pair, the container for the novel-place pair.
Categories with no partner (task5, task6) mark the defining word or phrase.
Tick labels set that text bold so a reader scanning the x axis lands on what
actually changed instead of re-reading near-identical phrases. Plain-text
consumers (tables, the scaling figure's axis label) strip the markers.
"""

import re

TASK_LABELS = {
    "task1": "*Milk* on plate",
    "task2": "*Salt* on plate",
    "task3": "Banana in *box*",
    "task4": "Banana in *bowl*",
    "task5": "Push milk *right*",
    "task6": "Banana in *bowl*",
}

# Which axis of generalization each task probes. Two tasks per category, and the
# tasks are ordered so each category's pair is contiguous -- the bar chart's x
# axis then reads as two blocks of two, and a reader can compare within a
# category without hunting.
TASK_CATEGORIES = {
    "task1": "[novel obj]",
    "task2": "[novel obj]",
    "task3": "[novel place]",
    "task4": "[novel place]",
    "task5": "[novel move]",
    "task6": "[novel bkg]",
}

# The same categories with "novel" dropped, for panels too narrow for the full
# form -- a third of \textwidth gives a six-group axis about 0.41in per tick,
# and "[novel place]" alone is 0.61in. Every task carries both spellings so the
# short one cannot drift into naming a different set of categories; the figure
# that uses it has to expand them in its caption.
TASK_CATEGORIES_SHORT = {
    "task1": "[obj]",
    "task2": "[obj]",
    "task3": "[place]",
    "task4": "[place]",
    "task5": "[move]",
    "task6": "[bkg]",
}

_EMPH = re.compile(r"\*([^*]+)\*")


def _plain(text):
    return _EMPH.sub(r"\1", text)


def _bold(text):
    r"""Render \*emphasis\* as mathtext bold.

    matplotlib sets one font weight per text object, so a label that is bold in
    part has to route the bold run through mathtext. With mathtext.fontset="cm"
    (see style.py) ``\mathbf`` resolves to Computer Modern bold, the same family
    the surrounding plain text is set in, so the two runs sit together as one
    phrase rather than reading as two typefaces.
    """
    return _EMPH.sub(r"$\\mathbf{\1}$", text)


def label(task):
    """One-line display name, unmarked -- for axis labels, captions and tables."""
    return _plain(TASK_LABELS.get(str(task), str(task)))


def category(task, short=False):
    table = TASK_CATEGORIES_SHORT if short else TASK_CATEGORIES
    return table.get(str(task), "")


def wrapped(task, short=False):
    """Category, then the name broken after its first word, for x tick labels.

    Three lines, not two: at column width four groups leave about 0.75in per
    tick, and "Banana in box" set on one line runs into "Banana in bowl" next to
    it with no gap at all. Breaking after the first word is what buys the
    clearance -- the category line is the widest thing left and it still fits.

    Every label gets the same number of lines whether or not its category is
    known, so the tick row is one height for all of them and the baselines stay
    level. The alternative, rotating the labels, costs more vertical space than
    the extra line does and reads worse.

    ``short=True`` swaps in the abbreviated category (see
    TASK_CATEGORIES_SHORT). The layout is otherwise identical, so a narrow panel
    and a full-width one still stack their labels the same way.
    """
    raw = TASK_LABELS.get(str(task), str(task))
    head, _, tail = raw.partition(" ")
    name = f"{head}\n{tail}" if tail else head
    cat = category(task, short=short)
    return _bold(f"{cat}\n{name}" if cat else name)
