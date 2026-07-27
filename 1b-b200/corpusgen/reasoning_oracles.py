"""Independent exact-answer validators for the v3 reasoning task set."""

from __future__ import annotations

import math
import re
from collections import Counter, deque
from collections.abc import Mapping
from functools import cache
from typing import Any


class ReasoningOracleError(RuntimeError):
    """A native task row disagrees with an independent exact oracle."""


class ReasoningOracleRejection(ReasoningOracleError):
    """A deterministic row is valid upstream but unsuitable for training."""


def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ReasoningOracleError("reasoning row has no metadata object")
    return metadata


def _matrix(value: object, *, square: bool = True) -> list[list[int]]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(row, list) or not row for row in value)
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for row in value
            for item in row
        )
    ):
        raise ReasoningOracleError("reasoning row has an invalid integer matrix")
    width = len(value[0])
    if any(len(row) != width for row in value) or (square and len(value) != width):
        raise ReasoningOracleError("reasoning row matrix shape is invalid")
    return [list(row) for row in value]


def _course_schedule(metadata: Mapping[str, Any]) -> str:
    courses = metadata.get("courses")
    prerequisites = metadata.get("prerequisites")
    if (
        not isinstance(courses, list)
        or not courses
        or any(isinstance(item, bool) or not isinstance(item, int) for item in courses)
        or len(courses) != len(set(courses))
        or not isinstance(prerequisites, list)
    ):
        raise ReasoningOracleError("course_schedule inputs are invalid")
    course_set = set(courses)
    adjacency = {course: [] for course in courses}
    indegree = {course: 0 for course in courses}
    seen_edges = set()
    for edge in prerequisites:
        if (
            not isinstance(edge, (list, tuple))
            or len(edge) != 2
            or edge[0] not in course_set
            or edge[1] not in course_set
        ):
            raise ReasoningOracleError("course_schedule edge is invalid")
        course, prerequisite = edge
        if (prerequisite, course) not in seen_edges:
            adjacency[prerequisite].append(course)
            indegree[course] += 1
            seen_edges.add((prerequisite, course))
    queue = deque(sorted(course for course in courses if indegree[course] == 0))
    visited = 0
    while queue:
        prerequisite = queue.popleft()
        visited += 1
        for course in sorted(adjacency[prerequisite]):
            indegree[course] -= 1
            if indegree[course] == 0:
                queue.append(course)
    return str(visited == len(courses))


def _binary_matrix(metadata: Mapping[str, Any]) -> str:
    matrix = _matrix(metadata.get("matrix"))
    height, width = len(matrix), len(matrix[0])
    distances = [[-1] * width for _ in range(height)]
    queue = deque()
    for row in range(height):
        for column in range(width):
            if matrix[row][column] == 0:
                distances[row][column] = 0
                queue.append((row, column))
            elif matrix[row][column] != 1:
                raise ReasoningOracleError("binary_matrix contains a non-binary cell")
    if not queue:
        raise ReasoningOracleRejection("binary_matrix has no zero")
    while queue:
        row, column = queue.popleft()
        for delta_row, delta_column in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_row = row + delta_row
            next_column = column + delta_column
            if (
                0 <= next_row < height
                and 0 <= next_column < width
                and distances[next_row][next_column] == -1
            ):
                distances[next_row][next_column] = distances[row][column] + 1
                queue.append((next_row, next_column))
    return "\n".join(" ".join(map(str, row)) for row in distances)


def _rotten_oranges(metadata: Mapping[str, Any]) -> str:
    matrix = _matrix(metadata.get("matrix"))
    height, width = len(matrix), len(matrix[0])
    queue = deque()
    fresh = 0
    for row in range(height):
        for column in range(width):
            value = matrix[row][column]
            if value == 2:
                queue.append((row, column, 0))
            elif value == 1:
                fresh += 1
            elif value != 0:
                raise ReasoningOracleError("rotten_oranges contains an invalid cell")
    infected = 0
    elapsed = 0
    while queue:
        row, column, minute = queue.popleft()
        elapsed = max(elapsed, minute)
        for delta_row, delta_column in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_row = row + delta_row
            next_column = column + delta_column
            if (
                0 <= next_row < height
                and 0 <= next_column < width
                and matrix[next_row][next_column] == 1
            ):
                matrix[next_row][next_column] = 2
                infected += 1
                queue.append((next_row, next_column, minute + 1))
    return str(elapsed if infected == fresh else -1)


def _futoshiki(metadata: Mapping[str, Any]) -> str:
    puzzle = _matrix(metadata.get("puzzle"))
    size = metadata.get("board_size")
    raw_constraints = metadata.get("constraints")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size != len(puzzle)
        or not isinstance(raw_constraints, list)
        or any(value < 0 or value > size for row in puzzle for value in row)
    ):
        raise ReasoningOracleError("futoshiki inputs are invalid")
    constraints: dict[tuple[tuple[int, int], tuple[int, int]], str] = {}
    for item in raw_constraints:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 5
            or item[4] not in ("<", ">")
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value >= size
                for value in item[:4]
            )
        ):
            raise ReasoningOracleError("futoshiki constraint is invalid")
        row_a, column_a, row_b, column_b, relation = item
        if abs(row_a - row_b) + abs(column_a - column_b) != 1:
            raise ReasoningOracleError("futoshiki constraint is not adjacent")
        constraints[((row_a, column_a), (row_b, column_b))] = relation

    grid = [row[:] for row in puzzle]
    solutions: list[list[list[int]]] = []

    def relation_valid(
        row_a: int,
        column_a: int,
        row_b: int,
        column_b: int,
        relation: str,
    ) -> bool:
        first = grid[row_a][column_a]
        second = grid[row_b][column_b]
        if not first or not second:
            return True
        return first < second if relation == "<" else first > second

    def candidates(row: int, column: int) -> list[int]:
        unavailable = set(grid[row])
        unavailable.update(grid[index][column] for index in range(size))
        values = []
        for value in range(1, size + 1):
            if value in unavailable:
                continue
            grid[row][column] = value
            if all(
                relation_valid(*first, *second, relation)
                for (first, second), relation in constraints.items()
                if (row, column) in (first, second)
            ):
                values.append(value)
            grid[row][column] = 0
        return values

    def solve() -> None:
        if len(solutions) >= 2:
            return
        unfilled = [
            (len(options), row, column, options)
            for row in range(size)
            for column in range(size)
            if grid[row][column] == 0
            for options in (candidates(row, column),)
        ]
        if not unfilled:
            expected = set(range(1, size + 1))
            if (
                all(set(row) == expected for row in grid)
                and all(
                    {grid[row][column] for row in range(size)} == expected
                    for column in range(size)
                )
                and all(
                    relation_valid(*first, *second, relation)
                    for (first, second), relation in constraints.items()
                )
            ):
                solutions.append([row[:] for row in grid])
            return
        _, row, column, options = min(unfilled)
        for value in options:
            grid[row][column] = value
            solve()
            grid[row][column] = 0
            if len(solutions) >= 2:
                return

    if any(
        len([value for value in row if value]) != len({value for value in row if value})
        for row in grid
    ) or any(
        len([grid[row][column] for row in range(size) if grid[row][column]])
        != len({grid[row][column] for row in range(size) if grid[row][column]})
        for column in range(size)
    ):
        raise ReasoningOracleError("futoshiki clues violate Latin constraints")
    solve()
    if len(solutions) != 1:
        raise ReasoningOracleRejection(
            f"futoshiki has {len(solutions)} independently found solutions"
        )

    def relation_at(row_a: int, column_a: int, row_b: int, column_b: int) -> str:
        direct = constraints.get(((row_a, column_a), (row_b, column_b)))
        if direct is None:
            reverse = constraints.get(((row_b, column_b), (row_a, column_a)))
            if reverse == "<":
                direct = ">"
            elif reverse == ">":
                direct = "<"
        if direct is None:
            return " "
        if row_a == row_b:
            return direct
        return "∧" if direct == "<" else "∨"

    solution = solutions[0]
    lines = []
    for row in range(size):
        cells = []
        for column in range(size):
            cells.append(str(solution[row][column]))
            if column < size - 1:
                cells.append(relation_at(row, column, row, column + 1))
        lines.append(" ".join(cells))
        if row < size - 1:
            vertical = []
            for column in range(size):
                vertical.append(relation_at(row, column, row + 1, column))
                if column < size - 1:
                    vertical.append(" ")
            lines.append(" ".join(vertical))
    return "\n".join(lines)


def _ransom_note(
    row: Mapping[str, Any],
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    question = row.get("question")
    note = metadata.get("ransom_note")
    magazine = metadata.get("magazine")
    note_length = metadata.get("note_length")
    magazine_length = metadata.get("magazine_length")
    declared_solvable = metadata.get("solvable")
    if (
        not isinstance(question, str)
        or not isinstance(note, str)
        or not note
        or not isinstance(magazine, str)
        or not magazine
        or set(note + magazine) - set("abcdefghijklmnopqrstuvwxyz")
        or isinstance(note_length, bool)
        or not isinstance(note_length, int)
        or isinstance(magazine_length, bool)
        or not isinstance(magazine_length, int)
        or note_length != len(note)
        or magazine_length != len(magazine)
        or not isinstance(declared_solvable, bool)
        or not config["min_note_length"] <= note_length <= config["max_note_length"]
        or not config["min_magazine_length"]
        <= magazine_length
        <= config["max_magazine_length"]
        or f"Ransom note: {note}\nMagazine: {magazine}\n" not in question
    ):
        raise ReasoningOracleError("ransom_note inputs are invalid")
    note_counts = Counter(note)
    magazine_counts = Counter(magazine)
    answer = all(
        magazine_counts[character] >= count for character, count in note_counts.items()
    )
    if answer != declared_solvable:
        raise ReasoningOracleError("ransom_note solvability metadata differs")
    return str(answer)


def _dice(metadata: Mapping[str, Any]) -> str:
    puzzle = metadata.get("puzzle")
    if not isinstance(puzzle, Mapping):
        raise ReasoningOracleError("dice puzzle metadata is invalid")
    dice_text = puzzle.get("dice_str")
    target = puzzle.get("target")
    if (
        not isinstance(dice_text, str)
        or isinstance(target, bool)
        or not isinstance(target, int)
    ):
        raise ReasoningOracleError("dice inputs are invalid")
    dice = [int(value) for value in re.findall(r"1d(\d+)", dice_text)]
    if not dice or any(sides < 2 for sides in dice):
        raise ReasoningOracleError("dice description is invalid")
    counts = [1]
    for sides in dice:
        next_counts = [0] * (len(counts) + sides)
        for subtotal, count in enumerate(counts):
            for face in range(1, sides + 1):
                next_counts[subtotal + face] += count
        counts = next_counts
    numerator = sum(counts[target:]) if target < len(counts) else 0
    denominator = math.prod(dice)
    divisor = math.gcd(numerator, denominator)
    return f"{numerator // divisor}/{denominator // divisor}"


def _string_splitting(
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    initial = metadata.get("initial_machines")
    maximum = config.get("max_iterations")
    if (
        not isinstance(initial, (list, tuple))
        or len(initial) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in initial
        )
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or maximum <= 0
    ):
        raise ReasoningOracleError("string_splitting inputs are invalid")
    counts = [*initial, 0, 0, 0]
    seen = {tuple(counts)}
    for _ in range(maximum):
        next_counts = counts[:]
        if next_counts[0] >= 1:
            next_counts[0] -= 1
            next_counts[3] += 2
            next_counts[4] += 1
        elif next_counts[1] >= 2:
            next_counts[1] -= 2
            next_counts[3] += 1
        elif next_counts[2] >= 2:
            next_counts[2] -= 2
            next_counts[4] += 1
        elif next_counts[1] >= 1 and next_counts[2] >= 1:
            next_counts[1] -= 1
            next_counts[2] -= 1
            next_counts[0] += 1
        elif next_counts[3] >= 1 and next_counts[4] >= 1:
            next_counts[3] -= 1
            next_counts[4] -= 1
            next_counts[5] += 1
        state = tuple(next_counts)
        if state in seen:
            break
        seen.add(state)
        counts = next_counts
    return " ".join(map(str, counts))


def _path_star(row: Mapping[str, Any]) -> str:
    question = row.get("question")
    if not isinstance(question, str) or "Solve the following task:\n" not in question:
        raise ReasoningOracleError("path_star question is invalid")
    task = question.rsplit("Solve the following task:\n", 1)[1].strip()
    try:
        edges_text, query = task.split("/", 1)
        endpoints, _blank = query.split("=", 1)
        start_text, goal_text = endpoints.split()
        start, goal = int(start_text), int(goal_text)
    except (TypeError, ValueError) as error:
        raise ReasoningOracleError("path_star task syntax is invalid") from error
    adjacency: dict[int, list[int]] = {}
    for item in edges_text.split("|"):
        if not item:
            continue
        try:
            first_text, second_text = item.split()
            first, second = int(first_text), int(second_text)
        except ValueError as error:
            raise ReasoningOracleError("path_star edge syntax is invalid") from error
        adjacency.setdefault(first, []).append(second)
        adjacency.setdefault(second, []).append(first)
    queue = deque([start])
    parent: dict[int, int | None] = {start: None}
    while queue and goal not in parent:
        node = queue.popleft()
        for neighbor in sorted(adjacency.get(node, [])):
            if neighbor not in parent:
                parent[neighbor] = node
                queue.append(neighbor)
    if goal not in parent:
        raise ReasoningOracleError("path_star goal is unreachable")
    path = []
    node: int | None = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return " ".join(map(str, reversed(path)))


def _largest_island(
    row: Mapping[str, Any],
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    question = row.get("question")
    grid = _matrix(metadata.get("grid"), square=False)
    rows, columns = len(grid), len(grid[0])
    if (
        not isinstance(question, str)
        or any(value not in (0, 1) for line in grid for value in line)
        or not config["min_rows"] <= rows <= config["max_rows"]
        or not config["min_cols"] <= columns <= config["max_cols"]
    ):
        raise ReasoningOracleError("largest_island inputs are invalid")
    rendered = "\n".join(" ".join(map(str, line)) for line in grid)
    if f"{rows} x {columns} binary matrix grid:\n{rendered}\n" not in question:
        raise ReasoningOracleError("largest_island visible grid differs")
    visited: set[tuple[int, int]] = set()
    maximum = 0
    for row_index in range(rows):
        for column_index in range(columns):
            origin = (row_index, column_index)
            if grid[row_index][column_index] != 1 or origin in visited:
                continue
            area = 0
            queue = deque([origin])
            visited.add(origin)
            while queue:
                current_row, current_column = queue.popleft()
                area += 1
                for delta_row, delta_column in (
                    (1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                ):
                    neighbor = (
                        current_row + delta_row,
                        current_column + delta_column,
                    )
                    if (
                        0 <= neighbor[0] < rows
                        and 0 <= neighbor[1] < columns
                        and grid[neighbor[0]][neighbor[1]] == 1
                        and neighbor not in visited
                    ):
                        visited.add(neighbor)
                        queue.append(neighbor)
            maximum = max(maximum, area)
    return str(maximum)


def _rotate_matrix(metadata: Mapping[str, Any]) -> str:
    matrix = _matrix(metadata.get("matrix"))
    rotations = metadata.get("num_rotations")
    if isinstance(rotations, bool) or not isinstance(rotations, int) or rotations < 0:
        raise ReasoningOracleError("rotate_matrix rotation count is invalid")
    for _ in range(rotations % 4):
        matrix = [list(row) for row in zip(*matrix[::-1])]
    return "\n".join(" ".join(map(str, row)) for row in matrix)


def _spiral_matrix(metadata: Mapping[str, Any]) -> str:
    matrix = _matrix(metadata.get("matrix"))
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    values = []
    while top <= bottom and left <= right:
        values.extend(matrix[top][left : right + 1])
        top += 1
        for row in range(top, bottom + 1):
            values.append(matrix[row][right])
        right -= 1
        if top <= bottom:
            values.extend(reversed(matrix[bottom][left : right + 1]))
            bottom -= 1
        if left <= right:
            for row in range(bottom, top - 1, -1):
                values.append(matrix[row][left])
            left += 1
    return " ".join(map(str, values))


def _prime_factorization(metadata: Mapping[str, Any]) -> str:
    number = metadata.get("number")
    if isinstance(number, bool) or not isinstance(number, int) or number < 2:
        raise ReasoningOracleError("prime_factorization number is invalid")
    factors = []
    divisor = 2
    remainder = number
    while divisor * divisor <= remainder:
        while remainder % divisor == 0:
            factors.append(divisor)
            remainder //= divisor
        divisor = 3 if divisor == 2 else divisor + 2
    if remainder > 1:
        factors.append(remainder)
    return " × ".join(map(str, factors))


_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"


def _base_conversion(metadata: Mapping[str, Any]) -> str:
    source = metadata.get("source_repr")
    source_base = metadata.get("source_base")
    target_base = metadata.get("target_base")
    if (
        not isinstance(source, str)
        or not source
        or isinstance(source_base, bool)
        or not isinstance(source_base, int)
        or isinstance(target_base, bool)
        or not isinstance(target_base, int)
        or not 2 <= source_base <= 36
        or not 2 <= target_base <= 36
    ):
        raise ReasoningOracleError("base_conversion inputs are invalid")
    value = 0
    for character in source.lower():
        digit = _DIGITS.find(character)
        if digit < 0 or digit >= source_base:
            raise ReasoningOracleError("base_conversion source digit is invalid")
        value = value * source_base + digit
    if value == 0:
        return "0"
    output = []
    while value:
        value, digit = divmod(value, target_base)
        output.append(_DIGITS[digit])
    return "".join(reversed(output))


@cache
def _prime_prefix(limit: int) -> tuple[int, ...]:
    if limit < 2:
        return tuple(0 for _ in range(limit + 1))
    prime = bytearray(b"\x01") * (limit + 1)
    prime[0:2] = b"\x00\x00"
    for divisor in range(2, math.isqrt(limit) + 1):
        if prime[divisor]:
            start = divisor * divisor
            prime[start : limit + 1 : divisor] = b"\x00" * (
                (limit - start) // divisor + 1
            )
    total = 0
    prefix = []
    for value in prime:
        total += value
        prefix.append(total)
    return tuple(prefix)


def _count_primes(
    row: Mapping[str, Any],
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    question = row.get("question")
    start = metadata.get("start")
    end = metadata.get("end")
    if (
        not isinstance(question, str)
        or isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or not config["min_n"] <= start <= end <= config["max_n"]
        or question
        != f"Count how many prime numbers there are between {start} and {end} "
        "(inclusive) ?"
    ):
        raise ReasoningOracleError("count_primes inputs are invalid")
    prefix = _prime_prefix(config["max_n"])
    answer = prefix[end] - (prefix[start - 1] if start else 0)
    return str(answer)


def canonical_reasoning_answer(
    task: str,
    row: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    """Return a canonical answer computed without native solution fields."""

    metadata = _metadata(row)
    if task == "course_schedule":
        return _course_schedule(metadata)
    if task == "binary_matrix":
        return _binary_matrix(metadata)
    if task == "rotten_oranges":
        return _rotten_oranges(metadata)
    if task == "futoshiki":
        return _futoshiki(metadata)
    if task == "ransom_note":
        return _ransom_note(row, metadata, config)
    if task == "dice":
        return _dice(metadata)
    if task == "string_splitting":
        return _string_splitting(metadata, config)
    if task == "path_star":
        return _path_star(row)
    if task == "largest_island":
        return _largest_island(row, metadata, config)
    if task == "rotate_matrix":
        return _rotate_matrix(metadata)
    if task == "spiral_matrix":
        return _spiral_matrix(metadata)
    if task == "prime_factorization":
        return _prime_factorization(metadata)
    if task == "base_conversion":
        return _base_conversion(metadata)
    if task == "count_primes":
        return _count_primes(row, metadata, config)
    raise ReasoningOracleError(f"no independent oracle for task: {task}")
