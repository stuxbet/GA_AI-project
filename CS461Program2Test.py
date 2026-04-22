#CS461-001 Program 2
#Contributers - Dylan G, Sheyla R, Luke M
import sys
import random
import itertools
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
from numpy.random import PCG64DXSM, Generator
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

#Start of part 1 per the docuement


# DOMAIN DATA

TIMES = ["10 AM", "11 AM", "12 PM", "1 PM", "2 PM", "3 PM"]

# Room name and capacity can be changed
ROOMS: dict[str, int] = {
    "Beach 201":   18,
    "Beach 301":   25,
    "Frank 119":   95,
    "Loft 206":    55,
    "Loft 310":    48,
    "James 325":  110,
    "Roman 201":   40,
    "Roman 216":   80,
    "Slater 003":  32,
}

FACILITATORS: list[str] = [
    "Lock", "Glen", "Banks", "Richards", "Shaw",
    "Singer", "Uther", "Tyler", "Numen", "Zeldin",
]

# Each activity: name, expected enrollment, preferred facilitators, other facilitators, contains all listed classes in the assignment's appendix
ACTIVITIES_DEF: list[dict] = [
    {
        "name": "SLA101A",
        "enrollment": 40,
        "preferred": ["Glen", "Lock", "Banks"],
        "other":     ["Numen", "Richards", "Shaw", "Singer"],
    },
    {
        "name": "SLA101B",
        "enrollment": 35,
        "preferred": ["Glen", "Lock", "Banks"],
        "other":     ["Numen", "Richards", "Shaw", "Singer"],
    },
    {
        "name": "SLA191A",
        "enrollment": 45,
        "preferred": ["Glen", "Lock", "Banks"],
        "other":     ["Numen", "Richards", "Shaw", "Singer"],
    },
    {
        "name": "SLA191B",
        "enrollment": 40,
        "preferred": ["Glen", "Lock", "Banks"],
        "other":     ["Numen", "Richards", "Shaw", "Singer"],
    },
    {
        "name": "SLA201",
        "enrollment": 60,
        "preferred": ["Glen", "Banks", "Zeldin", "Lock", "Singer"],
        "other":     ["Richards", "Uther", "Shaw"],
    },
    {
        "name": "SLA291",
        "enrollment": 50,
        "preferred": ["Glen", "Banks", "Zeldin", "Lock", "Singer"],
        "other":     ["Richards", "Uther", "Shaw"],
    },
    {
        "name": "SLA303",
        "enrollment": 25,
        "preferred": ["Glen", "Zeldin"],
        "other":     ["Banks"],
    },
    {
        "name": "SLA304",
        "enrollment": 20,
        "preferred": ["Singer", "Uther"],
        "other":     ["Richards"],
    },
    {
        "name": "SLA394",
        "enrollment": 15,
        "preferred": ["Tyler", "Singer"],
        "other":     ["Richards", "Zeldin"],
    },
    {
        "name": "SLA449",
        "enrollment": 30,
        "preferred": ["Tyler", "Zeldin", "Uther"],
        "other":     ["Zeldin", "Shaw"],
    },
    {
        "name": "SLA451",
        "enrollment": 90,
        "preferred": ["Lock", "Banks", "Zeldin"],
        "other":     ["Tyler", "Singer", "Shaw", "Glen"],
    },
]

ROOM_NAMES:       list[str] = list(ROOMS.keys())

# GA configuration (Part 3)
POPULATION_SIZE:       int   = 1000   # total schedules per generation
ELITE_COUNT:           int   = 100    # top 10% carried over (and used as the breeding pool)
PAIR_COUNT:            int   = 450    # 450 pairs * 2 children = 900 offspring
MUTATION_RATE:         float = 0.01   # per-gene probability, halved on plateau
MIN_GENERATIONS:       int   = 100
IMPROVEMENT_THRESHOLD: float = 0.01   # 1% avg-fitness improvement threshold
PLATEAU_DEBOUNCE:      int   = 2      # consecutive sub-1% gens before plateau action
HALVING_COOLDOWN:      int   = 20     # min gens between lambda halvings
MAX_HALVINGS:          int   = 4      # cap on lambda halvings before termination

BEST_SCHEDULE_FILE:  str = "best_schedule.txt"
FITNESS_HISTORY_CSV: str = "fitness_history.csv"


# DATA STRUCTURES

@dataclass
class ActivityDef:
    """Static definition of an activity (read-only reference data)."""
    name:       str
    enrollment: int
    preferred:  list[str]
    other:      list[str]


@dataclass
class Assignment:
    """
    A single activity's assignment within one schedule.
    When you implement the GA in this can be adjusted
    """
    activity:    ActivityDef   # reference to the static definition
    room:        str           # chosen room name
    time:        str           # chosen time slot
    facilitator: str           # chosen facilitator name


@dataclass
class Schedule:
    """
    One candidate schedule: 11 assignments (one per activity).
    fitness is set to None until scored in Part 2
    """
    assignments: list[Assignment]
    fitness:     Optional[float] = field(default=None)

    def copy(self) -> "Schedule":
        """Return a deep copy suitable for offspring creation."""
        new_assignments = [
            Assignment(
                activity=a.activity,
                room=a.room,
                time=a.time,
                facilitator=a.facilitator,
            )
            for a in self.assignments
        ]
        return Schedule(assignments=new_assignments)



# RANDOM SCHEDULE GENERATION


# Uses NumPy's PCG64DXSM for better random numbers
_rng = Generator(PCG64DXSM())

# Build ActivityDef objects once
ACTIVITY_DEFS: list[ActivityDef] = [
    ActivityDef(**d) for d in ACTIVITIES_DEF
]


def random_assignment(activity_def: ActivityDef) -> Assignment:
    """Randomly assigns a room, time, and facilitator to a single activity"""
    room        = ROOM_NAMES[_rng.integers(0, len(ROOM_NAMES))]
    time        = TIMES[_rng.integers(0, len(TIMES))]
    facilitator = FACILITATORS[_rng.integers(0, len(FACILITATORS))]
    return Assignment(activity=activity_def, room=room, time=time, facilitator=facilitator)


def generate_random_schedule() -> Schedule:
    """Creates a fully random schedule covering all 11 activities"""
    assignments = [random_assignment(ad) for ad in ACTIVITY_DEFS]
    return Schedule(assignments=assignments)


def generate_population(size: int = POPULATION_SIZE) -> list[Schedule]:
    """Generates an initial population of `size` random schedules, size can be changed above"""
    return [generate_random_schedule() for _ in range(size)]



# DISPLAY HELPERS

def _facilitator_label(assignment: Assignment) -> str:
    """Annotate facilitator with (P)referred, (O)ther, or (X) for neither."""
    f = assignment.facilitator
    if f in assignment.activity.preferred:
        return f"{f} (Preferred)"
    if f in assignment.activity.other:
        return f"{f} (Other)"
    return f"{f} (Not Prefered)"


def print_schedule(schedule: Schedule, sort_by: str = "time") -> None:
    """
    Pretty-print a schedule to stdout.

    sort_by: "time"     → ordered by time slot then activity name
             "activity" → ordered by activity name
    """
    assignments = schedule.assignments[:]

    if sort_by == "time":
        time_order = {t: i for i, t in enumerate(TIMES)}
        assignments.sort(key=lambda a: (time_order[a.time], a.activity.name))
        header = "Schedule — sorted by Time Slot"
    else:
        assignments.sort(key=lambda a: a.activity.name)
        header = "Schedule — sorted by Activity"

    col_w = [12, 14, 12, 28, 10]   # activity, room, time, facilitator, enrollment
    sep   = "─" * (sum(col_w) + len(col_w) * 3 + 1)

    print()
    print("=" * len(sep))
    print(f"  {header}")
    print("=" * len(sep))

    # column headers
    print(
        f"  {'Activity':<{col_w[0]}}  "
        f"{'Room':<{col_w[1]}}  "
        f"{'Time':<{col_w[2]}}  "
        f"{'Facilitator':<{col_w[3]}}  "
        f"{'Enroll':>{col_w[4]}}"
    )
    print(sep)

    prev_time = None
    for a in assignments:
        # blank separator between time groups when sorted by time
        if sort_by == "time" and a.time != prev_time and prev_time is not None:
            print()
        prev_time = a.time

        cap   = ROOMS[a.room]
        enroll = a.activity.enrollment
        room_note = ""
        if cap < enroll:
            room_note = " [TOO SMALL]"
        elif cap > 3 * enroll:
            room_note = " [>>3x]"
        elif cap > 1.5 * enroll:
            room_note = " [>1.5x]"

        print(
            f"  {a.activity.name:<{col_w[0]}}  "
            f"{a.room + room_note:<{col_w[1] + len(room_note)}}  "
            f"{a.time:<{col_w[2]}}  "
            f"{_facilitator_label(a):<{col_w[3]}}  "
            f"{enroll:>{col_w[4]}}"
        )

    print(sep)
    if schedule.fitness is not None:
        print(f"  Fitness: {schedule.fitness:.4f}")
    else:
        print("  Fitness: (not yet evaluated)")
    print()


# Start of Part 2


# Map each time slot to an index so we can do distance comparisons easily
TIME_INDEX: dict[str, int] = {t: i for i, t in enumerate(TIMES)}

ROMAN_BEACH_SCALAR = {"Roman 201", "Roman 216", "Beach 201", "Beach 301"}


def score_schedule(schedule: Schedule) -> float:
    # Scores a single schedule and stores the result in schedule.fitness
    # Returns the fitness value. Higher is better.
    total = 0.0

    # Counts how many times each facilitator appears at each time slot, and how many total activities they're assigned across the schedule
    fac_at_time:  dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    fac_total:    dict[str, int]            = defaultdict(int)
    room_at_time: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for a in schedule.assignments:
        fac_at_time[a.facilitator][a.time] += 1
        fac_total[a.facilitator]           += 1
        room_at_time[a.room][a.time]       += 1

    for a in schedule.assignments:
        cap    = ROOMS[a.room]
        enroll = a.activity.enrollment

        # Room size scoring
        if cap < enroll:
            total -= 0.5
        elif cap > 3 * enroll:
            total -= 0.4
        elif cap > 1.5 * enroll:
            total -= 0.2
        else:
            total += 0.3

        # Two activities booked in the same room at the same time
        if room_at_time[a.room][a.time] > 1:
            total -= 0.5

        # Facilitator preference
        if a.facilitator in a.activity.preferred:
            total += 0.5
        elif a.facilitator in a.activity.other:
            total += 0.2
        else:
            total -= 0.1

        # Facilitator load in this time slot
        if fac_at_time[a.facilitator][a.time] == 1:
            total += 0.2
        elif fac_at_time[a.facilitator][a.time] > 1:
            total -= 0.2

    # Penalize or reward based on how many total activities a facilitator teaches
    for fac, count in fac_total.items():
        if count > 4:
            total -= 0.5
        elif count < 3:
            # Tyler is committee chair: no penalty if he has fewer than 2 activities
            if not (fac == "Tyler" and count < 2):
                total -= 0.4

    # Activity-specific rules for the 101 and 191 sections
    by_name: dict[str, Assignment] = {a.activity.name: a for a in schedule.assignments}

    sla101a = by_name.get("SLA101A")
    sla101b = by_name.get("SLA101B")
    sla191a = by_name.get("SLA191A")
    sla191b = by_name.get("SLA191B")

    # The two SLA101 sections should be spread out
    if sla101a and sla101b:
        diff = abs(TIME_INDEX[sla101a.time] - TIME_INDEX[sla101b.time])
        if diff == 0:
            total -= 0.5
        elif diff >= 4:
            total += 0.5

    # Same idea for the two SLA191 sections
    if sla191a and sla191b:
        diff = abs(TIME_INDEX[sla191a.time] - TIME_INDEX[sla191b.time])
        if diff == 0:
            total -= 0.5
        elif diff >= 4:
            total += 0.5

    # Reward 101/191 pairs that run back-to-back, penalize if they overlap
    for s101 in filter(None, [sla101a, sla101b]):
        for s191 in filter(None, [sla191a, sla191b]):
            diff = abs(TIME_INDEX[s101.time] - TIME_INDEX[s191.time])
            if diff == 0:
                total -= 0.25
            elif diff == 1:
                total += 0.5
                # Penalize if one is in Roman/Beach and the other is not (widely separated buildings)
                if (s101.room in ROMAN_BEACH_SCALAR) != (s191.room in ROMAN_BEACH_SCALAR):
                    total -= 0.4
            elif diff == 2:
                total += 0.25

    # FIX: Facilitator consecutive time slots — apply SAME rules as SLA 101/191:
    #   +0.5 for being in consecutive slots (reward proximity)
    #   -0.4 if one room is Roman/Beach and the other is not (building mismatch penalty)
    # Previously only the -0.4 mismatch was applied; the +0.5 reward was missing.
    fac_rooms: dict[str, dict[str, str]] = defaultdict(dict)
    for a in schedule.assignments:
        fac_rooms[a.facilitator][a.time] = a.room

    for fac in fac_rooms:
        slots = sorted(fac_rooms[fac].keys(), key=lambda t: TIME_INDEX[t])
        for i in range(len(slots) - 1):
            if TIME_INDEX[slots[i + 1]] - TIME_INDEX[slots[i]] == 1:
                total += 0.5  # reward for facilitator in consecutive slots
                room_a = fac_rooms[fac][slots[i]]
                room_b = fac_rooms[fac][slots[i + 1]]
                if (room_a in ROMAN_BEACH_SCALAR) != (room_b in ROMAN_BEACH_SCALAR):
                    total -= 0.4

    schedule.fitness = total
    return total


def score_population(population: list[Schedule]) -> None:
    # Scores every schedule in the population (mutates in place)
    for s in population:
        score_schedule(s)


def population_stats(population: list[Schedule]) -> dict:
    # Returns best, worst, average fitness and references to the best/worst schedules
    scored = [s for s in population if s.fitness is not None]
    if not scored:
        return {}
    fitnesses   = [s.fitness for s in scored]
    best_sched  = max(scored, key=lambda s: s.fitness)
    worst_sched = min(scored, key=lambda s: s.fitness)
    return {
        "best":        best_sched.fitness,
        "worst":       worst_sched.fitness,
        "average":     sum(fitnesses) / len(fitnesses),
        "best_sched":  best_sched,
        "worst_sched": worst_sched,
        "count":       len(scored),
    }


def print_stats(stats: dict, generation: int = 0, prev_best: float = None) -> None:
    # Prints a one-line summary for a generation; shows % improvement if prev_best is given

    # NOTE: % improvement will be blank at generation 0 b/c there is nothing to compare to yet. It will start showing once Part 3's evolution loop runs and passes the previous generation's best fitness into prev_best each iteration

    pct_str = ""
    if prev_best is not None and prev_best != 0:
        improvement = (stats["best"] - prev_best) / abs(prev_best) * 100
        pct_str = f"  |  Improvement: {improvement:+.2f}%"

    print(
        f"  Gen {generation:>4}  |  "
        f"Best: {stats['best']:>8.4f}  |  "
        f"Worst: {stats['worst']:>8.4f}  |  "
        f"Avg: {stats['average']:>8.4f}"
        + pct_str
    )

# End of Part 2


# Start of Part 3: Genetic Algorithm (NumPy-vectorized)


# Index maps and lookup tables (built once at startup)
TIME_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(TIMES)}
ROOM_TO_IDX: dict[str, int] = {r: i for i, r in enumerate(ROOMS)}
FAC_TO_IDX:  dict[str, int] = {f: i for i, f in enumerate(FACILITATORS)}
ACT_NAMES:   list[str]      = [a["name"] for a in ACTIVITIES_DEF]

N_TIMES = len(TIMES)
N_ROOMS = len(ROOMS)
N_FACS  = len(FACILITATORS)
N_ACTS  = len(ACTIVITIES_DEF)

ROOM_CAPACITY       = np.array(list(ROOMS.values()), dtype=np.int64)
ACTIVITY_ENROLLMENT = np.array([a["enrollment"] for a in ACTIVITIES_DEF], dtype=np.int64)

# (11, 10) facilitator score: +0.5 preferred, +0.2 other, -0.1 neither.
# `other` is written first so `preferred` overwrites duplicates (e.g. SLA449 lists
# Zeldin in both) — matches score_schedule's if/elif order.
FAC_SCORE = np.full((N_ACTS, N_FACS), -0.1, dtype=np.float64)
for _a_idx, _act in enumerate(ACTIVITIES_DEF):
    for _f in _act["other"]:
        FAC_SCORE[_a_idx, FAC_TO_IDX[_f]] = 0.2
    for _f in _act["preferred"]:
        FAC_SCORE[_a_idx, FAC_TO_IDX[_f]] = 0.5

ROMAN_BEACH_ROOMS = np.array(
    [name.startswith(("Roman", "Beach")) for name in ROOMS],
    dtype=bool,
)

IDX_101A = ACT_NAMES.index("SLA101A")
IDX_101B = ACT_NAMES.index("SLA101B")
IDX_191A = ACT_NAMES.index("SLA191A")
IDX_191B = ACT_NAMES.index("SLA191B")
TYLER_IDX = FAC_TO_IDX["Tyler"]

PAIR_I, PAIR_J = np.triu_indices(N_ACTS, k=1)   # 55 pairs

_GA_RNG: Generator = Generator(PCG64DXSM())


def random_population_array(n: int, rng: Optional[Generator] = None) -> np.ndarray:
    # Generate n random schedules as a single (n, 11, 3) int array.
    if rng is None:
        rng = _GA_RNG
    pop = np.empty((n, N_ACTS, 3), dtype=np.int64)
    pop[:, :, 0] = rng.integers(0, N_TIMES, size=(n, N_ACTS))
    pop[:, :, 1] = rng.integers(0, N_ROOMS, size=(n, N_ACTS))
    pop[:, :, 2] = rng.integers(0, N_FACS,  size=(n, N_ACTS))
    return pop


def row_to_schedule(row: np.ndarray) -> Schedule:
    # Convert a (11, 3) genome row into a Schedule dataclass (for printing).
    assignments = []
    for a_idx in range(N_ACTS):
        t_idx = int(row[a_idx, 0])
        r_idx = int(row[a_idx, 1])
        f_idx = int(row[a_idx, 2])
        assignments.append(Assignment(
            activity=ACTIVITY_DEFS[a_idx],
            room=ROOM_NAMES[r_idx],
            time=TIMES[t_idx],
            facilitator=FACILITATORS[f_idx],
        ))
    return Schedule(assignments=assignments)


def score_batch(pop: np.ndarray, return_breakdown: bool = False):
    # Vectorized fitness for an (M, 11, 3) population.
    # Mirrors score_schedule exactly (verified by parity_test).
    # If return_breakdown=True, also returns a dict mapping component name to (M,) array.
    M = pop.shape[0]
    times = pop[:, :, 0]
    rooms = pop[:, :, 1]
    facs  = pop[:, :, 2]

    # Room size rubric
    cap = ROOM_CAPACITY[rooms]
    enr = ACTIVITY_ENROLLMENT[None, :]
    room_score = np.where(
        cap < enr, -0.5,
        np.where(cap > 3 * enr, -0.4,
        np.where(cap > 1.5 * enr, -0.2, 0.3))
    )
    room_total = room_score.sum(axis=1)

    # Same-time-same-room collisions (any colliding activity: -0.5)
    same_time = times[:, PAIR_I] == times[:, PAIR_J]
    same_room = rooms[:, PAIR_I] == rooms[:, PAIR_J]
    collides  = same_time & same_room
    counts = np.zeros((M, N_ACTS), dtype=np.int64)
    np.add.at(counts, (slice(None), PAIR_I), collides.astype(np.int64))
    np.add.at(counts, (slice(None), PAIR_J), collides.astype(np.int64))
    collision_total = (-0.5 * (counts > 0).astype(np.float64)).sum(axis=1)

    # Facilitator preference (per-activity lookup)
    act_idx = np.arange(N_ACTS)[None, :]
    fac_pref_total = FAC_SCORE[act_idx, facs].sum(axis=1)

    # Facilitator load per (time, fac)
    ft_counts = np.zeros((M, N_TIMES, N_FACS), dtype=np.int64)
    np.add.at(ft_counts, (np.arange(M)[:, None], times, facs), 1)
    shared = ft_counts[np.arange(M)[:, None], times, facs]
    fac_load_total = np.where(shared == 1, 0.2, -0.2).sum(axis=1)

    # Global facilitator totals (-0.5 if >4; -0.4 if <3 with Tyler exemption for <2)
    f_totals = ft_counts.sum(axis=1)
    over_4 = f_totals > 4
    under_3 = (f_totals > 0) & (f_totals < 3)
    tyler_col = (f_totals[:, TYLER_IDX] >= 2) & (f_totals[:, TYLER_IDX] < 3)
    under_3[:, TYLER_IDX] = tyler_col
    fac_global_total = -0.5 * over_4.sum(axis=1) - 0.4 * under_3.sum(axis=1)

    # SLA 101A vs 101B
    d = np.abs(times[:, IDX_101A] - times[:, IDX_101B])
    sla_101_total = np.where(d == 0, -0.5, 0.0) + np.where(d >= 4, 0.5, 0.0)

    # SLA 191A vs 191B
    d = np.abs(times[:, IDX_191A] - times[:, IDX_191B])
    sla_191_total = np.where(d == 0, -0.5, 0.0) + np.where(d >= 4, 0.5, 0.0)

    # Cross 101 x 191 pairs
    sla_cross_total = np.zeros(M, dtype=np.float64)
    pairs_101 = [IDX_101A, IDX_101A, IDX_101B, IDX_101B]
    pairs_191 = [IDX_191A, IDX_191B, IDX_191A, IDX_191B]
    for a_i, a_j in zip(pairs_101, pairs_191):
        t_a = times[:, a_i]; t_b = times[:, a_j]
        r_a = rooms[:, a_i]; r_b = rooms[:, a_j]
        d = np.abs(t_a - t_b)
        sla_cross_total += np.where(d == 0, -0.25, 0.0)
        sla_cross_total += np.where(d == 1,  0.5,  0.0)
        sla_cross_total += np.where(d == 2,  0.25, 0.0)
        consec = (d == 1)
        rb_a = ROMAN_BEACH_ROOMS[r_a]
        rb_b = ROMAN_BEACH_ROOMS[r_b]
        mismatch = consec & (rb_a != rb_b)
        sla_cross_total += np.where(mismatch, -0.4, 0.0)

    # FIX: Facilitator consecutive slots — apply SAME rules as SLA 101/191:
    #   +0.5 reward for being in consecutive slots
    #   -0.4 if building mismatch (Roman/Beach vs other)
    # Previously only the -0.4 mismatch was applied; the +0.5 reward was missing.
    # Note: last activity in activity-order wins the (time, fac) slot for room lookup.
    fac_room_at_time = np.full((M, N_TIMES, N_FACS), -1, dtype=np.int64)
    row_idx = np.arange(M)
    for act_i in range(N_ACTS):
        fac_room_at_time[row_idx, times[:, act_i], facs[:, act_i]] = rooms[:, act_i]
    occupied = fac_room_at_time >= 0
    fac_consec_total = np.zeros(M, dtype=np.float64)
    for t_slot in range(N_TIMES - 1):
        both = occupied[:, t_slot, :] & occupied[:, t_slot + 1, :]
        # +0.5 reward for each facilitator with consecutive slots
        fac_consec_total += 0.5 * both.sum(axis=1)
        # -0.4 penalty for building mismatch in those consecutive slots
        r_a = np.where(both, fac_room_at_time[:, t_slot,     :], 0)
        r_b = np.where(both, fac_room_at_time[:, t_slot + 1, :], 0)
        rb_a = ROMAN_BEACH_ROOMS[r_a]
        rb_b = ROMAN_BEACH_ROOMS[r_b]
        mismatch = both & (rb_a != rb_b)
        fac_consec_total += -0.4 * mismatch.sum(axis=1)

    fitness = (
        room_total + collision_total + fac_pref_total + fac_load_total
        + fac_global_total + sla_101_total + sla_191_total
        + sla_cross_total + fac_consec_total
    )

    if return_breakdown:
        breakdown = {
            "room size":       room_total,
            "collisions":      collision_total,
            "fac preference":  fac_pref_total,
            "fac load (slot)": fac_load_total,
            "fac totals":      fac_global_total,
            "SLA 101 split":   sla_101_total,
            "SLA 191 split":   sla_191_total,
            "SLA 101×191":     sla_cross_total,
            "fac consecutive": fac_consec_total,
        }
        return fitness, breakdown
    return fitness


def parity_test(n: int = 200, seed: int = 0) -> None:
    # Assert score_batch matches score_schedule on n random schedules.
    rng = Generator(PCG64DXSM(seed=seed))
    pop = random_population_array(n, rng=rng)
    vec_scores = score_batch(pop)
    for i in range(n):
        ref = score_schedule(row_to_schedule(pop[i]))
        if not np.isclose(vec_scores[i], ref, atol=1e-9):
            print(f"Parity mismatch at i={i}: vec={vec_scores[i]} ref={ref}")
            print(f"  genome: {pop[i].tolist()}")
            raise AssertionError("score_batch does not match score_schedule")
    print(f"Parity test passed ({n} schedules).")


def crossover_batch(parents_a: np.ndarray, parents_b: np.ndarray,
                    rng: Generator) -> tuple[np.ndarray, np.ndarray]:
    # Per-field coin-flip crossover. Returns two child arrays, each (P, 11, 3).
    mask = rng.random((parents_a.shape[0], N_ACTS, 3)) < 0.5
    child1 = np.where(mask, parents_a, parents_b).astype(np.int64, copy=False)
    child2 = np.where(mask, parents_b, parents_a).astype(np.int64, copy=False)
    return child1, child2


def mutate_batch(genomes: np.ndarray, rate: float, rng: Generator) -> None:
    # In-place per-gene mutation. Replaces fields with fresh domain draws.
    n = genomes.shape[0]
    mask = rng.random((n, N_ACTS, 3)) < rate
    if not mask.any():
        return
    replacements = np.empty_like(genomes)
    replacements[:, :, 0] = rng.integers(0, N_TIMES, size=(n, N_ACTS))
    replacements[:, :, 1] = rng.integers(0, N_ROOMS, size=(n, N_ACTS))
    replacements[:, :, 2] = rng.integers(0, N_FACS,  size=(n, N_ACTS))
    genomes[mask] = replacements[mask]


def choose_pair_indices(elite_fitness: np.ndarray, n_pairs: int,
                        rng: Generator) -> tuple[np.ndarray, np.ndarray]:
    # Softmax-weighted sampling within the elite block, with replacement across pairs.
    # Max-subtraction is a numerical-stability trick; produces identical probabilities
    # to plain softmax but avoids overflow when fitnesses are large.
    shifted = elite_fitness - elite_fitness.max()
    probs = np.exp(shifted)
    probs /= probs.sum()
    picks = rng.choice(len(elite_fitness), size=n_pairs * 2, replace=True, p=probs)
    a_idx = picks[0::2].copy()
    b_idx = picks[1::2].copy()
    dup = a_idx == b_idx
    while dup.any():
        b_idx[dup] = rng.choice(len(elite_fitness), size=int(dup.sum()), p=probs)
        dup = a_idx == b_idx
    return a_idx, b_idx


def _gen_line(gen: int, best: float, avg: float, worst: float,
              delta_pct: Optional[float], rate: float) -> str:
    d = f"{delta_pct:+7.2f}%" if delta_pct is not None else "   —   "
    return (f"G {gen:>4} | best={best:7.3f} | avg={avg:7.3f} | "
            f"worst={worst:7.3f} | Δavg={d} | λ={rate:.5f}")


def write_best_schedule(row: np.ndarray, fitness: float,
                        path: str = BEST_SCHEDULE_FILE) -> None:
    sched = row_to_schedule(row)
    sched.fitness = fitness
    time_order = {t: i for i, t in enumerate(TIMES)}
    assignments = sorted(sched.assignments,
                         key=lambda a: (time_order[a.time], a.activity.name))
    lines = [
        f"Best schedule — fitness = {fitness:.4f}",
        "",
        f"{'Activity':<10}  {'Time':<6}  {'Room':<12}  {'Facilitator':<12}  {'Enroll':>6}  {'Cap':>4}",
        "-" * 60,
    ]
    prev_time: Optional[str] = None
    for a in assignments:
        if prev_time is not None and a.time != prev_time:
            lines.append("")
        prev_time = a.time
        lines.append(
            f"{a.activity.name:<10}  {a.time:<6}  {a.room:<12}  "
            f"{a.facilitator:<12}  {a.activity.enrollment:>6}  {ROOMS[a.room]:>4}"
        )
    Path(path).write_text("\n".join(lines) + "\n")
    print(f"Wrote best schedule → {path}")


def plot_fitness_cli(history: list[dict]) -> None:
    # Terminal chart of best/avg/worst fitness per generation using plotext.
    try:
        import plotext as plt
    except ImportError:
        print("note: plotext not installed — skipping CLI chart. "
              "Install with: pip install plotext")
        return
    gens  = [h["gen"]   for h in history]
    best  = [h["best"]  for h in history]
    avg   = [h["avg"]   for h in history]
    worst = [h["worst"] for h in history]
    plt.clear_figure()
    plt.theme("clear")
    plt.plot(gens, best,  label="best",  color="cyan")
    plt.plot(gens, avg,   label="avg",   color="green")
    plt.plot(gens, worst, label="worst", color="red")
    plt.title("Fitness over generations")
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.plotsize(100, 25)
    plt.show()


def write_history_csv(history: list[dict], path: str = FITNESS_HISTORY_CSV) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gen", "best", "avg", "worst", "delta_pct", "lambda"])
        for h in history:
            w.writerow([
                h["gen"], h["best"], h["avg"], h["worst"],
                "" if h["delta_pct"] is None else h["delta_pct"],
                h["rate"],
            ])
    print(f"Wrote fitness history → {path}")


class Visualizer:
    # Live 6-panel matplotlib view that refreshes each generation.
    # Lazily imports matplotlib so the module can run headless without it.
    def __init__(self) -> None:
        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise RuntimeError(
                "matplotlib is required for --viz. Install with: pip install matplotlib"
            ) from e
        self._plt = plt
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 3, figsize=(16, 9))
        self.fig.suptitle("GA live visualization", fontsize=14)
        try:
            self.fig.canvas.manager.set_window_title("GA Live")
        except Exception:
            pass
        plt.show(block=False)

    def on_gen(self, history: list[dict], pop: np.ndarray, fitness: np.ndarray,
               rate: float, halving_gens: list[int]) -> None:
        axes = self.axes
        best_i = int(np.argmax(fitness))
        best_row = pop[best_i]
        gens = [h["gen"] for h in history]

        # (0,0) fitness over generations + halving markers
        ax = axes[0, 0]; ax.clear()
        ax.plot(gens, [h["best"]  for h in history], label="best",  color="#1f77b4")
        ax.plot(gens, [h["avg"]   for h in history], label="avg",   color="#2ca02c")
        ax.plot(gens, [h["worst"] for h in history], label="worst", color="#d62728")
        for g in halving_gens:
            ax.axvline(g, color="orange", linestyle="--", alpha=0.6, linewidth=1)
        ax.set_title("Fitness over generations")
        ax.set_xlabel("generation"); ax.set_ylabel("fitness")
        ax.legend(loc="lower right"); ax.grid(alpha=0.3)

        # (0,1) lambda schedule
        ax = axes[0, 1]; ax.clear()
        rates = [h["rate"] for h in history]
        ax.step(gens, rates, where="post", color="orange")
        ax.set_yscale("log")
        ax.set_title("Mutation rate λ"); ax.set_xlabel("generation")
        ax.set_ylabel("λ (log)"); ax.grid(alpha=0.3)

        # (0,2) constraint breakdown of current best
        ax = axes[0, 2]; ax.clear()
        _, breakdown = score_batch(best_row[None, :, :], return_breakdown=True)
        labels = list(breakdown.keys())
        values = [float(breakdown[k][0]) for k in labels]
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in values]
        ax.barh(labels, values, color=colors)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title(f"Best breakdown (total = {float(fitness[best_i]):.2f})")
        ax.set_xlabel("points"); ax.invert_yaxis()

        # (1,0) facilitator load of current best
        ax = axes[1, 0]; ax.clear()
        counts = np.bincount(best_row[:, 2], minlength=N_FACS)
        bar_colors = []
        for f_i, c in enumerate(counts):
            if c > 4:
                bar_colors.append("#d62728")
            elif c < 3 and not (f_i == TYLER_IDX and c < 2):
                bar_colors.append("#ff7f0e")
            else:
                bar_colors.append("#2ca02c")
        ax.bar(FACILITATORS, counts, color=bar_colors)
        ax.axhline(3, color="gray", linestyle=":", alpha=0.6)
        ax.axhline(4, color="gray", linestyle=":", alpha=0.6)
        ax.set_title("Facilitator load (best)"); ax.set_ylabel("# activities")
        ax.tick_params(axis="x", rotation=30)

        # (1,1) room utilization of current best
        ax = axes[1, 1]; ax.clear()
        slot_counts = np.bincount(best_row[:, 1], minlength=N_ROOMS)
        ax.bar(ROOM_NAMES, slot_counts, color="#1f77b4")
        ax.set_title("Room utilization (best)"); ax.set_ylabel("activities scheduled")
        ax.tick_params(axis="x", rotation=45)

        # (1,2) time x room heatmap — visualizes same-time-same-room collisions
        ax = axes[1, 2]; ax.clear()
        heat = np.zeros((N_TIMES, N_ROOMS), dtype=int)
        for t, r in zip(best_row[:, 0], best_row[:, 1]):
            heat[t, r] += 1
        ax.imshow(heat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(2, heat.max()))
        ax.set_xticks(range(N_ROOMS)); ax.set_xticklabels(ROOM_NAMES, rotation=30, ha="right")
        ax.set_yticks(range(N_TIMES)); ax.set_yticklabels(TIMES)
        ax.set_title("Time × room (best)")
        for (y, x), v in np.ndenumerate(heat):
            if v > 0:
                ax.text(x, y, str(v), ha="center", va="center",
                        color="black" if v < 2 else "white", fontsize=8)

        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        self._plt.pause(0.001)

    def close(self, block: bool = True) -> None:
        self._plt.ioff()
        if block:
            self._plt.show()


def run_ga(pop_size: int = POPULATION_SIZE,
           elite_count: int = ELITE_COUNT,
           pair_count: int  = PAIR_COUNT,
           initial_rate: float = MUTATION_RATE,
           min_generations: int = MIN_GENERATIONS,
           max_generations: Optional[int] = None,
           seed: Optional[int] = None,
           verbose: bool = True,
           visualizer: Optional[Visualizer] = None,
           write_outputs: bool = True) -> dict:
    # Main GA loop. Returns the final history snapshot (best-so-far at termination).
    rng = Generator(PCG64DXSM(seed=seed)) if seed is not None else _GA_RNG

    def say(msg: str) -> None:
        if verbose:
            print(msg)

    if verbose:
        say("Parity test: score_batch vs score_schedule …")
        parity_test(200)

    say(f"Generating initial population of {pop_size} …")
    pop = random_population_array(pop_size, rng)
    fitness = score_batch(pop)

    rate = initial_rate
    history: list[dict] = []
    halving_gens: list[int] = []
    halvings = 0
    last_halve_gen = 0
    streak = 0

    def snapshot(gen: int, delta_pct: Optional[float]) -> None:
        best_i = int(np.argmax(fitness))
        stats = {
            "gen": gen,
            "best":  float(fitness.max()),
            "avg":   float(fitness.mean()),
            "worst": float(fitness.min()),
            "delta_pct": delta_pct,
            "rate": rate,
            "best_row": pop[best_i].copy(),
        }
        history.append(stats)
        if verbose:
            print(_gen_line(gen, stats["best"], stats["avg"], stats["worst"], delta_pct, rate))

    snapshot(0, None)
    if visualizer is not None:
        visualizer.on_gen(history, pop, fitness, rate, halving_gens)

    for gen in itertools.count(1):
        elite_idx = np.argpartition(fitness, -elite_count)[-elite_count:]
        elite_pop = pop[elite_idx]
        elite_fit = fitness[elite_idx]

        a_idx, b_idx = choose_pair_indices(elite_fit, pair_count, rng)
        c1, c2 = crossover_batch(elite_pop[a_idx], elite_pop[b_idx], rng)
        offspring = np.concatenate([c1, c2], axis=0)
        mutate_batch(offspring, rate, rng)
        off_fit = score_batch(offspring)

        pop     = np.concatenate([elite_pop, offspring], axis=0)
        fitness = np.concatenate([elite_fit,  off_fit],  axis=0)

        prev_avg = history[-1]["avg"]
        denom = abs(prev_avg) if prev_avg != 0 else 1.0
        delta_pct = 100.0 * (fitness.mean() - prev_avg) / denom
        snapshot(gen, delta_pct)

        assert pop.shape == (pop_size, N_ACTS, 3), f"bad pop shape {pop.shape}"
        assert fitness.shape == (pop_size,), f"bad fitness shape {fitness.shape}"
        assert not np.isnan(fitness).any(), "fitness contains NaN"

        if visualizer is not None:
            visualizer.on_gen(history, pop, fitness, rate, halving_gens)

        if max_generations is not None and gen >= max_generations:
            say(f"--- hit max_generations={max_generations}, stopping ---")
            break

        if gen < min_generations:
            continue

        if delta_pct < IMPROVEMENT_THRESHOLD * 100:
            streak += 1
        else:
            streak = 0

        if streak >= PLATEAU_DEBOUNCE and (gen - last_halve_gen) >= HALVING_COOLDOWN:
            if halvings < MAX_HALVINGS:
                rate /= 2.0
                halvings += 1
                last_halve_gen = gen
                halving_gens.append(gen)
                streak = 0
                say(f"--- plateau detected: halving λ → {rate:.6f} ---")
            else:
                say(f"--- termination: avg improvement < 1% after {MAX_HALVINGS} halvings ---")
                break

    best = history[-1]

    if verbose:
        sep = "=" * 72
        print()
        print(sep)
        print(f"Final generation: {best['gen']}")
        print(f"Best fitness:     {best['best']:.4f}")
        print(f"Avg fitness:      {best['avg']:.4f}")
        print(f"Worst fitness:    {best['worst']:.4f}")
        print(f"Final λ:          {rate:.6f}")
        print(sep)

    if write_outputs:
        write_best_schedule(best["best_row"], best["best"])
        write_history_csv(history)
        if verbose:
            final_sched = row_to_schedule(best["best_row"])
            final_sched.fitness = best["best"]
            print("\nBest schedule (sorted by time):")
            print_schedule(final_sched, sort_by="time")
            if visualizer is None:
                plot_fitness_cli(history)

    best["final_rate"] = rate
    best["halving_gens"] = halving_gens
    best["history"] = history
    return best


def run_many(n_runs: int,
             max_generations: Optional[int] = None,
             base_seed: int = 0) -> dict:
    # Run the GA n_runs times with distinct seeds; report the overall best.
    import time
    print(f"Headless batch: {n_runs} independent GA runs (seeds {base_seed}..{base_seed + n_runs - 1})")
    print("-" * 72)
    bests: list[float] = []
    gens:  list[int]   = []
    times: list[float] = []
    overall_best: Optional[dict] = None
    overall_seed: Optional[int] = None
    batch_start = time.time()
    for i in range(n_runs):
        seed = base_seed + i
        t0 = time.time()
        result = run_ga(
            seed=seed,
            verbose=False,
            max_generations=max_generations,
            write_outputs=False,
        )
        elapsed = time.time() - t0
        bests.append(result["best"])
        gens.append(result["gen"])
        times.append(elapsed)
        flag = ""
        if overall_best is None or result["best"] > overall_best["best"]:
            overall_best = result
            overall_seed = seed
            flag = "  ← new best"
        print(f"  Run {i+1:>3}/{n_runs} | seed={seed:>4} | gen={result['gen']:>4} | "
              f"best={result['best']:7.3f} | time={elapsed:5.1f}s{flag}")
    total = time.time() - batch_start
    bests_arr = np.array(bests)
    gens_arr  = np.array(gens)

    sep = "=" * 72
    print()
    print(sep)
    print(f"Batch summary  ({n_runs} runs, total wall time {total:.1f}s)")
    print(sep)
    print(f"  Best fitness:    {bests_arr.max():.4f}  (seed={overall_seed})")
    print(f"  Mean fitness:    {bests_arr.mean():.4f}")
    print(f"  Median fitness:  {float(np.median(bests_arr)):.4f}")
    print(f"  Worst fitness:   {bests_arr.min():.4f}")
    print(f"  Std dev:         {bests_arr.std():.4f}")
    print(f"  Mean gens to stop: {gens_arr.mean():.1f}  (min {gens_arr.min()}, max {gens_arr.max()})")
    print(sep)

    # Write the overall winner separately and a CSV of per-run stats.
    write_best_schedule(overall_best["best_row"], overall_best["best"], path="best_of_runs.txt")
    with open("runs_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "seed", "gen", "best", "time_s"])
        for i, (b, g, t) in enumerate(zip(bests, gens, times)):
            w.writerow([i + 1, base_seed + i, g, b, t])
    print("Wrote runs_summary.csv")

    final_sched = row_to_schedule(overall_best["best_row"])
    final_sched.fitness = overall_best["best"]
    print("\nOverall best schedule (sorted by time):")
    print_schedule(final_sched, sort_by="time")
    return overall_best


# End of Part 3


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="CS461 Program 2 — Genetic Algorithm scheduler")
    parser.add_argument("--viz", action="store_true",
                        help="show live matplotlib visualization (single-run only)")
    parser.add_argument("--runs", type=int, default=1, metavar="N",
                        help="number of independent GA runs (headless batch mode)")
    parser.add_argument("--max-gen", type=int, default=None, metavar="G",
                        help="cap the number of generations (useful for testing)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for single-run reproducibility; base seed for --runs")
    args = parser.parse_args()

    if args.runs > 1:
        if args.viz:
            print("note: --viz is ignored in batch mode (--runs > 1).")
        run_many(args.runs, max_generations=args.max_gen, base_seed=args.seed or 0)
        return

    viz: Optional[Visualizer] = None
    if args.viz:
        viz = Visualizer()
    try:
        run_ga(max_generations=args.max_gen, seed=args.seed, visualizer=viz)
    finally:
        if viz is not None:
            viz.close(block=True)


if __name__ == "__main__":
    main()