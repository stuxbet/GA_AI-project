#CS461-001 Program 2
#Contributers - Dylan G, Sheyla R, YOUR NAME HERE
import sys
import random
from dataclasses import dataclass, field
from typing import Optional
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

ROOM_NAMES:       list[str] = list(ROOMS.keys()) #List of room names
MUTATION_RATE:    float     = 0.01                                                  #NOT USED CAN BE REMOVED OR CHANGED AS NEEDED IF YOU IMPLEMENT A BETTER MR
POPULATION_SIZE:  int       = 250  #Can be changed to whatever number we need


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

    print(sep) #USED TO TEST THAT THE PRINT FUNCTION IS WORKING CORRECTLY, YOU CAN REMOVE THIS SECTION AFTER YOU IMPLEMENT PART 3
    if schedule.fitness is not None:
        print(f"  Fitness: {schedule.fitness:.4f}") 
    else:
        print("  Fitness: (not yet evaluated)")
    print()
    # END OF PRINT TEST DELETE ABOVE AFTER PART 3 ^


# Start of Part 2


# Map each time slot to an index so we can do distance comparisons easily
TIME_INDEX: dict[str, int] = {t: i for i, t in enumerate(TIMES)}


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
            total -= 0.2
        elif cap > 1.5 * enroll:
            total -= 0.1
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

        # Facilitator scheduled for more than one activity in the same slot
        if fac_at_time[a.facilitator][a.time] > 1:
            total -= 0.2

    # Reward or penalize based on how many total activities a facilitator teaches
    for fac, count in fac_total.items():
        if count > 4:
            total -= 0.5
        elif count in (1, 2):
            total += 0.2

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


# 5. MAIN




#This section is also just to test that the schedules were created correctly, you can implement this in part 3 if you want or just delete it and redo the printing function for part 3
def main() -> None:
    print(f"Generating population of {POPULATION_SIZE} random schedules …", end=" ")
    population = generate_population(POPULATION_SIZE)
    print("done.")
    print(f"  Total schedules created : {len(population)}")
    print(f"  Activities per schedule : {len(population[0].assignments)}")
    print(f"  Mutation rate (λ)       : {MUTATION_RATE}")

    # Part 2: score all schedules and show stats
    score_population(population)

    # Print the first schedule sorted two ways for demonstration
    sample = population[0]
    print_schedule(sample, sort_by="time")
    print_schedule(sample, sort_by="activity")
    stats = population_stats(population)
    sep = "-" * 72
    print(f"\n{sep}")
    print("  Population Statistics  (Generation 0)")
    print(sep)
    print_stats(stats, generation=0)
    print(sep)
    print("\nBest schedule in initial population:")
    print_schedule(stats["best_sched"], sort_by="time")
    print("Worst schedule in initial population:")
    print_schedule(stats["worst_sched"], sort_by="time")




#Don't delete this
#Call this to generate the scheduled and do anything else that is in part 1
if __name__ == "__main__":
    main()
#End of Part 1
