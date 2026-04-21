# Comprehensive GA Plan — v2 (NumPy-vectorized)

## 1. Core data representation

Everything hot-path-related is NumPy arrays. Dataclasses survive only at the boundary (pretty-printing).

```python
# Index maps built once from the existing constants
TIME_TO_IDX = {t:i for i,t in enumerate(TIMES)}            # 6
ROOM_TO_IDX = {r:i for i,r in enumerate(ROOMS)}            # 9
FAC_TO_IDX  = {f:i for i,f in enumerate(FACILITATORS)}     # 10
ACT_NAMES   = [a["name"] for a in ACTIVITIES_DEF]          # 11, fixed order
```

**Genome**: `(11, 3)` int array per schedule — columns are `[time_idx, room_idx, facilitator_idx]`, rows are activities in `ACTIVITIES_DEF` order.

**Population**: one `(N, 11, 3)` int array, `pop`.

**Fitness**: `(N,)` float array, `fitness`, aligned 1:1 with `pop` rows. This is the cache — NaN means "not yet scored." After `score_population`, no NaNs remain.

## 2. Precomputed lookup tables (built once at startup)

```python
ROOM_CAPACITY      = np.array([c for c in ROOMS.values()])        # (9,)
ACTIVITY_ENROLLMENT = np.array([a["enrollment"] for a in ACTIVITIES_DEF])  # (11,)

# (11, 10) float table: +0.5 preferred, +0.2 other, -0.1 neither
FAC_SCORE = np.full((11, 10), -0.1)
for a_idx, act in enumerate(ACTIVITIES_DEF):
    for f in act["preferred"]:  FAC_SCORE[a_idx, FAC_TO_IDX[f]] = 0.5
    for f in act["other"]:      FAC_SCORE[a_idx, FAC_TO_IDX[f]] = 0.2

# Building membership mask for the Roman/Beach rule
ROMAN_BEACH_ROOMS = np.array([r.startswith(("Roman","Beach")) for r in ROOMS], bool)  # (9,)

# Fixed activity indices for special rules
IDX_101A, IDX_101B = ACT_NAMES.index("SLA101A"), ACT_NAMES.index("SLA101B")
IDX_191A, IDX_191B = ACT_NAMES.index("SLA191A"), ACT_NAMES.index("SLA191B")
TYLER_IDX = FAC_TO_IDX["Tyler"]

# Pairwise activity indices for collision checks
PAIR_I, PAIR_J = np.triu_indices(11, k=1)   # (55,), (55,)
```

## 3. Vectorized scoring

`score_batch(pop_slice) -> np.ndarray` returns a `(M,)` fitness array. Works on any `(M, 11, 3)` chunk (initial pop of 1000, or an offspring batch of 500). Produces **identical results** to the existing `score_schedule`; a parity test will assert this (see §9).

Key pieces — all operate on whole batches at once:

**Room-size rubric**
```python
times = pop[:, :, 0]            # (M, 11)
rooms = pop[:, :, 1]
facs  = pop[:, :, 2]
cap   = ROOM_CAPACITY[rooms]    # (M, 11)
enr   = ACTIVITY_ENROLLMENT[None, :]   # (1, 11)
room_score = np.where(cap < enr,           -0.5,
             np.where(cap > 3*enr,         -0.4,
             np.where(cap > 1.5*enr,       -0.2,
                                            0.3)))
# sum over activities later
```

**Facilitator preference**: `fac_pref_score = FAC_SCORE[np.arange(11)[None,:], facs]` → `(M, 11)`.

**Same-time-same-room collisions** (pair-based, vectorized over M):
```python
same_time = times[:, PAIR_I] == times[:, PAIR_J]   # (M, 55)
same_room = rooms[:, PAIR_I] == rooms[:, PAIR_J]
collides  = same_time & same_room                   # (M, 55)
# each offending activity gets -0.5; a collision flags both activities
collision_flags = np.zeros((M, 11), bool)
np.logical_or.at(collision_flags, (slice(None), PAIR_I), collides)
np.logical_or.at(collision_flags, (slice(None), PAIR_J), collides)
room_conflict_score = -0.5 * collision_flags
```

**Facilitator load**
```python
# per-schedule (time, facilitator) counts
ft_counts = np.zeros((M, 6, 10), int)
np.add.at(ft_counts, (np.arange(M)[:,None], times, facs), 1)

# per-schedule total counts per facilitator
f_totals = ft_counts.sum(axis=1)                    # (M, 10)

# for each activity: how many activities share its (time, facilitator)?
shared = ft_counts[np.arange(M)[:,None], times, facs]   # (M, 11)
fac_load_score = np.where(shared == 1,  0.2, -0.2)

# per-facilitator global penalties (applied once per schedule per facilitator)
over_4 = (f_totals > 4)                             # (M, 10)
under_3_mask = np.zeros_like(over_4)
under_3_mask[:, :] = f_totals < 3
under_3_mask[:, TYLER_IDX] = f_totals[:, TYLER_IDX] < 2   # Tyler exception
global_fac_score = -0.5 * over_4.sum(axis=1) - 0.4 * under_3_mask.sum(axis=1)
```

**SLA 101/191 special rules** — direct index arithmetic on the 4 time columns; produces a `(M,)` schedule-level adjustment.

**Facilitator consecutive slots** — cleanest way: for each schedule, build the `(6, 10)` occupancy matrix `(ft_counts > 0)`, then check adjacent rows with the Roman/Beach mismatch rule. All vectorizable.

**Sum**
```python
per_activity = room_score + fac_pref_score + room_conflict_score + fac_load_score
fitness = per_activity.sum(axis=1) + global_fac_score + sla_adjust + consec_adjust
```

## 4. Vectorized crossover + mutation

```python
def crossover_batch(parents_a, parents_b, rng):
    # parents_*: (P, 11, 3)
    mask = rng.random((len(parents_a), 11, 3)) < 0.5
    child1 = np.where(mask, parents_a, parents_b)
    child2 = np.where(mask, parents_b, parents_a)
    return child1, child2

def mutate_batch(genomes, rate, rng):
    # in-place on genomes; returns nothing
    n = len(genomes)
    mask = rng.random((n, 11, 3)) < rate
    if not mask.any(): return
    # generate replacement draws for each field independently
    new_vals = np.empty_like(genomes)
    new_vals[:, :, 0] = rng.integers(0, 6,  size=(n, 11))
    new_vals[:, :, 1] = rng.integers(0, 9,  size=(n, 11))
    new_vals[:, :, 2] = rng.integers(0, 10, size=(n, 11))
    genomes[mask] = new_vals[mask]
```

Offspring count: 250 pairs × 2 children = 500 genomes per gen. One call each to `crossover_batch` and `mutate_batch`.

## 5. Elite pass-through (improvement #4)

`next_generation` **never copies or mutates elites**. It produces the next `(N, 11, 3)` population by:

1. `np.argpartition(fitness, -500)[-500:]` → elite row indices (O(N), no full sort).
2. Build offspring as a new `(500, 11, 3)` array (crossover + mutate).
3. Score the offspring → `(500,)` fitness array.
4. `np.concatenate([pop[elite_idx], offspring], axis=0)` → new pop `(1000, 11, 3)`.
5. `np.concatenate([fitness[elite_idx], offspring_fitness])` → new fitness array.

Elite rows are copied exactly once (the concat), not mutated. Their fitness values are reused, never recomputed. **Invariant**: `np.isnan(fitness).sum() == 0` after every generation; checked in an assert.

## 6. Selection — softmax within elites

```python
def choose_pair_indices(elite_fitness, n_pairs, rng):
    # softmax with max-subtract for stability
    f = elite_fitness - elite_fitness.max()
    probs = np.exp(f); probs /= probs.sum()
    picks = rng.choice(len(elite_fitness), size=n_pairs*2, replace=True, p=probs)
    # redraw any pair where both parents are the same row
    a_idx = picks[0::2]; b_idx = picks[1::2]
    dup = a_idx == b_idx
    while dup.any():
        b_idx[dup] = rng.choice(len(elite_fitness), size=dup.sum(), p=probs)
        dup = a_idx == b_idx
    return a_idx, b_idx
```

Returns two index arrays into the 500-row elite block; use them to gather parents in one shot.

## 7. Driver loop

```python
def run_ga():
    rng = np.random.default_rng(np.random.PCG64DXSM())
    pop = random_population(POPULATION_SIZE, rng)         # (1000, 11, 3)
    fitness = score_batch(pop)                            # (1000,)

    rate = MUTATION_RATE
    history = []
    halvings = 0; last_halve_gen = 0; streak = 0

    def snapshot(gen, delta_pct):
        stats = dict(gen=gen, best=fitness.max(), avg=fitness.mean(),
                     worst=fitness.min(), delta_pct=delta_pct, rate=rate,
                     best_row=pop[np.argmax(fitness)].copy())
        history.append(stats); print_gen_line(stats)

    snapshot(0, None)

    for gen in itertools.count(1):
        # -- build next generation --
        elite_idx = np.argpartition(fitness, -ELITE_COUNT)[-ELITE_COUNT:]
        elite_pop, elite_fit = pop[elite_idx], fitness[elite_idx]

        a, b = choose_pair_indices(elite_fit, PAIR_COUNT, rng)
        c1, c2 = crossover_batch(elite_pop[a], elite_pop[b], rng)
        offspring = np.concatenate([c1, c2], axis=0)      # (500, 11, 3)
        mutate_batch(offspring, rate, rng)
        off_fit = score_batch(offspring)                  # only 500 scored

        pop     = np.concatenate([elite_pop, offspring], axis=0)
        fitness = np.concatenate([elite_fit,  off_fit], axis=0)

        delta_pct = 100.0 * (fitness.mean() - history[-1]["avg"]) / abs(history[-1]["avg"] or 1)
        snapshot(gen, delta_pct)

        # -- termination + λ halving --
        if gen < MIN_GENERATIONS: continue
        streak = streak + 1 if delta_pct < 1.0 else 0
        if streak >= PLATEAU_DEBOUNCE and gen - last_halve_gen >= HALVING_COOLDOWN:
            if halvings < MAX_HALVINGS:
                rate /= 2; halvings += 1
                last_halve_gen = gen; streak = 0
                print(f"--- plateau: halving λ → {rate:.5f} ---")
            else:
                print("--- termination condition met ---"); break

    write_best_schedule(history[-1]["best_row"], history[-1]["best"])
    write_history_csv(history)
    plot_history(history)
```

## 8. Boundary: arrays ↔ Schedule (printing only)

One helper, `row_to_schedule(genome_row) -> Schedule`, materializes a dataclass `Schedule` for `print_schedule` / file output. Only called at the end of the run and optionally on best-of-generation for logging. Hot path never touches dataclasses.

## 9. Parity test (important — don't skip)

Before wiring the new scorer into the driver, generate ~500 random schedules and assert:

```python
assert np.allclose(score_batch(arr_form), [score_schedule(sched_form(r)) for r in arr_form])
```

This catches sign errors, off-by-ones, and the facilitator-consecutive-slot ambiguity. Do not proceed to §7 until parity holds. After parity is green, the old `score_schedule` can be kept for reference or removed.

## 10. Per-gen output + final artifacts

- Per-gen line: `G <n> | best= … | avg= … | worst= … | Δavg= …% | λ= …`.
- `best_schedule.txt` — final best, sorted by time, with fitness on the first line.
- `fitness_history.csv` — `gen, best, avg, worst, delta_pct, rate`.
- `plotext` line chart (best/avg/worst).

## 11. Invariants to assert in the driver

- `pop.shape == (POPULATION_SIZE, 11, 3)` and dtype int after every generation.
- `fitness.shape == (POPULATION_SIZE,)` and contains no NaN.
- `score_batch(pop[elite_idx])` equals the cached `fitness[elite_idx]` on a spot check (catches accidental elite mutation).

## 12. Order of implementation

1. Build constants + lookup tables (§2).
2. `random_population` (batched `rng.integers`) + `row_to_schedule` printer helper.
3. `score_batch` — **implement + parity-test against existing `score_schedule` (§9)**.
4. `crossover_batch`, `mutate_batch`, `choose_pair_indices`.
5. Driver (§7) + per-gen printer.
6. File outputs, CSV, plotext chart.
7. Delete or flag-gate the old `score_schedule` / Schedule-based population code.

## Confirmed parameters

| Param | Value | Source |
|---|---|---|
| `POPULATION_SIZE` | 1000 | user; spec requires ≥ 250 |
| `ELITE_COUNT` | 500 | user (top half kept) |
| `PAIR_COUNT` | 250 | user (2 children per pair → 500 offspring) |
| `MUTATION_RATE` (λ) | 0.01 per gene | spec |
| Selection | softmax within top 500 | spec + user (option B) |
| Crossover | per-field independent mix | user |
| Mutation target | offspring only | user + spec |
| `MIN_GENERATIONS` | 100 | spec |
| Termination | avg-fitness gen-to-gen improvement < 1% | spec |
| λ halving | on plateau, up to 4 times, 20-gen cooldown | spec + user |
