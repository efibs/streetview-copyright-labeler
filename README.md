# CR Labeler

Reads the **copyright year printed into Google Street View imagery** and tags locations with it.

> ### About this project
> **The code in this repository was written by an AI** (Claude), working from a brief and iterating
> against measurements. **The ground truth was labelled by a human** — 543 panoramas read by eye,
> plus the original 20-panorama seed set. That division matters when reading the accuracy numbers
> below: the program was tuned by its author, but it was *scored* against labels it never saw.

---

## What it does

Every Street View panorama carries a faint `© YYYY Google` stamp burned into the imagery. That year
is **not** the capture date — it is when Google last processed the imagery. A panorama photographed
in 2012 can carry a 2026 copyright. No API exposes it. The only source of truth is the pixels.

```bash
cr-label label my-map.json
```

Input and output are [map-making.app](https://map-making.app/) / GeoGuessr map JSON. Each location
gains one tag — `CR_2024`, `CR_None` (no watermark), or `CR_unknown` (found one, couldn't read it) —
and everything else in the file is preserved untouched.

## Results

Measured against 543 hand-labelled panoramas the program never saw during development:

| | Unbiased sample (n=203) | Whole labelled set (n=543) |
|---|---|---|
| Precision — of the answers it gives | **100.0%** | **99.4%** |
| Coverage — how often it answers at all | **100.0%** | 99.8% |
| **Wrong years** | **0** | **0** |

Of 455 year answers, **all 455 were exactly right**. Every year from 2009 to 2026 scored 100%.

Most of the labelled set was chosen deliberately to be hard, so the unbiased column is the one that
describes normal use. The three errors across the whole set are all at the *has a watermark / has
none* boundary, never a wrong year.

---

## How it works

There is **no neural network and nothing is trained.** The watermark is a fixed, machine-rendered
piece of text, so this is a signal-processing problem rather than a learning one.

### The problem

A single stamp is only a few grey levels above the noise. Here is one, at real brightness:

![A watermark before and after the high-pass](docs/01-highpass.png)

The top strip is the panorama as it comes. The bottom is the same pixels after subtracting a blurred
copy of themselves — a **high-pass filter**. The sky's smooth gradient disappears; the thin strokes
of the text survive, because they are the only sharp thing there.

### Step 1 — find every copy

The stamp is not printed once. It is scattered across the whole panorama, dozens of times, and about
96% of them land in the upper half:

![Detected watermark instances across a panorama](docs/03-instances.png)

Each red box is one detection. They are found by **matched filtering** — sliding a picture of the
`Google` wordmark across the image and looking for peaks. The wordmark is used rather than the whole
string because it is the part that never changes, whatever the year.

### Step 2 — throw out the false ones

Matched filtering finds every real stamp but also fires on tree branches and JPEG noise. The filter
is left deliberately permissive, and the false hits are removed afterwards by **consensus**:

> Every real stamp is *the same bitmap*, so they all look like one another.
> A false hit looks like nothing else in the set.

Keeping only the largest mutually-agreeing group discards the contamination without needing a
delicately tuned threshold. On a panorama with no watermark at all no such group forms — which is
exactly the answer wanted there.

### Step 3 — average them

This is what makes the whole thing work:

![One instance versus twenty-two averaged](docs/02-averaging.png)

Averaging N aligned copies suppresses noise by roughly √N while the watermark, identical in every
copy, reinforces. One instance is barely legible. Twenty-two of them are unambiguous.

### Step 4 — read the year

The averaged image is compared against a **bank of reference images** — real averaged watermarks, one
per (rendering, year). Each of the four digit positions is scored in its own slot, so every digit
counts equally, and the answer is the year that matches best.

Then three checks decide whether to trust it, because **declining to answer beats answering wrongly**:

| Check | Rejects |
|---|---|
| Does this look like a watermark at all? | panoramas with none → `CR_None` |
| Does *every* digit match, not just on average? | years the bank has never seen |
| Does the winner beat the runner-up? | genuine ties |

If a reading is weak it escalates, each step kept only if it scores better than what came before:
zoom 3, then the full sphere, then zoom 4 — which covers four times the area at the same glyph size
and typically triples the number of instances found. "Weak" includes two readings that *look*
answered: one built from a single instance (nothing agreed with it, so consensus never checked it)
and one whose winning year barely beat the runner-up.

### Why a bank of real images, and not rendered text

Synthetic text in the same font scores **10/20** against ground truth. Google's rasterisation has a
halo and sub-pixel placement that a font render does not reproduce. The same classifier using real
averaged composites scores **20/20**. Synthetic text is used once, only to bootstrap a bank from
nothing — it is good enough to *find* the stamps, and the data takes over from there.

---

## Install

```bash
conda create -n cr_labeler python=3.12 numpy pillow requests tqdm pytest
conda activate cr_labeler
pip install -e .
```

Only `numpy`, `pillow`, `requests` and `tqdm`. No OpenCV. Tile fetching needs no API key.

**Optional GPU.** If PyTorch with a working CUDA device happens to be installed, the correlation runs
there instead — about 1.4x end to end, and nothing about it is required. There is no GPU-only code
path: without torch, without a driver, or with `CR_LABELER_DEVICE=cpu`, everything runs on numpy and
produces the same labels (verified identical across all 543 hand-labelled panoramas). To try it:

```bash
pip install torch            # optional, ~2.5 GB
CR_LABELER_DEVICE=cpu ...    # force the numpy path back on at any time
```

## Quick start

```bash
cr-label label CR_GT.json                                   # tag a map
cr-label label in.json -o out.json --report r.csv --cache cache
cr-label evaluate --gt CR_GT.json                           # score against GT_* tags
```

A long run reports progress and how much of it is left:

```
  47%|████▋     | 47562/101233 [1:12:04<1:21:19, 11.0pano/s, unknown 12, errors 0]
```

The two counts are the ones worth abandoning a run over — the full breakdown by year is
printed at the end. `--quiet` turns the bar off.

| Option | Default | Meaning |
|---|---|---|
| `--report FILE.csv` | off | Per-panorama scores: confidence, instances, style, margins |
| `--cache DIR` | off | Tile cache; makes repeat runs nearly free |
| `--workers N` | 24, or 8 with a GPU | Panoramas in parallel |
| `--zoom N` | 2 | Where the ladder starts; weak reads fall through to 3 and 4 anyway |
| `--rows top\|all` | `top` | `top` reads the upper hemisphere: ~96% of stamps, half the bytes |
| `--no-escalate` | off | Skip retries — much faster, at real cost to coverage |
| `--save-composites DIR` | off | Write each averaged watermark as a PNG, to check reads by eye |
| `--quiet` | off | No progress bar |
| `--api-key-file PATH` | — | Only for rows with no `panoId`; see below |

### Throughput

Measured on modern Gen 4 coverage, cold cache, live network, this machine (Ryzen 9 7950X3D, 32
threads, RTX A5000). Live-network runs vary by roughly ±20%, so these are medians of several
300-panorama batches:

| | pano/s | 101,233 panoramas |
|---|---|---|
| no GPU | **~7.4** | ~3.8 hours |
| with GPU | **~11** | ~2.6 hours |
| tiles already cached | ~13 | ~2.2 hours |

**What actually costs time is tiles, not maths.** Fetching alone caps throughput at 10.8 panoramas/s
at zoom 3 — so once correlation moved to the GPU, the network became the wall. That is why the ladder
starts at zoom 2: four tiles instead of sixteen. About 65% of Gen 4 panoramas settle there and the
rest fall through to zoom 3 unchanged, for an effective 9.7 tiles per panorama against 17.6.

Three changes got it from 5.8 to ~11 panoramas/s, and none of them cost accuracy:

| Change | Effect |
|---|---|
| One shared tile pool instead of one per panorama | warm fetch 16.6 ms → 6.3 ms; decode moved into the workers |
| Ladder starts at zoom 2 | 1.8x fewer tiles |
| Optional GPU correlation | that step 10.7x faster (167 ms vs 1793 ms at zoom 4) |

`--workers` is chosen by measurement and differs by backend. On the CPU most of a panorama is spent
waiting on the network, so mild oversubscription wins (8.11 pano/s at 24 threads against 7.32 at 16);
that used to backfire because each panorama opened its own tile pool and Google throttled the flood,
which the shared pool fixed. With a GPU the transforms serialise on one device and extra threads just
queue: 11.54 at 8 threads against 9.38 at 16.

---

## Commands

### `build-bank` — bootstrap reference templates from labelled panoramas

```bash
cr-label build-bank --seed CR_GT.json
```

Only needed to rebuild from scratch; a bank ships in `bank/templates.npz`. Build and label with the
same `--rows` setting — templates and the composites they are matched against must be averaged over
the same part of the panorama.

### `harvest` + `adopt` — teach it a year it does not know

A year missing from the bank is the one case that can produce a *confidently wrong* answer: ranking
only compares the years present, so `2022` against a bank holding 2021/2024/2025/2026 picks 2021 and
wins by a wide margin, because three of its four digits do match. Observed on real panoramas.

The per-digit check exists for exactly this. Measured on real data, correct reads never fall below
`0.854` at their weakest digit while missing years peak at `0.678`, so a gate at `0.75` makes the
missing year abstain instead of guessing.

To add the year properly you do not label thousands of panoramas — composites of one year are
near-identical, so they cluster, and you label the handful of cluster centroids:

```bash
cr-label harvest --panos ids.txt --out clusters --cache cache
# open clusters/cluster_*.png, read the year, type it into clusters/labels.json
cr-label adopt --clusters clusters
```

### `refine-styles` — split a style holding more than one rendering

```bash
cr-label refine-styles --dry-run
```

A "style" must satisfy one property: every template in it shares a set of digit slots. Eras are too
coarse for that. On the real bank, what looked like one `modern` style held **three** renderings,
with glyph blocks spanning columns 26–120, 28–121 and 20–120 — so slots derived for the group landed
off the digits of at least one member, collapsing the margin between adjacent years until the
classifier declined to choose.

Splitting on the glyph bounding box separates them without needing to know when Google changed the
overlay. Effect on the labelled set: abstentions 11 → 1, wrong years 1 → **0**, and throughput *rose*,
because styles sharing an anchor are now detected once instead of once each.

The bank holds 18 years across five renderings:

| Style | Years |
|---|---|
| `legacy_0` | 2009 |
| `legacy_1` | 2015–2020 |
| `modern_0` | 2010–2014 |
| `modern_1` | 2021–2022 |
| `modern_2` | 2023–2026 |

They are **not** chronological — 2010–2014 use the spaced form, 2015–2020 the tight one, 2021 onward
spaced again — so style is decided empirically per template, never inferred from the year.

---

## API key (optional)

**Not needed for normal use.** Tile fetching — the entire image pipeline — requires no key. One is
consulted only for input rows with **no `panoId`**, to resolve coordinates to a panorama.

There is deliberately **no `--api-key` flag**, which would leak the secret into shell history, `ps`
output and CI logs. Configure it in one of these instead:

```bash
export GSV_API_KEY=...                                  # simplest
cp .env.example .env && edit .env                       # .env is gitignored
cr-label label in.json --api-key-file ~/.secrets/gsv    # chmod 600, outside the repo
python -m keyring set cr_labeler google_maps_api_key    # needs the [keyring] extra
```

The key is wrapped in a type whose `repr` renders as `***`, so it cannot reach a log by accident.
Without a key, a row needing one produces a clear error naming the row and its coordinates, and the
rest of the batch continues.

> Google documents Street View **metadata** requests as free of charge, but they still require a
> billing-enabled project. Confirm against
> [current pricing](https://developers.google.com/maps/documentation/streetview/usage-and-billing).

---

## Datasets

Built with [Vali](https://github.com/slashP/Vali), which generates GeoGuessr locations and resolves
each to a verified `panoId`.

**Tracked** — small, and not reproducible by running anything:

| File | n | What it is |
|---|---|---|
| `datasets/labeled.json` | 543 | **Hand-labelled ground truth.** Every accuracy claim rests on it |
| `datasets/to_label*.json` | 543 | The label requests. `accuracy_report.py` reads them to tell the unbiased rows from the adversarial ones |

**Not tracked** — `datasets/generated/` is gitignored, since `vali/make_sets.py` reproduces it and it
runs to tens of megabytes:

| File | n | What it is |
|---|---|---|
| `generated/train.json` | 4,330 | Used to build the template bank |
| `generated/test.json` | 2,419 | Held out, 18 countries |
| `generated/test2.json` | 5,860 | Held out, all 31 countries |
| `generated/gen1.json` | 650 | First-generation panoramas, verified by resolution |

Train and test are split from one pool by a hash of the panorama id, so they are guaranteed disjoint
and identically distributed.

```bash
export PATH=$PATH:$HOME/.dotnet/tools
vali set-download-folder ~/vali-data
bash vali/download.sh                 # the 31 countries these datasets came from, ~25 GB
python vali/make_sets.py --total 4000 --countries "IT,RO,CL,..."
```

`download.sh` takes country codes as arguments if you want a smaller set:
`bash vali/download.sh IT RO CL`.

Both `vali download` and `vali generate` poll the console for a keypress, so they **crash when stdin
is not a terminal**. Wrap them in a pseudo-terminal (`make_sets.py` does this for you):

```bash
script -qec "vali download --country IT" /dev/null
```

### Why the sampling is stratified

The copyright year cannot be filtered on — it exists only in the pixels, which is the whole problem.
The *capture* year can be, and it is the best available proxy: imagery never re-processed still
carries its original stamp. A naive sample returns mostly `2026`; sampling evenly across capture-year
bands is what reaches the rare old years.

### Generations

Vali has **no** generation property — `Generation eq 1` returns *"Unknown property 'Generation'"* —
and capture year is not a substitute: screening Vali's oldest stratum by true panorama resolution
found **0 of 396** to be first generation. Vali cannot reach Gen 1 at all; asked for the oldest
panorama within 20 km of a known Gen 1 location it returns nothing older than 2010.

So generation is taken from the panorama's true resolution, which Google's metadata reports:

| Size | Generation |
|---|---|
| 3328 × 1664 | 1 |
| 13312 × 6656 | 2/3 |
| 16384 × 8192 | 4 |

The same metadata lists neighbouring panorama ids, so crawling outward from one confirmed Gen 1
panorama collects a whole region. `datasets/generated/gen1.json` was built that way from two seeds (South
Australia and Missouri), every entry verified at 3328 × 1664.

**Gen 1 reads better than average — 649/650 (99.8%) yield a year, none come back `None`** — because
the watermark is composited at a fixed pixel size regardless of the imagery underneath.

> Google's metadata also returns a `"© YYYY Google"` string. It is **not** the watermark year — it is
> the year the response was generated. It read `2026` for all twenty ground-truth panoramas, agreeing
> only with the four whose true year happened to be 2026.

### Auditing

```bash
python vali/audit.py --locations datasets/generated/test.json --report r.csv --sample 40
python vali/accuracy_report.py --report r.csv     # against datasets/labeled.json
```

`audit.py` needs no hand labels: it uses capture-year consistency (a stamp cannot predate its photo),
the trekker stratum (~75% carry no watermark), and a contact sheet for reading by eye.
`accuracy_report.py` produces the confusion matrix and separates unbiased from adversarial blocks.

---

## Limits

- **`None` recall is 96.6%**, precision 100%. When it says "no watermark" it is right; it
  occasionally reads a year on a panorama that has none. Every remaining error is of this kind.
- **Confusable digit pairs.** `2016`/`2018` and `2021`/`2022` templates score >0.96 against each
  other. Removing each year from the bank in turn and re-reading panoramas of that year, 7 of 8
  correctly abstained; 2018 was read as 2016. At this glyph size `8` and `6` overlap with genuine
  reads, so no threshold separates them — bank completeness is the mitigation.
- **Years outside 2009–2026** return `CR_unknown`. See `harvest`.
- **203 unbiased rows with zero errors** gives a 95% lower bound of ~98.2%, not proof of perfection.
- The tile endpoint is undocumented. If tiles start returning HTTP 403, check the `User-Agent` in
  `cr_labeler/fetch.py`.

---

## Development

```bash
pytest        # 60 tests, no network — synthetic panoramas throughout
ruff check .
```

| Module | Responsibility |
|---|---|
| `signal.py` | High-pass, FFT cross-correlation, peak picking, consensus |
| `accel.py` | Optional GPU correlation; falls back to numpy when there is none |
| `composite.py` | Detect instances and average them into one watermark image |
| `classify.py` | Style, alignment, per-digit scoring, the abstain/`None` decision |
| `bank.py` | Reference templates: load, save, digit slots, clustering |
| `refine.py` | Splitting a style that holds more than one rendering |
| `build.py` / `discover.py` | Bootstrapping and extending a bank (build time only) |
| `panometa.py` | Panorama dimensions and neighbours, for generation detection |
| `fetch.py` | Tile retrieval, stitching, retry, cache |
| `labeler.py` | Orchestration and the escalation ladder |
| `geoguessr_io.py` | Reading and writing the map JSON |
| `config.py` / `metadata.py` | API key handling and coordinate lookup |
| `vali/` | Dataset generation, auditing, accuracy reporting, doc images |
