# ⏸️ Timeouts

Whether a match had technical or tactical timeouts — and when — isn't obvious
from a demo file, and `datasets` only gives you the terse column reference.
This page covers the three ways to find them, why there are three, and how
they were validated against each other.

```python
demo.timeouts                          # the primary signal
demo.rounds.filter(pl.col("extended_freeze"))   # a corroborating cross-check
from awpy import find_timeout_calls
find_timeout_calls(demo)               # an explicit third signal, if chat survives
```

## `demo.timeouts`: the primary signal

Tactical timeouts (a team-called 30-second break) and technical timeouts (an
admin pause) are both tracked in `CCSGameRules` state, the same way `rounds`
reconstructs round boundaries without needing `round_start` / `round_end`
events:

- Tactical: `m_bTerroristTimeOutActive` / `m_bCTTimeOutActive` (booleans, one
  per side) and `m_flTerroristTimeOutRemaining` / `m_flCTTimeOutRemaining`
  (the countdown, in seconds — CS2's tactical timeout is 30s).
- Technical: `m_bGamePaused` (a single, non-team-specific boolean).

`Parser::timeouts` (`crates/awpy/src/datasets.rs`) watches these fields the
same way `rounds()` watches `m_bFreezePeriod` — a rising edge opens a timeout,
a falling edge closes it, and the round in progress is read from
`m_totalRoundsPlayed` at that moment. Both mechanisms were confirmed
**directly against real demo entity state** (not assumed from documentation)
before being wired up — unlike most `CCSGameRules` fields, which competitive
demos commonly strip (see the `player_blind` / chat / round-event caveats
throughout these docs), the tactical-timeout flags were verified present and
correctly flipping on a real match.

Technical timeouts (`m_bGamePaused`) resolve through the identical mechanism
but haven't been observed actually firing on any demo in the validation set —
not because the field doesn't work, just because none of those matches had an
admin pause. If you have a demo you know contains one and `demo.timeouts`
comes back without a `"technical"` row, that's worth reporting — it would mean
this specific field behaves differently than tactical timeouts on some demos,
which the FAQ's "commonly stripped" pattern would suggest checking for.

## `rounds.extended_freeze`: a corroborating, field-name-agnostic cross-check

`demo.rounds` gets two extra columns: `freeze_ticks` (a round's freeze-period
length) and `extended_freeze` (`true` when that length is an outlier relative
to the demo's own median freeze length, by a factor of 1.5×). The idea: a
pause of *any kind* — not just the two mechanisms above — makes the
surrounding freeze period run long, so this catches a pause even if the
specific field driving it isn't one `Parser::timeouts` already knows to check
(a real gap right now: `svc_SetPause`, the net-message-level admin pause, is
present in the compiled proto surface but not decoded anywhere in this
codebase — a demo whose only pause mechanism is that message would show up
here but not in `demo.timeouts`).

This is corroborating, not authoritative — it flags "something happened
during this round's freeze period," not what. On the one match where this was
validated end-to-end, the two mechanisms agreed exactly:

```python
>>> demo.timeouts["round_num"].to_list()
[9, 10, 18, 19, 20, 22]
>>> demo.rounds.filter(pl.col("extended_freeze"))["round_num"].to_list()
[9, 10, 18, 19, 20, 22]
```

Same six rounds, from two independently-computed signals — one reading two
specific boolean flags, the other computing a length-outlier over a totally
different column. That agreement is the actual evidence this feature works,
not just that it compiles.

Why a median baseline instead of the `mp_freezetime` convar (which would give
an exact expected length rather than an inferred one): `Parser::convars` is
its own full demo scan, independent of the entity-decode pass `rounds()`
already does. Calling it from inside `rounds()` would tax every `rounds()`
call — including the overwhelming majority that don't care about
timeouts — for a signal that's explicitly secondary to `demo.timeouts` anyway.

## `find_timeout_calls`: explicit chat commands, when chat survives

Competitive match plugins (get5, MatchZy, and similar) let a team rep call a
timeout by typing `.tac` (tactical) or `.tech` (technical) in chat. This is
the most *explicit* signal — a literal command, not an inference — but it
depends on `demo.chat` having rows at all, and **GOTV/broadcast demos
overwhelmingly don't**: every demo in the local validation corpus (15 files
across four different broadcast sources) had zero chat rows. `demo.chat`'s
own FAQ entry covers why. `find_timeout_calls` is a thin filter over that same
dataset — no new Rust plumbing, since it's not adding data, just querying
existing data differently:

```python
from awpy import find_timeout_calls

find_timeout_calls(demo)                          # default: .tac / .tech
find_timeout_calls(demo, patterns=(".pause",))     # custom patterns
```

Treat this as "useful on client-recorded or LAN demos, expect an empty frame
on broadcast ones" rather than a dependable primary source.

## Which one should I use?

- Start with `demo.timeouts` — it's the direct, field-based signal and was the
  one validated against real data.
- Cross-reference `rounds.extended_freeze` if you suspect a pause mechanism
  `demo.timeouts` doesn't cover (or just want a second opinion on the rounds
  it does).
- Try `find_timeout_calls` only if you know your demos retain chat — check
  `demo.chat.height` first.
