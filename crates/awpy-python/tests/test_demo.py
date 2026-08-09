"""Tests for the awpy Python bindings.

The fixture-backed tests are skipped when no demo is present (see conftest).
The error-handling tests always run.
"""

from collections import Counter
from pathlib import Path

import polars as pl
import pytest
from awpy import Demo, InvalidDemoError


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        Demo("does-not-exist.dem")


def test_invalid_file_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.dem"
    bogus.write_bytes(b"not a demo file at all")
    with pytest.raises(InvalidDemoError):
        Demo(bogus)


def test_header(demo_path: Path) -> None:
    demo = Demo(demo_path)
    header = demo.header
    assert isinstance(header, dict)
    assert "map_name" in header
    assert header["map_name"].startswith("de_")


def test_events_listing(demo_path: Path) -> None:
    events = Demo(demo_path).events
    assert "player_death" in events
    assert "player_death" in events.names
    assert list(events) == events.names == sorted(events.names)
    assert len(events) == len(events.names)
    assert events.counts["player_death"] > 0
    assert "player_death" in repr(events)


def test_events_access_and_caching(demo_path: Path) -> None:
    demo = Demo(demo_path)
    deaths = demo.events.player_death
    assert isinstance(deaths, pl.DataFrame)
    assert "tick" in deaths.columns
    assert "attacker" in deaths.columns
    assert deaths.height > 0
    # Item access hits the same cached frame; the accessor itself is cached too.
    assert demo.events["player_death"] is deaths
    assert demo.events is demo.events


def test_events_unknown_name_raises(demo_path: Path) -> None:
    events = Demo(demo_path).events
    with pytest.raises(KeyError, match="no_such_event"):
        events["no_such_event"]
    with pytest.raises(AttributeError, match="no_such_event"):
        _ = events.no_such_event


def test_parse_ticks(demo_path: Path) -> None:
    demo = Demo(demo_path)
    # Default props: one row per player per tick, keyed by steamid, with computed
    # world position and core state.
    ticks = demo.ticks()
    assert isinstance(ticks, pl.DataFrame)
    assert {"tick", "steamid", "X", "Y", "Z", "health", "armor", "team_num"} <= set(ticks.columns)
    assert ticks.height > 0
    # Identity is complete (filled from the global slot map) and unique per tick:
    # no pawn/controller double-emission.
    assert ticks["steamid"].is_null().sum() == 0
    assert ticks.select(["tick", "steamid"]).n_unique() == ticks.height
    # A standard match has 10 humans; each tick should have ~10 player rows.
    per_tick = ticks.group_by("tick").len()["len"]
    assert per_tick.max() <= 12  # 10 players (+ occasional connecting/GOTV)
    assert per_tick.median() >= 8
    # X/Y/Z are world coordinates (computed from cell + offset), not raw offsets.
    assert ticks["X"].abs().max() > 100.0
    # Health is a sane 0..100.
    assert 0 <= ticks["health"].min() and ticks["health"].max() <= 100

    # Friendly aliases and raw names both work; a pawn field (team) and a
    # controller field (name) resolve in one call.
    named = demo.ticks(["m_iTeamNum", "name"])
    assert {"tick", "steamid", "m_iTeamNum", "name"} <= set(named.columns)
    assert named["m_iTeamNum"].is_not_null().any()
    assert named["name"].is_not_null().any()

    # players_only=False still dumps every entity separately.
    raw = demo.ticks(["m_iTeamNum"], players_only=False)
    assert {"tick", "entity_id", "class_name", "m_iTeamNum"} <= set(raw.columns)
    assert raw.height > ticks.height


def test_snapshots_parallel_matches_serial(
    demo_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Snapshots decode in parallel across keyframe segments (including the loadout,
    # which follows weapon-entity handles); the result must carry the same
    # rows as a single serial pass -- as a *multiset*, not a fixed order.
    #
    # Several dead players can be true duplicates of each other across every
    # column (steamid/name null, and the rest of a corpse's state -- health,
    # armor, is_scoped, flash_duration, inventory -- frozen identically at
    # death; see `player_states`'s identity-cache doc comment). Verified
    # directly against this fixture: at its last sampled tick, serial and
    # parallel produce byte-identical rows in the same order. But
    # `.sort(all_columns).equals(...)` can still spuriously disagree on which
    # physical row occupies which position among such duplicates (polars'
    # tie-breaking for a many-column sort with heavy nulls isn't guaranteed
    # stable across independently-sorted frames), so compare as multisets
    # (each row hashed, then counted) instead of relying on any sort at all.
    monkeypatch.setenv("AWPY_TICK_SEGMENTS", "1")
    serial = Demo(demo_path).snapshots(every=64)
    monkeypatch.setenv("AWPY_TICK_SEGMENTS", "8")
    parallel = Demo(demo_path).snapshots(every=64)
    assert parallel.height == serial.height
    assert Counter(serial.hash_rows().to_list()) == Counter(parallel.hash_rows().to_list())


def test_ticks_parallel_matches_serial(demo_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # ticks() decodes the demo in parallel across keyframe segments; the result
    # must be bit-identical to a single serial pass (AWPY_TICK_SEGMENTS forces the
    # segment count).
    keys = ["tick", "steamid"]
    monkeypatch.setenv("AWPY_TICK_SEGMENTS", "1")
    serial = Demo(demo_path).ticks()
    monkeypatch.setenv("AWPY_TICK_SEGMENTS", "8")
    parallel = Demo(demo_path).ticks()
    assert parallel.height == serial.height
    assert serial.sort(keys).equals(parallel.sort(keys))


def test_players(demo_path: Path) -> None:
    players = Demo(demo_path).players
    assert isinstance(players, pl.DataFrame)
    assert {"steamid", "name", "side", "team_clan_name"} <= set(players.columns)
    humans = players.filter(pl.col("steamid") > 0)
    assert humans.height == 10
    assert humans["steamid"].n_unique() == 10
    assert set(humans["side"].unique()) <= {"terrorist", "counter-terrorist"}


def test_players_team_clan_names(demo_path: Path) -> None:
    players = Demo(demo_path).players
    playing = players.filter(pl.col("side").is_in(["terrorist", "counter-terrorist"]))
    named = playing.drop_nulls("team_clan_name")
    if named.is_empty():
        pytest.skip("demo has no team clan names (casual matchmaking)")

    # A clan belongs to one side at a time, so a named match has at most two.
    assert named["team_clan_name"].n_unique() <= 2
    assert all(clan.strip() for clan in named["team_clan_name"])
    # Everyone sharing a side shares a clan: the name is read alongside the side,
    # so the two can never describe different moments.
    per_side = named.group_by("side").agg(pl.col("team_clan_name").n_unique().alias("clans"))
    assert (per_side["clans"] == 1).all()
    # Spectators and unassigned players are on no team, so they carry no clan.
    bench = players.filter(~pl.col("side").is_in(["terrorist", "counter-terrorist"]))
    assert bench["team_clan_name"].null_count() == bench.height


def test_tick_rate(demo_path: Path) -> None:
    demo = Demo(demo_path)
    assert isinstance(demo.tick_rate, float)
    # Every competitive demo is 64 or 128 tick; the fallback is 64.
    assert demo.tick_rate in (64.0, 128.0)
    # Consistent with the header's own playback timing, when it reports it.
    header = demo.header
    ticks, seconds = header.get("playback_ticks"), header.get("playback_time")
    if ticks and seconds:
        assert demo.tick_rate == pytest.approx(ticks / seconds, rel=1e-3)


def test_snapshot_single_tick(demo_path: Path) -> None:
    demo = Demo(demo_path)
    tick = demo.rounds.row(0, named=True)["freeze_end_tick"]
    snap = demo.snapshots(ticks=tick)
    assert {
        "tick",
        "steamid",
        "name",
        "side",
        "health",
        "armor",
        "x",
        "y",
        "z",
        "pitch",
        "yaw",
    } <= set(snap.columns)
    assert snap.height == 10  # every player, alive at freeze end
    assert snap["tick"].unique().to_list() == [tick]
    assert snap["health"].min() == 100
    assert snap["x"].null_count() == 0
    # Snapshots must work anywhere in the demo, not just near full packets.
    for probe in (2000, 29000, 60000, 150000):
        assert demo.snapshots(ticks=probe).height == 10, f"empty snapshot at tick {probe}"


def test_snapshot_tick_range(demo_path: Path) -> None:
    demo = Demo(demo_path)
    start = demo.rounds.row(0, named=True)["freeze_end_tick"]
    span = demo.snapshots(start_tick=start, end_tick=start + 128)
    ticks = span["tick"]
    assert ticks.min() >= start and ticks.max() <= start + 128
    assert ticks.n_unique() > 1
    assert span.height == ticks.n_unique() * 10


def test_snapshots_sampled(demo_path: Path) -> None:
    demo = Demo(demo_path)

    # `every=N`: evenly spaced ticks, same schema as a single snapshot, 10 players each.
    every = demo.snapshots(every=256)
    one = demo.snapshots(ticks=demo.rounds["freeze_end_tick"][0])
    assert set(every.columns) == set(one.columns)
    gaps = every["tick"].unique().sort().diff().drop_nulls()
    assert gaps.min() >= 256 and gaps.max() == 256  # gap-robust stride hits exactly N
    assert every.height == every["tick"].n_unique() * 10
    # Weapons must resolve in the sampled pass, not just single snapshot() — the
    # filtered decode has to keep weapon entities, else the loadout is all null.
    assert every["primary_weapon"].drop_nulls().len() > 0
    assert every.filter(pl.col("inventory") != "").height > 0

    # `seconds=S` converts via the tick rate (64-tick demo -> 256-tick stride).
    per_sec = demo.snapshots(seconds=4.0)
    assert per_sec["tick"].unique().sort().diff().drop_nulls().max() == 256

    # `events=` samples exactly the ticks those events fired on (they are real
    # frame ticks, so every kill tick is present).
    kill_ticks = set(demo.kills["tick"].to_list())
    on_kills = demo.snapshots(events="player_death")
    assert set(on_kills["tick"].to_list()) == kill_ticks

    # Explicit ticks (drawn from real frames), and the stride ∪ events union.
    some_ticks = every["tick"].unique().to_list()[:3]
    picked = demo.snapshots(ticks=some_ticks)
    assert set(picked["tick"].to_list()) == set(some_ticks)
    union = demo.snapshots(every=256, events="player_death")
    assert kill_ticks <= set(union["tick"].to_list())
    assert set(every["tick"].to_list()) <= set(union["tick"].to_list())

    # Selectors are required, and every/seconds are mutually exclusive.
    with pytest.raises(ValueError):
        demo.snapshots()
    with pytest.raises(ValueError):
        demo.snapshots(every=64, seconds=1.0)


def test_snapshot_economy(demo_path: Path) -> None:
    demo = Demo(demo_path)
    # Freeze end of the first (pistol) round: all 10 players alive and armed.
    tick = demo.rounds.row(0, named=True)["freeze_end_tick"]
    snap = demo.snapshots(ticks=tick)
    econ = {
        "health",
        "armor",
        "has_helmet",
        "has_defuser",
        "has_bomb",
        "active_weapon",
        "primary_weapon",
        "secondary_weapon",
        "fire_grenades",
        "smoke_grenades",
        "he_grenades",
        "flashbangs",
        "decoy_grenades",
        "equipment_value",
        "equipment_value_round_start",
        "money",
        "is_crouched",
        "is_walking",
        "is_jumping",
        "is_in_bomb_zone",
        "is_scoped",
        "is_defusing",
        "flash_duration",
        "inventory",
    }
    assert econ <= set(snap.columns)
    assert snap.height == 10
    assert snap["equipment_value"].dtype == pl.Int32
    # Every alive player bought something and carries a knife.
    assert (snap["equipment_value"] > 0).all()
    assert snap["inventory"].str.contains("knife").all()
    # Pistol round: everyone has a pistol, nobody has a rifle.
    assert snap["secondary_weapon"].null_count() == 0
    assert snap["primary_weapon"].null_count() == 10
    # Exactly one player carries the bomb.
    assert snap["has_bomb"].sum() == 1
    # Money is a sane amount and typed; the active weapon resolves for everyone.
    assert snap["money"].min() >= 0
    assert snap["active_weapon"].null_count() == 0
    # Grenade counts are non-negative; the loadout never lists two bombs (the
    # over-scan bug that a stale inventory slot would cause).
    for col in ("he_grenades", "flashbangs", "smoke_grenades", "fire_grenades", "decoy_grenades"):
        assert snap[col].min() >= 0
    assert (snap["inventory"].str.count_matches("c4") <= 1).all()


def test_snapshot_place_and_ping(demo_path: Path) -> None:
    """``place`` (``m_szLastPlaceName``) and ``ping`` (``m_iPing``, read off
    the controller, not the pawn) are both present on every snapshot row.

    ``ping`` reads a constant ``0`` on LAN-recorded demos (the fixture this
    was validated against is a LAN event) -- that's expected, not a
    GOTV-stripped-field artifact like some other fields in this codebase;
    it isn't asserted non-zero here since a LAN fixture legitimately has
    near-zero ping.
    """
    demo = Demo(demo_path)
    snap = demo.snapshots(seconds=1.0)
    assert {"place", "ping"} <= set(snap.columns)
    assert snap["ping"].dtype == pl.Int32
    assert snap["place"].dtype == pl.String
    assert snap["ping"].min() >= 0
    # Named callouts appear once players have moved past spawn -- `place` is
    # `""` (not null) before a player's first tick in a named area, so a real
    # match has both `""` early on and multiple real callouts thereafter.
    places = set(snap["place"].drop_nulls().unique().to_list())
    assert len(places - {""}) > 1


def test_dead_players_keep_identity(demo_path: Path) -> None:
    """CS2 explicitly invalidates a pawn's `m_hController` back-link once the
    controller hands off to a fresh observer pawn on death -- the corpse
    itself keeps position/team/health, just not the link back to who it was.
    `player_states` caches each pawn's last-known identity (keyed by its own
    entity index+serial) while the link is live, and falls back to it once
    the link goes invalid -- so most dead-player rows keep their steamid/name
    instead of going null.

    Not a 100% guarantee: the cache resets at every full-packet keyframe (a
    boundary intrinsic to the demo, not to how many segments it's decoded in
    -- needed so parallel decoding stays bit-identical to a serial pass, see
    `collect_states`'s doc comment), so a player already dead at the start of
    a keyframe interval has no earlier live tick in that interval to have
    seeded it from. That trades away some resolution rate for a real
    correctness guarantee (resetting on segment boundaries instead, which are
    coarser but segment-count-dependent, resolved ~90% on this fixture but
    broke serial/parallel equivalence) -- hence a generous floor here, well
    below what's typically observed (~60%+ on the fixture this was tuned
    against), rather than a tight bound tied to one demo's keyframe spacing.
    """
    demo = Demo(demo_path)
    snap = demo.snapshots(seconds=1.0)
    dead = snap.filter(pl.col("health") == 0)
    assert dead.height > 0, "a full match always has dead-player rows"
    resolved = dead.filter(pl.col("steamid").is_not_null())
    assert resolved.height / dead.height > 0.3


def test_flashbangs_reaches_two(demo_path: Path) -> None:
    """CS2 lets a player hold 2 flashbangs -- the one grenade type that isn't
    capped at 1. `flashbangs` is read from the pawn's own
    `m_pWeaponServices.m_iAmmo[14]` (reserve ammo) rather than counted from
    `m_hMyWeapons` entities, since CS2 only ever instantiates one
    `CFlashbang` entity per player regardless of hold count -- entity-counting
    alone can never see past 1. Bounds-check every other grenade type stays
    capped at 1 (still counted from entities) while flashbangs reaches 2 on a
    real match, matching an independent count (demoparser2's own `inventory`
    list) on the fixture this was validated against.
    """
    demo = Demo(demo_path)
    snap = demo.snapshots(seconds=1.0)
    for col in ("he_grenades", "smoke_grenades", "fire_grenades", "decoy_grenades"):
        assert snap[col].max() <= 1
    assert snap["flashbangs"].max() <= 2
    # Not a hard guarantee for every possible demo, but overwhelmingly true
    # for any real competitive match of reasonable length -- double-flash
    # buys are common. A demo where this never holds would be worth a look.
    assert (snap["flashbangs"] == 2).any()
    # The primary/secondary names, when present, appear in the inventory string.
    for row in snap.iter_rows(named=True):
        if row["secondary_weapon"]:
            assert row["secondary_weapon"] in row["inventory"]


def test_blinds(demo_path: Path) -> None:
    blinds = Demo(demo_path).blinds
    assert isinstance(blinds, pl.DataFrame)
    expected = {
        "tick",
        "attacker_steamid",
        "attacker_name",
        "attacker_side",
        "victim_steamid",
        "victim_name",
        "victim_side",
        "victim_x",
        "victim_y",
        "victim_z",
        "duration",
        "is_teammate",
    }
    assert expected <= set(blinds.columns)
    assert blinds["duration"].dtype == pl.Float32
    assert blinds["is_teammate"].dtype == pl.Boolean
    # Demos without flashbang_detonate (rare) yield an empty frame; the fixture
    # is a real match, so it has flashes.
    if blinds.height:
        assert (blinds["duration"] > 0).all()
        assert blinds["duration"].max() <= 6.0  # engine flash cap is ~5 s
        # Every blinded victim is resolved to a side and a position.
        assert blinds["victim_side"].null_count() == 0
        assert set(blinds["victim_side"].unique()) <= {"terrorist", "counter-terrorist"}
        assert blinds["victim_x"].null_count() == 0
        # Rows are in tick order.
        assert blinds["tick"].to_list() == sorted(blinds["tick"].to_list())
        # A self-flash (attacker == victim) is always a team-flash.
        self_flash = blinds.filter(pl.col("attacker_steamid") == pl.col("victim_steamid"))
        if self_flash.height:
            assert self_flash["is_teammate"].all()
        # demo.flashes is a literal alias for demo.blinds.
        assert Demo(demo_path).flashes.equals(blinds)


def test_item_events(demo_path: Path) -> None:
    items = Demo(demo_path).item_events
    assert isinstance(items, pl.DataFrame)
    expected = {
        "tick",
        "action",
        "steamid",
        "name",
        "side",
        "item",
        "x",
        "y",
        "z",
        "original_owner_steamid",
        "cost",
        "near_buy_zone",
    }
    assert expected <= set(items.columns)
    assert items.height > 0
    assert set(items["action"].unique()) <= {"purchase", "pickup", "drop"}
    # The knife is excluded (default loadout, never bought/dropped meaningfully).
    assert "knife" not in set(items["item"].unique())
    # Rows are in tick order.
    assert items["tick"].to_list() == sorted(items["tick"].to_list())

    purchases = items.filter(pl.col("action") == "purchase")
    assert purchases.height > 0
    # A purchase is the buyer's own weapon, at a plausible cost (never a money
    # reset, which would show as a huge "cost").
    assert (purchases["original_owner_steamid"] == purchases["steamid"]).all()
    assert purchases["cost"].min() > 0
    assert purchases["cost"].max() <= 6500
    # Pickups and drops carry no cost.
    non_purchase = items.filter(pl.col("action") != "purchase")
    assert non_purchase["cost"].null_count() == non_purchase.height
    # Drops record whether they happened near a buy zone.
    drops = items.filter(pl.col("action") == "drop")
    if drops.height:
        assert drops["near_buy_zone"].null_count() == 0


def test_chat(demo_path: Path) -> None:
    chat = Demo(demo_path).chat
    assert isinstance(chat, pl.DataFrame)
    # Server-side demos may strip chat entirely; the schema must hold anyway.
    assert {"tick", "entity_index", "name", "message", "channel"} <= set(chat.columns)
    if chat.height:
        assert chat["message"].null_count() == 0


def test_convars(demo_path: Path) -> None:
    convars = Demo(demo_path).convars
    assert isinstance(convars, dict)
    assert convars  # every demo carries at least the signon convars
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in convars.items())
    assert any(key.startswith("mp_") for key in convars)


def test_rounds(demo_path: Path) -> None:
    demo = Demo(demo_path)
    rounds = demo.rounds
    assert isinstance(rounds, pl.DataFrame)
    assert {
        "round_num",
        "start_tick",
        "freeze_end_tick",
        "end_tick",
        "winner",
        "winner_side",
        "reason_name",
        "freeze_ticks",
        "extended_freeze",
    } <= set(rounds.columns)
    assert rounds.height > 0
    # Winners are terrorist / counter-terrorist.
    assert set(rounds["winner_side"].unique()) <= {"terrorist", "counter-terrorist"}
    # End ticks are strictly increasing across rounds.
    end = rounds["end_tick"].to_list()
    assert end == sorted(end)
    # freeze_ticks, where known, is a positive span.
    known = rounds.filter(pl.col("freeze_ticks").is_not_null())
    if known.height:
        assert (known["freeze_ticks"] > 0).all()


def test_timeouts(demo_path: Path) -> None:
    demo = Demo(demo_path)
    timeouts = demo.timeouts
    assert isinstance(timeouts, pl.DataFrame)
    assert {
        "side",
        "type",
        "start_tick",
        "end_tick",
        "remaining_at_start",
        "round_num",
    } <= set(timeouts.columns)
    assert set(timeouts["type"].unique()) <= {"tactical", "technical"}
    if timeouts.height:
        tactical = timeouts.filter(pl.col("type") == "tactical")
        assert set(tactical["side"].unique()) <= {"terrorist", "counter-terrorist"}
        # Every closed timeout ends at or after it starts.
        closed = timeouts.filter(pl.col("end_tick").is_not_null())
        if closed.height:
            assert (closed["end_tick"] >= closed["start_tick"]).all()
        # Cross-check: every tactical timeout lands inside a round flagged
        # extended_freeze (the two detection mechanisms should agree).
        extended = set(demo.rounds.filter(pl.col("extended_freeze"))["round_num"].to_list())
        assert set(tactical["round_num"].to_list()) <= extended


def test_kills(demo_path: Path) -> None:
    demo = Demo(demo_path)
    kills = demo.kills
    assert isinstance(kills, pl.DataFrame)
    expected = {
        "attacker_steamid",
        "attacker_name",
        "attacker_side",
        "attacker_x",
        "attacker_y",
        "attacker_z",
        "victim_steamid",
        "victim_name",
        "victim_side",
        "victim_x",
        "victim_y",
        "victim_z",
        "assister_steamid",
        "assister_name",
        "assister_side",
        "weapon",
        "headshot",
        "hitgroup_name",
        "tick",
    }
    assert expected <= set(kills.columns)
    assert kills.height > 0
    assert kills["headshot"].dtype == pl.Boolean
    assert kills["attacker_x"].dtype == pl.Float32
    assert kills["attacker_steamid"].dtype == pl.UInt64
    # Sides are terrorist / counter-terrorist (or null for world kills).
    sides = set(kills["attacker_side"].drop_nulls().unique())
    assert sides <= {"terrorist", "counter-terrorist"}


def test_kill_event_fields(demo_path: Path) -> None:
    """``attacker_blind`` / ``attacker_in_air`` / ``distance`` are read
    directly off `player_death`'s own event keys (the server's own
    determination), not derived from entity state.
    """
    demo = Demo(demo_path)
    kills = demo.kills
    assert {"attacker_blind", "attacker_in_air", "distance"} <= set(kills.columns)
    assert kills["attacker_blind"].dtype == pl.Boolean
    assert kills["attacker_in_air"].dtype == pl.Boolean
    assert kills["distance"].dtype == pl.Float32
    # A full match has at least one blind kill (flash-into-peek is common).
    assert kills["attacker_blind"].any()
    # `distance` is in meters, not Hammer units -- roughly 39.37x smaller than
    # the geometric distance between attacker_*/victim_* (which are in
    # Hammer units/inches; 39.37 is exactly inches-per-meter).
    non_world = kills.filter(pl.col("attacker_steamid").is_not_null() & (pl.col("distance") > 0))
    geo_dist = (
        (pl.col("attacker_x") - pl.col("victim_x")) ** 2
        + (pl.col("attacker_y") - pl.col("victim_y")) ** 2
        + (pl.col("attacker_z") - pl.col("victim_z")) ** 2
    ).sqrt()
    ratio = non_world.select((geo_dist / pl.col("distance")).alias("ratio"))["ratio"]
    assert (ratio - 39.37).abs().max() < 2.0
    # `weapon == "world"` kills (self/environment deaths) have no
    # attacker-victim pair for the server to measure a distance from.
    world = kills.filter(pl.col("weapon") == "world")
    if world.height:
        assert (world["distance"] == 0.0).all()


def test_damages(demo_path: Path) -> None:
    demo = Demo(demo_path)
    damages = demo.damages
    assert isinstance(damages, pl.DataFrame)
    assert {
        "attacker_name",
        "victim_name",
        "weapon",
        "dmg_health",
        "hitgroup_name",
        "health_pre",
        "health_post",
        "armor_pre",
        "armor_post",
        "tick",
    } <= set(damages.columns)
    assert damages.height > 0
    # Pre-health is health_post + dmg_health, clamped to the 100 HP maximum
    # (raw damage can exceed remaining health on a lethal hit).
    bad = damages.filter(
        pl.col("health_pre")
        != pl.min_horizontal(pl.col("health_post") + pl.col("dmg_health"), pl.lit(100))
    )
    assert bad.height == 0
    assert damages["health_pre"].max() <= 100


def test_dmg_health_real(demo_path: Path) -> None:
    """`dmg_health_real` is the victim's *actual* health lost to a hit: their
    true health at the moment of the hit (`health_post` from their previous
    hit this life, or 100 on their first hit since spawning) minus this hit's
    own `health_post`. It is **not** always `<= dmg_health` -- that only
    holds for the overkill case (a raw event value that overstates real
    remaining health); it can also run the other way, e.g. a round-timeout
    `weapon == "world"` loss networks a small placeholder `dmg_health` rather
    than a real damage amount, while `dmg_health_real` correctly reports the
    victim's actual (often much larger) health loss.

    Re-deriving it independently for consecutive hits within one life
    (previous row's `health_post` minus this row's) matches *most* of the
    time but not always exactly -- a real, small, not-fully-root-caused
    precision gap documented on the field itself (see `Damage.dmg_health_real`'s
    doc comment). So this only checks the invariants that are unconditionally
    true, not exact reconstruction.
    """
    demo = Demo(demo_path)
    dmg = demo.damages
    assert "dmg_health_real" in dmg.columns
    assert (dmg["dmg_health_real"] >= 0).all()
    # A full match has at least one overkill hit (dmg_health_real < dmg_health,
    # a lethal hit that dealt more raw damage than the victim had left), and
    # every one of those is lethal by definition.
    overkill = dmg.filter(pl.col("dmg_health_real") < pl.col("dmg_health"))
    assert overkill.height > 0
    assert (overkill["health_post"] == 0).all()


def test_rounds_official_end(demo_path: Path) -> None:
    rounds = Demo(demo_path).rounds
    assert "official_end_tick" in rounds.columns


def test_bomb(demo_path: Path) -> None:
    bomb = Demo(demo_path).bomb
    assert isinstance(bomb, pl.DataFrame)
    assert {"tick", "event", "steamid", "name", "bombsite", "x", "y", "z"} <= set(bomb.columns)
    assert set(bomb["event"].unique()) <= {
        "pickup",
        "drop",
        "start_plant",
        "interrupt_plant",
        "finish_plant",
        "defuse",
        "start_defuse",
        "explode",
    }


def test_grenades(demo_path: Path) -> None:
    g = Demo(demo_path).grenades
    assert isinstance(g, pl.DataFrame)
    assert {"tick", "thrower_name", "type", "entity_id", "x", "y", "z"} <= set(g.columns)
    # The smoke-grenade projectile is BOTH a grenade (its throw) and a smoke (its
    # cloud); its throw trajectory must still land in grenades. Guards the shared
    # projectile pass against dropping the class's second role.
    assert g.filter(pl.col("type") == "smoke").height > 0
    assert set(g["type"].unique()) <= {"smoke", "he", "flashbang", "molotov", "decoy", "grenade"}


def test_grenade_throws(demo_path: Path) -> None:
    demo = Demo(demo_path)
    throws = demo.grenade_throws
    assert isinstance(throws, pl.DataFrame)
    assert {
        "thrower_name",
        "thrower_steamid",
        "thrower_side",
        "type",
        "entity_id",
        "throw_tick",
        "throw_x",
        "throw_y",
        "throw_z",
        "land_tick",
        "land_x",
        "land_y",
        "land_z",
        "land_is_precise",
    } <= set(throws.columns)
    assert throws.height > 0
    # One row per throw, and every throw's land is at or after its own throw.
    assert (throws["land_tick"] >= throws["throw_tick"]).all()
    # Every entity_id in grenade_throws also appears in the full trajectory,
    # scoped to that throw's own tick window (entity ids can be reused later
    # in the match by an unrelated throw, so a plain entity_id join is wrong).
    traj = demo.grenades
    for row in throws.head(20).iter_rows(named=True):
        window = traj.filter(
            (pl.col("entity_id") == row["entity_id"])
            & (pl.col("tick") >= row["throw_tick"])
            & (pl.col("tick") <= row["land_tick"])
        )
        assert window.height > 0
        assert window["tick"].min() == row["throw_tick"]
        assert window["tick"].max() == row["land_tick"]
    # HE landings that were refined report it via land_is_precise; every other
    # type is always trajectory-derived (never "precise").
    assert not throws.filter(pl.col("type") != "he")["land_is_precise"].any()


def test_fires_and_smokes(demo_path: Path) -> None:
    demo = Demo(demo_path)
    for df in (demo.fires, demo.smokes):
        assert isinstance(df, pl.DataFrame)
        assert {
            "start_tick",
            "end_tick",
            "thrower_steamid",
            "entity_id",
            "x",
            "y",
            "z",
        } <= set(df.columns)
        assert "tick" not in df.columns  # one row per instance, not per tick
        assert df.height > 0
        # Exactly one row per (entity_id, start_tick) instance.
        assert df.height == df.select("entity_id", "start_tick").n_unique()
        # A single burn covers a positive span of ticks.
        assert (df["end_tick"] > df["start_tick"]).all()


def test_shots(demo_path: Path) -> None:
    shots = Demo(demo_path).shots
    assert isinstance(shots, pl.DataFrame)
    assert {
        "tick",
        "steamid",
        "name",
        "side",
        "x",
        "y",
        "z",
        "pitch",
        "yaw",
        "weapon",
        "scoped",
        "inaccuracy",
        "num_bullets_remaining",
    } <= set(shots.columns)
    assert shots.height > 0

    # Every shot resolves its shooter from the pawn handle.
    assert shots["steamid"].is_not_null().all()

    # Clip / accuracy come from following the shooter's active-weapon handle to a
    # weapon entity, which the shared event pass decodes via the weapon-class
    # filter. If that filter ever stops covering a fired weapon, clip/accuracy
    # silently go null — so guard the common case: firearm shots dominate
    # weapon_fire events, so the large majority of shots must resolve a clip, and
    # at least one must carry a real (positive) round count.
    clip_frac = shots["num_bullets_remaining"].is_not_null().mean()
    assert clip_frac >= 0.7, f"only {clip_frac:.1%} of shots resolved a weapon clip"
    assert shots["inaccuracy"].is_not_null().mean() >= 0.7
    assert (shots["num_bullets_remaining"] > 0).sum() > 0


def test_parallel_parse_populates_all(demo_path: Path) -> None:
    # Accessing any one dataset kicks off the batched parallel parse, which
    # decodes every independent pass concurrently and caches them all. So after a
    # single access, every dataset must be present, non-empty, and identical to
    # what a fresh (independently parsed) Demo produces for it.
    d = Demo(demo_path)
    _ = d.kills  # triggers ensure_parsed -> all passes in parallel
    fresh = Demo(demo_path)
    for name in ("kills", "damages", "bomb", "shots", "grenades", "rounds", "players", "stats"):
        got = getattr(d, name)
        assert isinstance(got, pl.DataFrame)
        # Row counts are deterministic across independent parses of the same demo
        # (the parallel passes are byte-identical to serial ones).
        assert got.height == getattr(fresh, name).height, f"{name} row count differs"
    # A standard match has a full server of players and at least one round.
    assert d.players.height >= 10
    assert d.rounds.height > 0


def test_stats(demo_path: Path) -> None:
    stats = Demo(demo_path).stats
    assert isinstance(stats, pl.DataFrame)
    assert {
        "steamid",
        "name",
        "rounds_played",
        "kills",
        "deaths",
        "assists",
        "flash_assists",
        "headshot_kills",
        "opening_kills",
        "opening_deaths",
        "traded_deaths",
        "kast",
        "adr",
    } <= set(stats.columns)
    assert stats.height > 0
    # KAST is a percentage; ADR is non-negative; every opening kill has a death.
    assert stats["kast"].max() <= 100.0
    assert stats["adr"].min() >= 0.0
    assert stats["opening_kills"].sum() == stats["opening_deaths"].sum()
    # Non-negative counts throughout.
    assert stats["kills"].min() >= 0
    # Flash assists are a subset of assists (every flash assist is an assist).
    assert (stats["flash_assists"] <= stats["assists"]).all()

    # Utility stats: present, non-negative, and something happened over a match.
    assert {
        "utility_damage",
        "flashes_thrown",
        "enemies_flashed",
        "flash_duration_dealt",
    } <= set(stats.columns)
    for col in ("utility_damage", "flashes_thrown", "enemies_flashed", "flash_duration_dealt"):
        assert stats[col].min() >= 0
    assert stats["flashes_thrown"].sum() > 0
    assert stats["utility_damage"].sum() > 0
    # You can't blind more enemies than you threw flashes at (loosely).
    assert stats["flash_duration_dealt"].sum() > 0


CLUTCH_COLS = ("clutch_1v1", "clutch_1v2", "clutch_1v3", "clutch_1v4", "clutch_1v5")


def test_stats_clutches(demo_path: Path) -> None:
    demo = Demo(demo_path)
    stats = demo.stats
    assert {"clutches_played", "clutches_won", *CLUTCH_COLS} <= set(stats.columns)
    for col in ("clutches_played", "clutches_won", *CLUTCH_COLS):
        assert stats[col].min() >= 0

    # You cannot win more clutches than you played.
    assert (stats["clutches_won"] <= stats["clutches_played"]).all()
    # The 1vN breakdown covers exactly the wins.
    breakdown = sum(stats[col] for col in CLUTCH_COLS)
    assert (breakdown == stats["clutches_won"]).all()

    # At most one clutch per side per round, so never more than 2 per round.
    n_rounds = demo.rounds.filter(~pl.col("is_knife_round")).height
    assert stats["clutches_played"].sum() <= 2 * n_rounds
    # A full match always leaves someone last alive at least once.
    assert stats["clutches_played"].sum() > 0


def test_kills_trade_flags(demo_path: Path) -> None:
    kills = Demo(demo_path).kills
    assert {"is_trade", "victim_traded"} <= set(kills.columns)
    assert kills["is_trade"].dtype == pl.Boolean
    assert kills["victim_traded"].dtype == pl.Boolean

    trades = kills.filter(pl.col("is_trade"))
    traded = kills.filter(pl.col("victim_traded"))
    assert trades.height > 0, "a full match always has trade kills"
    # Duals: every trade kill must be preceded by a traded death on the other
    # side, and one kill can avenge several teammates at once — so traded deaths
    # are at least as numerous as the kills that avenged them, never fewer.
    assert traded.height >= trades.height
    # A trade kill is by definition against the opposing side.
    assert (trades["attacker_side"] != trades["victim_side"]).all()
    # A death with no resolved victim side has no team to avenge it.
    assert traded["victim_side"].null_count() == 0


def test_traded_deaths_match_the_kills_dataset(demo_path: Path) -> None:
    """``stats.traded_deaths`` and ``kills.victim_traded`` share one classifier."""
    demo = Demo(demo_path)
    rounds = demo.rounds.sort("round_num")
    live = rounds.filter(~pl.col("is_knife_round"))
    per_player = (
        demo.kills.filter(pl.col("victim_traded"))
        .group_by("victim_steamid")
        .len()
        .rename({"victim_steamid": "steamid", "len": "flagged"})
    )
    joined = demo.stats.join(per_player, on="steamid", how="left").with_columns(
        pl.col("flagged").fill_null(0)
    )
    if live.height == rounds.height:
        # No knife round to exclude, so the two must agree exactly.
        assert (joined["traded_deaths"] == joined["flagged"]).all()
    else:
        # Knife-round trades are dropped from stats but still flagged on kills.
        assert (joined["traded_deaths"] <= joined["flagged"]).all()


def test_round_economy(demo_path: Path) -> None:
    econ = Demo(demo_path).round_economy
    assert isinstance(econ, pl.DataFrame)
    assert {"round_num", "side", "equipment_value", "buy_type", "n_players"} <= set(econ.columns)
    assert econ.height > 0
    # One row per (round, side).
    assert econ.select("round_num", "side").n_unique() == econ.height
    assert set(econ["side"].unique()) <= {"terrorist", "counter-terrorist"}
    assert set(econ["buy_type"].unique()) <= {"pistol", "eco", "force", "full"}
    # Team equipment is non-negative, and a real match has a mix of buy types.
    assert econ["equipment_value"].min() >= 0
    assert econ["buy_type"].n_unique() >= 2
    # Round 1 is a pistol round for both teams (detected via the halftime side
    # flip, so there is a second pistol round later too).
    assert (econ.filter(pl.col("round_num") == 1)["buy_type"] == "pistol").all()
    pistol_rounds = econ.filter(pl.col("buy_type") == "pistol")["round_num"].unique()
    assert len(pistol_rounds) == 2  # round 1 and the second-half pistol


def test_round_num_joins_kills_damages_shots_and_snapshots(demo_path: Path) -> None:
    """`round_num` on kills/damages/shots/snapshots is a join against
    `demo.rounds` by tick (those datasets come from a separate decode pass
    than `rounds()`, so it can't be read inline) -- the round whose own
    boundary (`start_tick`, falling back to `freeze_end_tick`/`end_tick`) is
    the latest one at or before that row's own tick.
    """
    demo = Demo(demo_path)
    rounds = demo.rounds.sort("round_num")
    max_round = rounds["round_num"].max()

    for name in ("kills", "damages", "shots"):
        df = getattr(demo, name)
        assert "round_num" in df.columns
        resolved = df.filter(pl.col("round_num").is_not_null())
        assert resolved.height > 0, f"{name}: no rows resolved a round_num"
        assert resolved["round_num"].min() >= 1
        assert resolved["round_num"].max() <= max_round

    snap = demo.snapshots(seconds=1.0)
    assert "round_num" in snap.columns
    resolved = snap.filter(pl.col("round_num").is_not_null())
    assert resolved.height > 0
    assert resolved["round_num"].min() >= 1
    assert resolved["round_num"].max() <= max_round

    # Spot-check against the actual boundaries: no kill is attributed to a
    # round before that round's own start_tick (when known -- the very first
    # round of a demo that starts mid-round may have none).
    joined = demo.kills.filter(pl.col("round_num").is_not_null()).join(
        rounds.select("round_num", "start_tick"), on="round_num"
    )
    bad = joined.filter(
        pl.col("start_tick").is_not_null() & (pl.col("tick") < pl.col("start_tick"))
    )
    assert bad.height == 0


def test_team_clan_name_and_cash_spent_this_round(demo_path: Path) -> None:
    demo = Demo(demo_path)
    snap = demo.snapshots(seconds=1.0)
    assert {"team_clan_name", "cash_spent_this_round"} <= set(snap.columns)
    # A real match has (at least) two distinct team names.
    assert snap["team_clan_name"].drop_nulls().n_unique() >= 2
    assert snap["cash_spent_this_round"].min() >= 0
    assert (snap["cash_spent_this_round"] > 0).any()

    kills = demo.kills
    assert {
        "attacker_team_clan_name",
        "victim_team_clan_name",
        "attacker_cash_spent_this_round",
        "victim_cash_spent_this_round",
    } <= set(kills.columns)
    resolved = kills.filter(pl.col("attacker_team_clan_name").is_not_null())
    assert resolved.height > 0
    assert resolved["attacker_cash_spent_this_round"].min() >= 0


def test_schema_constants() -> None:
    from awpy import GAME_EVENTS, SNAPSHOT_PROPERTIES

    assert isinstance(SNAPSHOT_PROPERTIES, dict) and SNAPSHOT_PROPERTIES
    assert isinstance(GAME_EVENTS, dict) and "player_death" in GAME_EVENTS
    # Values map to engine property names / descriptions (all strings).
    assert all(isinstance(v, str) for v in SNAPSHOT_PROPERTIES.values())


def test_snapshot_properties_match_schema(demo_path: Path) -> None:
    # The SNAPSHOT_PROPERTIES catalog must stay in sync with what snapshot() emits.
    from awpy import SNAPSHOT_PROPERTIES

    tick = Demo(demo_path).rounds.row(1, named=True)["freeze_end_tick"]
    cols = set(Demo(demo_path).snapshots(ticks=tick).columns) - {"tick"}
    assert cols == set(SNAPSHOT_PROPERTIES)
