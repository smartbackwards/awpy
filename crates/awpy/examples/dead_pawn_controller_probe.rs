//! Diagnoses why `PlayerState.steamid`/`name` go null for dead players for
//! the rest of the round (found via `dem_to_data_v3.py`'s ticks.csv export
//! showing ~17% null-identity rows, all correlated with `health == 0`).
//!
//! For every `CCSPlayerPawn` at a target tick, prints: team, health, the raw
//! `m_hController` handle value (present or absent), whether that handle's
//! index resolves to *any* entity, and whether that entity is active and is
//! a `CCSPlayerController`. This distinguishes three failure modes:
//!   1. `m_hController` field itself is absent on the corpse (never
//!      networked / cleared on death).
//!   2. The handle is present but its index resolves to no entity, or an
//!      inactive one (dangling/dormant).
//!   3. The handle resolves to a live entity that isn't a
//!      `CCSPlayerController` (index got reused for something else).
//!
//! Confirmed: mode 1 (`0xffffff` -- the invalid-handle sentinel, `0x3FFF`
//! masked). CS2 deliberately clears the corpse's `m_hController` once the
//! controller hands off to a fresh observer pawn on death; it's real game
//! state, not a parser bug, and it isn't recoverable from the corpse alone.
//! The fix shipped as a pawn-identity cache (keyed by the pawn's own
//! `(index, serial)`, filled while the link is live, consulted once it
//! isn't) -- see `player_states`'s doc comment in `datasets.rs`.
//!
//! `cargo run --release --example dead_pawn_controller_probe -- <demo> <tick>`

use std::collections::HashSet;
use std::path::PathBuf;

use awpy::Parser;

const PLAYER_PAWN_CLASS: &str = "CCSPlayerPawn";
const PLAYER_CONTROLLER_CLASS: &str = "CCSPlayerController";

fn main() {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().expect("demo path"));
    let target_tick: i32 = args.next().expect("tick").parse().expect("tick int");

    let parser = Parser::from_file(&path).expect("open");
    let filter: HashSet<&str> = HashSet::from([PLAYER_PAWN_CLASS, PLAYER_CONTROLLER_CLASS]);
    let mut ctrl_key: Option<u64> = None;
    let mut team_key: Option<u64> = None;
    let mut health_key: Option<u64> = None;
    let mut steamid_key: Option<u64> = None;
    let mut done = false;

    parser
        .run_to_end_filtered(&filter, |ctx| {
            if done || ctx.tick() != target_tick {
                return;
            }
            done = true;
            let pser = ctx.serializers().get(PLAYER_PAWN_CLASS);
            let cser = ctx.serializers().get(PLAYER_CONTROLLER_CLASS);
            let ctrl_h = *ctrl_key
                .get_or_insert_with(|| pser.and_then(|s| s.resolve_field_key("m_hController")).expect("ctrl key"));
            let team_h = *team_key.get_or_insert_with(|| {
                pser.and_then(|s| s.resolve_field_key("m_iTeamNum")).expect("team key")
            });
            let health_h = *health_key.get_or_insert_with(|| {
                pser.and_then(|s| s.resolve_field_key("m_iHealth")).expect("health key")
            });
            let sid_h = *steamid_key
                .get_or_insert_with(|| cser.and_then(|s| s.resolve_field_key("m_steamID")).expect("steamid key"));

            for (idx, pawn) in ctx.entities().iter() {
                if !pawn.active || pawn.class_name.as_ref() != PLAYER_PAWN_CLASS {
                    continue;
                }
                let team = pawn.get_i64(Some(team_h));
                if team <= 0 {
                    continue;
                }
                let health = pawn.get_i64(Some(health_h));
                let raw_handle = pawn.get_handle(Some(ctrl_h));
                match raw_handle {
                    None => {
                        println!(
                            "pawn idx={idx} team={team} health={health} m_hController=ABSENT (field not in fields map)"
                        );
                    }
                    Some(h) => {
                        let looked_up = ctx.entities().get_by_handle(h);
                        match looked_up {
                            None => println!(
                                "pawn idx={idx} team={team} health={health} m_hController=0x{h:x} -> no entity at that index"
                            ),
                            Some(ctrl_entity) => {
                                let sid = ctrl_entity.get_u64(Some(sid_h));
                                println!(
                                    "pawn idx={idx} team={team} health={health} m_hController=0x{h:x} -> entity idx={} active={} class={} steamid={sid:?}",
                                    ctrl_entity.index, ctrl_entity.active, ctrl_entity.class_name
                                );
                            }
                        }
                    }
                }
            }
        })
        .expect("decode");
    if !done {
        println!("tick {target_tick} never reached");
    }
}
