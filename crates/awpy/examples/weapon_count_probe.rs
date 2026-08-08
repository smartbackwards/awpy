//! Demonstrates the fix that shipped: reading `m_pWeaponServices.m_iAmmo[14]`
//! (flashbang reserve ammo — the true 0/1/2 held count) for one player at one
//! tick, using the same "manually build a `FieldPath`" technique as
//! `reserve_ammo_probe.rs` — but here the field pans out, unlike that one.
//!
//! `m_iAmmo` is nested one level deeper than that example's `m_pReserveAmmo`
//! (`m_pWeaponServices.m_iAmmo[i]` vs a bare `m_pReserveAmmo[i]` on the
//! weapon entity itself), so this is also the reference example for a fixed
//! array reached *through* a pointer field: resolve the bare dotted path up
//! to (but not including) the array index — `resolve_field_key` handles that
//! part fine, since it's a plain nested-pointer traversal, not a fixed-array
//! index — then append the array index onto the resulting `FieldPath` by
//! hand. See `fill_loadout`'s doc comment in `datasets.rs` for how this is
//! actually wired into `PlayerState.flashbangs`.
//!
//! `cargo run --release --example weapon_count_probe -- <demo> <tick> <steamid>`

use std::collections::HashSet;
use std::path::PathBuf;

use awpy::Parser;
use awpy::entity::field_path::FieldPath;

const PLAYER_PAWN_CLASS: &str = "CCSPlayerPawn";
const PLAYER_CONTROLLER_CLASS: &str = "CCSPlayerController";
const FLASHBANG_AMMO_INDEX: u8 = 14;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().expect("demo path"));
    let target_tick: i32 = args.next().expect("tick").parse().expect("tick int");
    let target_steamid: u64 = args.next().expect("steamid").parse().expect("steamid int");

    let parser = Parser::from_file(&path).expect("open");
    let filter: HashSet<&str> = HashSet::from([PLAYER_PAWN_CLASS, PLAYER_CONTROLLER_CLASS]);
    let mut ammo_key: Option<u64> = None;
    let mut steamid_key: Option<u64> = None;
    let mut ctrl_key: Option<u64> = None;
    let mut found = false;

    parser
        .run_to_end_filtered(&filter, |ctx| {
            if found || ctx.tick() != target_tick {
                return;
            }
            let pser = ctx.serializers().get(PLAYER_PAWN_CLASS);
            let cser = ctx.serializers().get(PLAYER_CONTROLLER_CLASS);

            // The bare dotted path resolves fine -- no fixed-array index yet.
            let ammo = *ammo_key.get_or_insert_with(|| {
                let base = pser
                    .and_then(|s| s.resolve_field_key("m_pWeaponServices.m_iAmmo"))
                    .expect("m_iAmmo base key");
                let base_fp = FieldPath::unpack(base);
                let mut fp = FieldPath::default();
                fp.data[0] = base_fp.data[0];
                fp.data[1] = base_fp.data[1];
                fp.data[2] = FLASHBANG_AMMO_INDEX;
                fp.last = 2;
                fp.pack()
            });
            let ctrl_h = *ctrl_key.get_or_insert_with(|| {
                pser.and_then(|s| s.resolve_field_key("m_hController"))
                    .expect("ctrl key")
            });
            let sid = *steamid_key.get_or_insert_with(|| {
                cser.and_then(|s| s.resolve_field_key("m_steamID"))
                    .expect("steamid key")
            });

            for (_, pawn) in ctx.entities().iter() {
                if !pawn.active || pawn.class_name.as_ref() != PLAYER_PAWN_CLASS {
                    continue;
                }
                let Some(controller) = pawn
                    .get_handle(Some(ctrl_h))
                    .and_then(|h| ctx.entities().get_by_handle(h))
                else {
                    continue;
                };
                if controller.get_u64(Some(sid)) != Some(target_steamid) {
                    continue;
                }
                println!(
                    "tick={target_tick} entity_index={} flashbang_ammo={}",
                    pawn.index,
                    pawn.get_i64(Some(ammo))
                );
                found = true;
            }
        })
        .expect("decode");
    if !found {
        println!("tick={target_tick} steamid not found");
    }
}
