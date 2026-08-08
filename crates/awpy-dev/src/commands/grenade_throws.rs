use std::path::Path;

use anyhow::{Context, Result};
use colored::Colorize;

pub fn run(file: &Path, limit: Option<usize>, json: bool) -> Result<()> {
    let parser = awpy::Parser::from_file(file)
        .with_context(|| format!("failed to open {}", file.display()))?;
    let throws = parser.grenade_throws()?;

    let display_limit = limit.unwrap_or(throws.len());
    let shown = &throws[..display_limit.min(throws.len())];

    if json {
        println!("{}", serde_json::to_string_pretty(&shown)?);
        return Ok(());
    }

    println!(
        "{:<8} {:<10} {:<16} {:<8} {}",
        "Throw".bold(),
        "Type".bold(),
        "Thrower".bold(),
        "Land".bold(),
        "Precise".bold(),
    );
    println!("{}", "-".repeat(60));
    for t in shown {
        println!(
            "{:<8} {:<10} {:<16} {:<8} {}",
            t.throw_tick,
            t.grenade_type,
            t.thrower_name.as_deref().unwrap_or("?"),
            t.land_tick,
            t.land_is_precise,
        );
    }
    println!(
        "\n{} throws total{}",
        throws.len(),
        if display_limit < throws.len() {
            format!(" (showing first {display_limit})")
        } else {
            String::new()
        }
    );
    Ok(())
}
