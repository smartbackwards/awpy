use std::path::Path;

use anyhow::{Context, Result};
use colored::Colorize;

pub fn run(file: &Path, limit: Option<usize>, json: bool) -> Result<()> {
    let parser = awpy::Parser::from_file(file)
        .with_context(|| format!("failed to open {}", file.display()))?;
    let timeouts = parser.timeouts()?;

    let display_limit = limit.unwrap_or(timeouts.len());
    let shown = &timeouts[..display_limit.min(timeouts.len())];

    if json {
        println!("{}", serde_json::to_string_pretty(&shown)?);
        return Ok(());
    }

    println!(
        "{:<20} {:<10} {:<8} {:<8} {:<10} {}",
        "Side".bold(),
        "Type".bold(),
        "Start".bold(),
        "End".bold(),
        "Length".bold(),
        "Round".bold(),
    );
    println!("{}", "-".repeat(70));
    for t in shown {
        println!(
            "{:<20} {:<10} {:<8} {:<8} {:<10} {}",
            t.side.as_deref().unwrap_or("-"),
            t.kind,
            t.start_tick,
            t.end_tick
                .map(|e| e.to_string())
                .unwrap_or_else(|| "-".to_string()),
            t.remaining_at_start
                .map(|r| format!("{r:.1}s"))
                .unwrap_or_else(|| "-".to_string()),
            t.round_num,
        );
    }
    println!(
        "\n{} timeouts total{}",
        timeouts.len(),
        if display_limit < timeouts.len() {
            format!(" (showing first {display_limit})")
        } else {
            String::new()
        }
    );
    Ok(())
}
