"""Find explicit timeout calls (``.tac`` / ``.tech``) in chat.

CS2 match plugins (get5, MatchZy, and similar) let team representatives call a
tactical or technical timeout by typing a chat command — conventionally
``.tac`` (tactical, per-team) or ``.tech`` (technical, admin-called). This is
a thin, literal filter over :attr:`Demo.chat <awpy._awpy.Demo.chat>`; it adds
no new Rust dataframe since ``chat`` already carries everything needed.

For the primary, field-based timeout detection (reliable, works on demos with
no chat at all), see :attr:`Demo.timeouts <awpy._awpy.Demo.timeouts>` instead.
This module is a corroborating, explicit signal for demos where chat happens
to survive.

**Chat is commonly stripped from GOTV/broadcast demo recordings** — the same
limitation documented for :attr:`Demo.chat <awpy._awpy.Demo.chat>` itself.
This scanner returns an empty frame (not an error) on any such demo; it's
mainly useful on client-recorded or LAN demos where chat is present.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

    from awpy._awpy import Demo

DEFAULT_PATTERNS: tuple[str, ...] = (".tac", ".tech")


def find_timeout_calls(demo: Demo, patterns: tuple[str, ...] = DEFAULT_PATTERNS) -> pl.DataFrame:
    """Scan ``demo.chat`` for timeout-call commands.

    Args:
        demo: A parsed :class:`~awpy._awpy.Demo`.
        patterns: Case-insensitive substrings to match against each chat
            message. Defaults to ``(".tac", ".tech")``.

    Returns:
        A DataFrame with ``tick``, ``entity_index``, ``name``, ``message``,
        ``channel`` (``demo.chat``'s own columns), and ``matched_pattern``
        (which of ``patterns`` matched). Empty — not an error — when the demo
        has no chat, which is common for GOTV/broadcast recordings.
    """
    import polars as pl

    chat = demo.chat
    if chat.height == 0 or not patterns:
        return chat.with_columns(pl.lit(None, dtype=pl.Utf8).alias("matched_pattern"))

    lowered = pl.col("message").str.to_lowercase()
    matched_pattern = pl.coalesce(
        [
            pl.when(lowered.str.contains(pattern.lower(), literal=True))
            .then(pl.lit(pattern))
            .otherwise(None)
            for pattern in patterns
        ]
    )
    return chat.with_columns(matched_pattern.alias("matched_pattern")).filter(
        pl.col("matched_pattern").is_not_null()
    )
