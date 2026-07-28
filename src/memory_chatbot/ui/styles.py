"""Small CSS layer for the Streamlit UI.

Only tweaks that Streamlit cannot express through its own API live here, and all
colours come from Streamlit theme variables so light and dark both work.
"""

from __future__ import annotations

CUSTOM_CSS = """
<style>
  /* Tighten the default page padding - the sidebar carries a lot here. */
  .block-container { padding-top: 2.2rem; padding-bottom: 5rem; max-width: 60rem; }

  /* Memory chips: a compact way to show what was recalled for a turn. */
  .mem-chip {
    display: inline-block;
    padding: 0.12rem 0.55rem;
    margin: 0.12rem 0.2rem 0.12rem 0;
    border-radius: 999px;
    font-size: 0.74rem;
    line-height: 1.5;
    border: 1px solid color-mix(in srgb, currentColor 22%, transparent);
    background: color-mix(in srgb, currentColor 8%, transparent);
    white-space: nowrap;
  }
  .mem-chip .mem-score { opacity: 0.6; font-variant-numeric: tabular-nums; }

  /* Category label above a group of stored facts in the sidebar. */
  .mem-cat {
    font-size: 0.68rem;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    opacity: 0.65;
    margin: 0.6rem 0 0.15rem;
  }
  .mem-fact { font-size: 0.83rem; line-height: 1.45; margin: 0.05rem 0; }
  .mem-fact b { font-weight: 600; }
  .mem-fact .mem-old { opacity: 0.55; font-size: 0.75rem; }

  /* Status line under the title. */
  .status-line { font-size: 0.8rem; opacity: 0.7; margin-top: -0.6rem; }
  .status-line code { font-size: 0.78rem; }
</style>
"""
