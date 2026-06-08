# Quant-research Goggle
Custom Brave re-rank: boosts academic/practitioner (SSRN/arXiv/Quantpedia/Fed/AQR/code),
downranks bot-vendor marketing (3commas/cryptohopper/arbitragescanner/...).
- Hosted (public gist): see .raw_url  | Wired: BRAVE_QUANT_GOGGLE in ~/.profile; adapter passes goggles_id.
- STATUS: built+hosted+wired; Brave indexes new goggles with latency -> activates within its processing window.
- Re-verify it's live: run agent/brave.py web_search on a hype query (e.g. "crypto arbitrage bot") with/without
  BRAVE_QUANT_GOGGLE and check that 3commas/arbitragescanner drop. Edit the gist to tune rules (stable raw URL).
