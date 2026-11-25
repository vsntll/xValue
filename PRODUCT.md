# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Casual football fans and non-technical recruiters or hiring managers, in equal measure, arriving from a CV, a portfolio link, a GitHub profile or a shared link. Most spend a few minutes and do not want statistics in front of them: the site has to be enjoyable and plainly worded first, with the technical proof (held-out scores, the prediction ledger, the model write-up) one click away for the technical reviewers who look for it.

## Product Purpose

xValue is a football predictor for the big-5 top flights (England, Spain, Germany, Italy and France, 2020-21 to 2026-27, plus the cups those clubs play in). It combines a player market-value regression, a pre-match home/draw/away outcome model and a player Elo. The public site is its shop window. Success means a reviewer leaves convinced the work is real and rigorous: the models work, the claims are measured, and the reasoning is visible.

## Positioning

What makes xValue different, all of it true and checkable in the repo:

- **Market value model.** It predicts a player's value and puts it next to the listed Transfermarkt value, so the biggest bargains and the most overpriced players show up (R²(log) 0.89 on a 2024-26 holdout).
- **Player Elo with no market value in it.** Players are rated match by match only on opponent-adjusted, role-weighted output (attack, defence, passing), in club and international matches alike.
- **An honest track record.** Pre-match odds are frozen in an append-only ledger with a timestamp and model-version hash, then scored against results and the closing market, with a calibration diagram.
- **It shows its working.** How It Works runs the actual arithmetic on one real upcoming fixture and one real player.

## Operating Context

- The site is a single self-contained HTML page with its data embedded inline. There is no server and no fetch. `src/export_site_data.py` builds it from `site/template.html`, and it is published to `docs/` for GitHub Pages.
- A pipeline refreshes it twice a week (Mon/Thu): fixtures, stats, values and model scores. The headline numbers come from `data/processed/model_metrics.json`.
- Tabs: Players, Fixtures & Odds, Teams, Value Screen, Rankings, Match Sim, Streak, Track Record, How It Works. A league chip filter scopes the views.

## Capabilities and Constraints

- **Odds are a display format, not a betting product.** Probabilities are shown as odds only because that's an easy format to read. The site doesn't promote wagering and doesn't present betting edges as the point. *Open decision:* the author welcomes another way to present outcome probabilities (for example percentages or probability bars).
- Club crests are generated from the club name (a hashed colour plus initials), so there are no external crest assets.
- Several data sources are unofficial and change availability (Understat has been down since late Sept 2026 and FotMob fills its gaps; Sofascore blocks CI). The site should handle missing data gracefully.
- Terminology: xG, xA, npxG, Elo, Dixon-Coles, 1X2, log-loss, R²(log), big-5.

## Brand Commitments

- The name is **xValue**.
- Voice: plain-spoken and candid about limits. The docs say outright what a model can't see (injuries, transfers, a manager getting sacked in November). Claims are always backed by measured numbers.

## Evidence on Hand

- The model scores in `data/processed/model_metrics.json`, which also feed the README headline.
- The forward-test ledger in `data/processed/predictions_ledger.csv`, plus the backtest in `betting_backtest.csv`.
- The architecture diagram `merged_predictor_architecture.png`, plus `docs/models.md` and `docs/modeling_decisions.md`.
- There are no user testimonials, traffic figures or press coverage. Don't invent any.

## Product Principles

1. **Show the working.** Every headline number should be traceable to how it was produced.
2. **Measured over claimed.** Only numbers from the committed metrics and the ledger. No unmeasured superlatives.
3. **Honest about limits.** State what each model can't see, and keep misses visible in the track record.
4. **Legible to a non-fan.** A reviewer who doesn't follow football should still be able to tell what each view shows and why it's impressive.
