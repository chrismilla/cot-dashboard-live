# COT Dashboard Live

Public dashboard for **weekly COT bias + intraday market pulse** across:

- Dow Jones (`DIA`)
- Nasdaq-100 (`QQQ`)
- S&P 500 (`SPY`)

Live site: https://chrismilla.github.io/cot-dashboard-live/

## What it shows

- Weekly COT bias score (bullish/bearish/neutral)
- Bias component breakdown (institutional pressure, fast-money pressure, momentum, alignment)
- Trade playbook bullets (directional framing for next 1-5 sessions)
- Intraday pulse from 5-minute bars (price change, range position, session structure)
- Forex Factory red-folder calendar (high-impact events, USD-only or all)
- COT positioning history chart and latest report table

## Data sources

- CFTC TFF disaggregated report (weekly)
- Yahoo Finance chart API (5-minute bars, intraday)
- Forex Factory weekly XML calendar (high-impact events)

## Automation

GitHub Actions workflow: `.github/workflows/refresh-data.yml`

Schedules:
- Hourly during weekday US session window (`5 13-22 * * 1-5` UTC)
- Weekly post-CFTC refresh (`20 22 * * 5` UTC)

The workflow updates `data/cot_snapshot.json` and pushes only when data changed.
