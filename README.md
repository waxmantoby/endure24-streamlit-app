# Endure24 Relay Monte Carlo

Local Streamlit app for simulating a 24-hour Endure24 relay using last year's lap data and this year's runner projections.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app includes the provided workbook at:

```text
data/endure_info_for_lake_dwellers_to_review.xlsx
```

You can also upload a replacement workbook in the app.

## Expected Excel Format

The preferred workbook has these sheets:

- `last_year_laps`
- `this_year_roster`

Expected `last_year_laps` columns, or close equivalents:

- `runner`
- `lap_number_for_runner`
- `lap_time`
- `lap_time_minutes`
- `start_time`
- `start_minute`
- `is_night`

Expected `this_year_roster` columns, or close equivalents:

- `runner`
- `projected_mean_minutes`
- `projected_sd_minutes`
- `running_order`
- `max_laps`
- `available`
- `fatigue_pct_per_lap`
- `night_penalty_pct`
- `notes`

The loader also supports the dashboard-style workbook supplied for this project. It extracts the lap table from the "Interesting Graphs" sheet and uses the default projected roster from the brief.

## Race Rules

- Race starts Saturday 12:00.
- A runner may start an official lap at or before Sunday 12:00.
- That lap counts if it finishes at or before Sunday 13:00.
- A lap starting after Sunday 12:00 does not count.
- A lap finishing after Sunday 13:00 does not count.

Internally:

```python
last_start_minute = 1440
final_cutoff_minute = 1500
```

## Max-Lap Caps

`max_laps` controls runner caps:

- blank means unlimited
- `4` means at most 4 completed laps
- `5` means at most 5 completed laps
- `0` means unavailable
- `available = FALSE` means unavailable

When a runner reaches their cap, the simulator skips them in the rotation and moves to the next eligible runner. If every runner is capped out, the simulation stops and records that the team ran out of eligible runners.

Default caps:

- Jared: 4 laps
- Becs: 4 laps
- Everyone else: unlimited

Becca and Becs are normalised to the same runner.

## Simulation Assumptions

The default lap-time model is a truncated normal distribution:

```python
run_minutes = projected_mean_minutes * fatigue_factor * night_factor + random_noise
```

Fatigue and night penalties are interpreted as percentages and default to `0.0`:

- `1.5` would mean 1.5% per previous completed lap by that runner
- `3.0` would mean a 3.0% night penalty

Transitions default to zero seconds, with optional random mistake delays.

## Interpreting Outputs

The Results Dashboard shows:

- expected official laps
- median, mode and percentile range
- probability of hitting the selected target
- probability of target + 1
- expected official laps, which is the primary competition objective
- probability of squeezing in a lap after Sunday 12:00
- probability of missing the Sunday 13:00 cutoff
- probability of running out of eligible runners
- expected runner contributions
- cap-hit probabilities
- likely final runner
- likely runners between Sunday 10:30 and Sunday 12:00
- a final lap number probability plot showing the discrete probability of each final official lap count

## Adjusting Runner Paces

Use the editable roster table:

- Set Toby's projected mean to `33.0`.
- Set Alex's projected mean to `34.0`.
- Alex's SD defaults to Toby's SD, but can be edited.
- Change any runner's fatigue or night penalty from `0.0` if you want a more conservative model.

## Comparing Capped and Uncapped Outcomes

Leave "Also run uncapped comparison" enabled when running the simulation. The app will show:

- expected lap count with caps
- expected lap count without caps
- estimated cap impact in laps
- probability that capped runners hit their cap

## Running Order Optimiser

The optimiser evaluates candidate orders. By default it ranks orders for the competition goal: complete as many official laps as possible.

Default ranking:

1. expected official laps
2. median official laps
3. 10th percentile official laps
4. probability of the reference target
5. lower risk of missing the Sunday 13:00 cutoff
6. lower risk of running out of eligible runners

You can also enter a target in "Target laps for route finder" and click "Find route for target" to get the best running order and estimated probability of reaching that lap count.

Reference-target ranking:

1. probability of reaching the reference target
2. expected official laps
3. probability of reference target + 1
4. lower risk of missing the Sunday 13:00 cutoff
5. lower risk of running out of eligible runners

Use locked positions only if a runner must stay in a specific slot. Leave every locked position blank for an unconstrained order search. Increase candidate orders and simulations per order for a more robust search.

## What-if Pace Overrides

The final section lets you test one or more runner pace changes without permanently editing the roster:

- tick `Apply` for each runner you want to override
- enter their what-if mean lap time
- run the recalculation
- compare expected laps, reference-target probability, and the final lap number probability plot
- optionally run the target route finder with the overridden paces

## Race-Day Mode

The `Race Day` tab turns the app into a live command centre:

- log actual laps with a manual form
- paste a lap table for bulk updates
- compare the pre-race forecast with an actual-adjusted forecast
- keep the planned order fixed while showing the next eligible runner
- download or upload a CSV backup

The race log uses these columns:

```text
lap_number, runner, start_minute, finish_minute, lap_duration_minutes, notes, official
```

Pasted tables can use race-clock text such as `Saturday 13:04`, `Sunday 00:12`, or elapsed minutes such as `724`.

## Google Sheets Sync

The simplest race-day setup is an editable Google Sheet owned by `waxmantoby@gmail.com`.

1. In the `Race Day` tab, download `Google Sheet template CSV`.
2. In Google Sheets, create a blank spreadsheet.
3. Import or paste the template headers.
4. Share the Sheet so the app can view it, or use `File > Share > Publish to web`.
5. Paste the Sheet link into `Editable Google Sheet link`.
6. Click `Load editable Google Sheet` whenever you want to refresh the app from the Sheet.

Use these columns:

```text
lap_number, runner, finish_time, lap_duration_minutes, notes
```

Fill either `finish_time` or `lap_duration_minutes` for each lap:

- `finish_time`: `Saturday 13:04`, `Sunday 00:12`, or elapsed minutes such as `724`
- `lap_duration_minutes`: `33.5`, `41.2`, etc.

This editable-Sheet mode is read-only from the app: you edit the Google Sheet, then reload it in Streamlit.

Google Sheets write-back is optional. Without credentials, the Race Day tab still works with editable-Sheet import and CSV backup.

Add these Streamlit secrets to enable sync:

```toml
google_sheet_id = "your-google-sheet-id"

[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
client_email = "..."
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."
```

Share the Google Sheet with the service account `client_email`. The app reads and writes a worksheet named `race_log`.

## Fatigue and Night Penalty Sources

The default fatigue and night penalties are zero:

- fatigue: 0.0% slower per previous completed lap by that runner
- night penalty: 0.0% slower for a lap that starts inside the configured night window

These are editable per runner in the app. Last-year stats show day/night and fatigue patterns where the data supports it, but the app does not apply those slowdowns automatically.
