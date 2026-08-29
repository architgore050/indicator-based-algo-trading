"""Pull Optuna trial results from PostgreSQL."""
import psycopg2
import pandas as pd
import json
from pathlib import Path

V6 = Path(r"C:\Users\gorea\Desktop\indicator based algo trading\indicator based algo trading\v6")

conn = psycopg2.connect(
    host="localhost", port=5432, user="postgres", password="postgres", database="optuna_db"
)


def get_top_trials(prefix, n=20):
    cur = conn.cursor()
    cur.execute("""
        SELECT s.study_name, t.trial_id, tv.value AS score, t.state,
               t.datetime_start, t.datetime_complete
        FROM trials t
        JOIN studies s USING(study_id)
        JOIN trial_values tv USING(trial_id)
        WHERE s.study_name LIKE %s
          AND tv.objective = 0
          AND tv.value IS NOT NULL
          AND t.state = 'COMPLETE'
        ORDER BY tv.value DESC
        LIMIT %s
    """, (prefix + "%", n))
    rows = cur.fetchall()
    cur.close()
    return rows


def get_trial_params(trial_id):
    cur = conn.cursor()
    cur.execute("""
        SELECT param_name, param_value
        FROM trial_params WHERE trial_id = %s
    """, (trial_id,))
    rows = cur.fetchall()
    cur.close()
    params = {}
    for name, val in rows:
        params[name] = float(val) if val is not None else None
    return params


def main():
    # Top VM trials
    print("=" * 80)
    print("TOP 20 VM TRIALS (from PostgreSQL optuna_db)")
    print("=" * 80)
    top_vm = get_top_trials("vm_discovery", 20)
    for study, tid, score, state, dt_start, dt_end in top_vm:
        params = get_trial_params(tid)
        # Only print key params + score
        key_params = {k: v for k, v in params.items()
                      if k in ["stop_loss_usd_per_oz", "take_profit_usd_per_oz",
                               "trailing_stop_usd_per_oz", "max_hold_bars",
                               "use_fast_stoch_exit", "use_slow_stoch_exit",
                               "confirmation_lookback_bars"]}
        print(f"  trial {tid}: score={score:.3f} ({dt_end.date()}) key_params={key_params}")

    # Top TV trials
    print("\n" + "=" * 80)
    print("TOP 20 TV TRIALS (from PostgreSQL optuna_db)")
    print("=" * 80)
    top_tv = get_top_trials("tv_discovery", 20)
    for study, tid, score, state, dt_start, dt_end in top_tv:
        params = get_trial_params(tid)
        key_params = {k: v for k, v in params.items()
                      if k in ["stop_loss_usd_per_oz", "take_profit_usd_per_oz",
                               "trailing_stop_usd_per_oz", "max_hold_bars",
                               "min_buy_confirmations", "min_sell_confirmations"]}
        print(f"  trial {tid}: score={score:.3f} ({dt_end.date()}) key_params={key_params}")

    # Score distribution by study: how many trials > 0 in each study?
    print("\n" + "=" * 80)
    print("DISTRIBUTION OF TRIAL SCORES BY STUDY")
    print("=" * 80)
    cur = conn.cursor()
    cur.execute("""
        SELECT s.study_name, COUNT(*) FILTER (WHERE tv.value > 0) AS positive,
               COUNT(*) FILTER (WHERE tv.value = 0) AS zero,
               COUNT(*) FILTER (WHERE tv.value < 0) AS negative,
               COUNT(*) AS total,
               MIN(tv.value) AS minv, MAX(tv.value) AS maxv,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tv.value) AS medianv
        FROM trials t JOIN studies s USING(study_id) JOIN trial_values tv USING(trial_id)
        WHERE s.study_name LIKE 'vm_discovery%' AND tv.objective = 0 AND tv.value IS NOT NULL
        GROUP BY s.study_name ORDER BY positive DESC LIMIT 30
    """)
    rows = cur.fetchall()
    print(f"\n{'Study':<30} {'+':>5} {'0':>5} {'-':>5} {'Total':>6} {'Min':>10} {'Max':>10} {'Median':>10}")
    for study, pos, zero, neg, total, minv, maxv, medianv in rows:
        print(f"{study:<30} {pos:>5} {zero:>5} {neg:>5} {total:>6} {minv:>10.2f} {maxv:>10.2f} {medianv:>10.2f}")

    # Same for TV
    cur.execute("""
        SELECT s.study_name, COUNT(*) FILTER (WHERE tv.value > 0) AS positive,
               COUNT(*) FILTER (WHERE tv.value = 0) AS zero,
               COUNT(*) FILTER (WHERE tv.value < 0) AS negative,
               COUNT(*) AS total,
               MIN(tv.value) AS minv, MAX(tv.value) AS maxv,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tv.value) AS medianv
        FROM trials t JOIN studies s USING(study_id) JOIN trial_values tv USING(trial_id)
        WHERE s.study_name LIKE 'tv_discovery%' AND tv.objective = 0 AND tv.value IS NOT NULL
        GROUP BY s.study_name ORDER BY positive DESC LIMIT 30
    """)
    rows = cur.fetchall()
    print(f"\n--- TV STUDIES ---")
    print(f"{'Study':<30} {'+':>5} {'0':>5} {'-':>5} {'Total':>6} {'Min':>10} {'Max':>10} {'Median':>10}")
    for study, pos, zero, neg, total, minv, maxv, medianv in rows:
        print(f"{study:<30} {pos:>5} {zero:>5} {neg:>5} {total:>6} {minv:>10.2f} {maxv:>10.2f} {medianv:>10.2f}")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()