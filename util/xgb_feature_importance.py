"""XGBoost feature-importance report.

The tree pipeline trains 24 models (one per forecast hour). This aggregates their
importances and reports them at the level that is actually readable, because the
raw feature space is ~3.7k columns (22 lookback features x 168 lags + statics):

  1. BY FAMILY   — collapse the 168 lag columns of each base feature into one number
                   (e.g. all `temp_f_h*` -> Temp_F). This is the headline view.
  2. STATIC ONLY — the broadcast features (calendar / macro / thermal), ranked. These
                   are the ones we recently added, so they get their own table.
  3. TOP COLUMNS — the individual highest-gain columns (tells you WHICH lag matters).
  4. BY LAG      — importance summed per lookback position h0..h167 (h0 = oldest,
                   h167 = the hour just before cutoff).

Importance is summed over the 24 hourly models. 'gain' (default) is the right metric
for "how much did this feature actually improve the splits"; 'weight' just counts
splits and over-rewards high-cardinality columns.

Usage
  PJM_DATASET=dom python util/xgb_feature_importance.py
  PJM_DATASET=dom python util/xgb_feature_importance.py --importance weight --top 40
  python util/xgb_feature_importance.py --model-path models/dom/xgboost/tail_test0.1/xgboost_24_models.pkl

Prints tables and writes CSVs + a bar chart to results/<dataset>/diagnostics/.
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as cfg

pd.set_option('display.width', 200)

_LAG_RE = re.compile(r'_h(\d+)$')          # trailing "_h<k>" = flattened lookback column


def default_model_path():
    tf = cfg.TREE_FEATURE_CONFIG
    tag = f"{tf['split_strategy']}_test{tf['test_frac']}{cfg._xgb_lds}"
    return os.path.join('models', cfg.DATASET, 'xgboost', tag, 'xgboost_24_models.pkl')


def _split_name(col):
    """'temp_ma_24h_h37' -> ('temp_ma_24h', 37);  'heatwave_day_count' -> (name, None)."""
    m = _LAG_RE.search(col)
    if m:
        return col[:m.start()], int(m.group(1))
    return col, None


def collect_importance(model_path, importance_type='gain'):
    """Sum importance over the 24 hourly models -> Series indexed by feature column."""
    models = joblib.load(model_path)
    print(f"Loaded {len(models)} hourly models from {model_path}")

    total = {}
    for h, model in enumerate(models):
        booster = model.get_booster()
        # get_score omits features never split on -> they are simply absent (=0)
        for feat, val in booster.get_score(importance_type=importance_type).items():
            total[feat] = total.get(feat, 0.0) + float(val)

    names = models[0].get_booster().feature_names
    s = pd.Series({n: total.get(n, 0.0) for n in names}, name=importance_type)
    used = int((s > 0).sum())
    print(f"Features: {len(s)} total | {used} actually used in splits "
          f"({100 * used / len(s):.1f}%) | importance='{importance_type}'\n")
    return s


def report(imp, top=30, out_dir=None):
    df = pd.DataFrame({'importance': imp})
    df[['family', 'lag']] = [ _split_name(c) for c in df.index ]
    df['pct'] = 100 * df['importance'] / df['importance'].sum()

    # 1) by family (collapse the 168 lags)
    fam = (df.groupby('family')
             .agg(importance=('importance', 'sum'),
                  pct=('pct', 'sum'),
                  n_cols=('importance', 'size'))
             .sort_values('importance', ascending=False))
    print("=== 1. Importance BY FAMILY (lookback lags collapsed) ===")
    print(fam.head(top).round(3).to_string())

    # 2) static (non-lag) features only
    static = df[df['lag'].isna()].sort_values('importance', ascending=False)
    print("\n=== 2. STATIC features only (calendar / macro / thermal) ===")
    print(static[['importance', 'pct']].round(3).to_string())

    # 3) top individual columns
    print(f"\n=== 3. TOP {top} individual columns ===")
    print(df.sort_values('importance', ascending=False)
            .head(top)[['importance', 'pct', 'family', 'lag']].round(3).to_string())

    # 4) by lag position
    lagged = df[df['lag'].notna()]
    by_lag = lagged.groupby('lag')['importance'].sum().sort_index()
    if len(by_lag):
        print("\n=== 4. BY LAG (h0 = oldest .. h167 = hour before cutoff) ===")
        print(f"  most important lags: "
              f"{', '.join(f'h{int(k)}({v:.0f})' for k, v in by_lag.nlargest(10).items())}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fam.to_csv(os.path.join(out_dir, 'xgb_importance_by_family.csv'))
        static[['importance', 'pct']].to_csv(os.path.join(out_dir, 'xgb_importance_static.csv'))
        df.sort_values('importance', ascending=False).to_csv(
            os.path.join(out_dir, 'xgb_importance_all_columns.csv'))

        fig, axes = plt.subplots(1, 2, figsize=(17, max(6, 0.32 * min(top, len(fam)))))
        f = fam.head(top).iloc[::-1]
        axes[0].barh(f.index, f['importance'], color='#4C72B0', edgecolor='black', alpha=0.85)
        axes[0].set_title(f'{cfg.DATASET} XGBoost — importance by family (top {top})')
        axes[0].set_xlabel(f'summed {imp.name} over 24 hourly models')
        axes[0].tick_params(labelsize=8)
        axes[0].grid(axis='x', linestyle='--', alpha=0.5)

        if len(by_lag):
            axes[1].plot(by_lag.index, by_lag.values, color='#C44E52', linewidth=1.2)
            axes[1].set_title('importance by lookback position')
            axes[1].set_xlabel('lag index (h0 = 168h ago  →  h167 = just before cutoff)')
            axes[1].grid(linestyle='--', alpha=0.5)

        plt.tight_layout()
        path = os.path.join(out_dir, 'xgb_feature_importance.png')
        fig.savefig(path, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f"\nSaved CSVs + chart to: {out_dir}")

    return fam, static, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', default=None, help='default: derived from PJM_DATASET + config')
    ap.add_argument('--importance', default='gain',
                    choices=['gain', 'total_gain', 'weight', 'cover', 'total_cover'])
    ap.add_argument('--top', type=int, default=30)
    ap.add_argument('--no-save', action='store_true')
    args = ap.parse_args()

    model_path = args.model_path or default_model_path()
    if not os.path.exists(model_path):
        raise SystemExit(f"Model not found: {model_path}\n"
                         f"(train it first, or pass --model-path)")

    imp = collect_importance(model_path, args.importance)
    out_dir = None if args.no_save else os.path.join('results', cfg.DATASET, 'diagnostics')
    report(imp, top=args.top, out_dir=out_dir)


if __name__ == '__main__':
    main()
