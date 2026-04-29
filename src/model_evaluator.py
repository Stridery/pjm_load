# src/model_evaluator.py
import os
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

from src.feature_engine import build_or_load_matrix, build_timeseries_matrix, _split_indices
from src.config import (
    CLEANED_PATH, MATRIX_DIR,
    TRANSFORMER_FEATURE_CONFIG, TRANSFORMER_PARAMS,
    LSTM_PARAMS,
)
from src.model_trainer import TimeSeriesTransformer3D, LSTMModel


# ---------------------------------------------------------------------------
# Standalone single-day plot function (public interface)
# ---------------------------------------------------------------------------

def plot_single_day(model_name, date_str, true_24h, pred_24h, ax=None, pred_color='#C44E52', save_path=None):
    """
    Plot 24-hour true vs predicted load for one day.

    ax=None  → creates its own figure and saves to save_path
    ax=<Axes> → draws into that axes (for embedding in subplots, save_path ignored)
    """
    hours = range(24)
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(hours, true_24h, label='True', color='#4C72B0', linewidth=2)
    ax.plot(hours, pred_24h, label='Pred', color=pred_color, linewidth=2, linestyle='--')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Load (MW)')
    ax.legend()
    ax.grid(linestyle='--', alpha=0.6)

    if standalone:
        if save_path is None:
            raise ValueError("save_path is required when calling plot_single_day without an ax.")
        mape = mean_absolute_percentage_error(true_24h, pred_24h) * 100
        ax.set_title(f'{model_name} | {date_str} | MAPE: {mape:.2f}%', fontsize=12)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Single-day plot saved to: {save_path}")


# ---------------------------------------------------------------------------
# ModelEvaluator
# ---------------------------------------------------------------------------

class ModelEvaluator:
    def __init__(self, eval_cfg):
        self.cfg = eval_cfg
        self.X_opt = self.y_opt = None
        self.X_3d = self.y_3d = self.mask_3d = self.timestamps_3d = self.scaler_ts = None

    # --- Data loading ---

    def _needs_tree(self):
        tree = ['xgboost', 'lightgbm']
        if any(self.cfg['models'][m]['enabled'] for m in tree):
            return True
        sd = self.cfg.get('single_day', {})
        return sd.get('enabled', 0) and sd.get('model') in tree

    def _needs_sequence(self):
        seq = ['transformer', 'lstm']
        if any(self.cfg['models'][m]['enabled'] for m in seq):
            return True
        sd = self.cfg.get('single_day', {})
        return sd.get('enabled', 0) and sd.get('model') in seq

    def load_data(self):
        if self._needs_tree():
            self.X_opt, self.y_opt = build_or_load_matrix(CLEANED_PATH, MATRIX_DIR)
        if self._needs_sequence():
            self.X_3d, self.y_3d, self.mask_3d, self.timestamps_3d = build_timeseries_matrix(
                CLEANED_PATH, MATRIX_DIR
            )
            _lb = TRANSFORMER_FEATURE_CONFIG['lookback_hours']
            _h  = TRANSFORMER_FEATURE_CONFIG['latest_info_hour']
            self.scaler_ts = joblib.load(
                os.path.join(MATRIX_DIR, f'scaler_ts_lb{_lb}_h{_h}.pkl')
            )

    # --- Test split helpers ---

    def _tree_test_split(self):
        _, test_pos = _split_indices(
            len(self.X_opt),
            self.cfg['split_strategy'],
            self.cfg['test_frac'],
            self.cfg['random_state'],
        )
        test_idx = self.X_opt.index[test_pos]
        X_test = self.X_opt.loc[test_idx].drop(columns=['is_target_valid'])
        y_test = self.y_opt.loc[test_idx]
        return X_test, y_test

    def _seq_test_split(self):
        _, test_pos = _split_indices(
            len(self.X_3d),
            self.cfg['split_strategy'],
            self.cfg['test_frac'],
            self.cfg['random_state'],
        )
        return self.X_3d[test_pos], self.y_3d[test_pos], self.timestamps_3d[test_pos]

    # --- Inference ---

    def _predict_tree(self, model_path, X):
        models = joblib.load(model_path)
        return np.array([m.predict(X) for m in models]).T  # (N, 24)

    def _predict_sequence(self, model_name, model_path, X_np):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_features = X_np.shape[2]
        if model_name == 'transformer':
            model = TimeSeriesTransformer3D(num_features=num_features, params=TRANSFORMER_PARAMS).to(device)
        else:
            model = LSTMModel(num_features=num_features, params=LSTM_PARAMS).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        with torch.no_grad():
            preds_scaled = model(torch.FloatTensor(X_np).to(device)).cpu().numpy()
        return self._inverse_transform(preds_scaled)

    def _inverse_transform(self, scaled_2d):
        N, P = scaled_2d.shape
        flat = scaled_2d.flatten()
        dummy = np.zeros((len(flat), self.scaler_ts.n_features_in_))
        dummy[:, 0] = flat
        return self.scaler_ts.inverse_transform(dummy)[:, 0].reshape(N, P)

    # --- Metrics & detailed df ---

    def _build_detailed_df(self, model_name, y_true_np, y_pred_np, timestamps):
        dates = pd.to_datetime(timestamps).repeat(24)
        hours_arr = np.tile(np.arange(24), len(timestamps))
        datetimes = dates + pd.to_timedelta(hours_arr, unit='h')
        df = pd.DataFrame({
            'datetime': pd.to_datetime(datetimes),
            'true_load': y_true_np.flatten(),
            f'{model_name}_pred': y_pred_np.flatten(),
        })
        df['abs_error'] = (df['true_load'] - df[f'{model_name}_pred']).abs()
        df['mape_pct'] = df['abs_error'] / df['true_load'] * 100
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_month'] = df['datetime'].dt.day
        df['date'] = df['datetime'].dt.date
        return df

    # --- Plots ---

    def _plot_error_distributions(self, model_name, hourly_mape, dow_mape, dom_mape, result_dir):
        plt.figure(figsize=(15, 12))

        plt.subplot(3, 1, 1)
        plt.bar(range(24), hourly_mape, color='#4C72B0', edgecolor='black', alpha=0.8)
        plt.title(f'{model_name} - Hourly MAPE (%)', fontsize=14, fontweight='bold')
        plt.xlabel('Hour of Day (0-23)', fontsize=12)
        plt.ylabel('MAPE (%)', fontsize=12)
        plt.xticks(range(24))
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.subplot(3, 1, 2)
        days_labels = ['Mon (0)', 'Tue (1)', 'Wed (2)', 'Thu (3)', 'Fri (4)', 'Sat (5)', 'Sun (6)']
        dow_values = [dow_mape.get(i, 0) for i in range(7)]
        plt.bar(days_labels, dow_values, color='#55A868', edgecolor='black', alpha=0.8)
        plt.title(f'{model_name} - Day of Week MAPE (%)', fontsize=14, fontweight='bold')
        plt.xlabel('Day of Week', fontsize=12)
        plt.ylabel('MAPE (%)', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.subplot(3, 1, 3)
        dom_idx = sorted(dom_mape.index)
        dom_values = [dom_mape[d] for d in dom_idx]
        plt.bar(dom_idx, dom_values, color='#C44E52', edgecolor='black', alpha=0.8)
        plt.title(f'{model_name} - Day of Month MAPE (%)', fontsize=14, fontweight='bold')
        plt.xlabel('Day of Month (1-31)', fontsize=12)
        plt.ylabel('MAPE (%)', fontsize=12)
        plt.xticks(range(1, 32))
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        save_path = os.path.join(result_dir, f'{model_name}_error_dashboard.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Error dashboard saved to: {save_path}")

    def _plot_best_worst_days(self, model_name, detailed_df, daily_mape, result_dir):
        worst_days = list(daily_mape.head(3).index)
        best_days  = list(daily_mape.tail(3).index[::-1])

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{model_name} — Best & Worst 3 Days', fontsize=16, fontweight='bold')

        for col, date in enumerate(worst_days):
            day_df = detailed_df[detailed_df['date'] == date].sort_values('datetime')
            plot_single_day(
                model_name, str(date),
                day_df['true_load'].values,
                day_df[f'{model_name}_pred'].values,
                ax=axes[0, col], pred_color='#C44E52',
            )
            axes[0, col].set_title(f'Worst #{col+1}: {date}\nMAPE: {daily_mape.loc[date]:.2f}%', fontsize=11)

        for col, date in enumerate(best_days):
            day_df = detailed_df[detailed_df['date'] == date].sort_values('datetime')
            plot_single_day(
                model_name, str(date),
                day_df['true_load'].values,
                day_df[f'{model_name}_pred'].values,
                ax=axes[1, col], pred_color='#55A868',
            )
            axes[1, col].set_title(f'Best #{col+1}: {date}\nMAPE: {daily_mape.loc[date]:.2f}%', fontsize=11)

        plt.tight_layout()
        save_path = os.path.join(result_dir, f'{model_name}_best_worst_days.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Best/worst day plots saved to: {save_path}")

    # --- Core single-model evaluation ---

    def _evaluate_one(self, model_name, y_true_np, y_pred_np, timestamps, result_dir):
        os.makedirs(result_dir, exist_ok=True)
        print(f"\n====== {model_name} Error Analysis ======")

        mape = mean_absolute_percentage_error(y_true_np.flatten(), y_pred_np.flatten()) * 100
        mae  = mean_absolute_error(y_true_np.flatten(), y_pred_np.flatten())
        rmse = np.sqrt(mean_squared_error(y_true_np.flatten(), y_pred_np.flatten()))
        print(f"Global -> MAPE: {mape:.2f}% | MAE: {mae:.2f} | RMSE: {rmse:.2f}")

        hourly_mape = [
            mean_absolute_percentage_error(y_true_np[:, h], y_pred_np[:, h]) * 100
            for h in range(24)
        ]
        detailed_df = self._build_detailed_df(model_name, y_true_np, y_pred_np, timestamps)
        dow_mape   = detailed_df.groupby('day_of_week')['mape_pct'].mean()
        dom_mape   = detailed_df.groupby('day_of_month')['mape_pct'].mean()
        daily_mape = detailed_df.groupby('date')['mape_pct'].mean().sort_values(ascending=False)

        csv_path = os.path.join(result_dir, f'{model_name}_detailed_errors.csv')
        detailed_df.drop(columns=['date']).to_csv(csv_path, index=False)
        print(f"Detailed errors saved to: {csv_path}")

        self._plot_error_distributions(model_name, hourly_mape, dow_mape, dom_mape, result_dir)
        self._plot_best_worst_days(model_name, detailed_df, daily_mape, result_dir)

    # --- Public API ---

    def evaluate_all(self):
        result_base = self.cfg.get('result_dir', 'results/evaluation')
        strategy    = self.cfg['split_strategy']

        tree_enabled = [m for m in ['xgboost', 'lightgbm'] if self.cfg['models'][m]['enabled']]
        if tree_enabled:
            X_test, y_test = self._tree_test_split()
            timestamps = pd.to_datetime(y_test.index)
            y_true_np  = y_test.values  # already in MW, no scaling

            for model_name in tree_enabled:
                print(f"\n====== Loading {model_name.upper()} ======")
                y_pred_np = self._predict_tree(
                    self.cfg['models'][model_name]['model_path'], X_test
                )
                result_dir = os.path.join(result_base, model_name, strategy)
                self._evaluate_one(model_name.upper(), y_true_np, y_pred_np, timestamps, result_dir)

        seq_enabled = [m for m in ['transformer', 'lstm'] if self.cfg['models'][m]['enabled']]
        if seq_enabled:
            X_te, y_te_scaled, test_timestamps = self._seq_test_split()
            y_true_mw = self._inverse_transform(y_te_scaled)

            for model_name in seq_enabled:
                print(f"\n====== Loading {model_name.upper()} ======")
                y_pred_mw = self._predict_sequence(
                    model_name,
                    self.cfg['models'][model_name]['model_path'],
                    X_te,
                )
                result_dir = os.path.join(result_base, model_name, strategy)
                self._evaluate_one(model_name.upper(), y_true_mw, y_pred_mw, test_timestamps, result_dir)

    def show_single_day(self, model_name, model_path, date_str):
        """
        Load a saved model, find `date_str` in the full matrix, run inference,
        and display an interactive 24-hour plot. Nothing is saved to disk.
        """

        if model_name in ['xgboost', 'lightgbm']:
            target = pd.to_datetime(date_str)
            if target not in self.X_opt.index:
                raise ValueError(f"{date_str} not found in 2D matrix index.")
            x_row    = self.X_opt.loc[[target]].drop(columns=['is_target_valid'])
            true_24h = self.y_opt.loc[target].values
            pred_24h = self._predict_tree(model_path, x_row).flatten()

        else:  # transformer / lstm
            target = pd.to_datetime(date_str)
            idx = np.where(pd.to_datetime(self.timestamps_3d) == target)[0]
            if len(idx) == 0:
                raise ValueError(f"{date_str} not found in 3D matrix timestamps.")
            pred_24h = self._predict_sequence(model_name, model_path, self.X_3d[idx]).flatten()
            true_24h = self._inverse_transform(self.y_3d[idx]).flatten()

        # model_path e.g. models/lstm/tail_test0.1_random_val0.1/lstm_best.pth
        # → save to results/singleday/lstm/tail_test0.1_random_val0.1/2023-08-15.png
        from pathlib import Path
        model_subdir = os.path.join(*Path(model_path).parts[1:-1])
        save_path = os.path.join('results', 'singleday', model_subdir, f'{date_str}.png')
        plot_single_day(model_name.upper(), date_str, true_24h, pred_24h, save_path=save_path)
