import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error


# ---------------------------------------------------------------------------
# DST helpers
# ---------------------------------------------------------------------------

def _dst_type(date):
    """Return 'spring_forward', 'fall_back', or None for a given EPT date."""
    start = pd.Timestamp(date).tz_localize('America/New_York')
    end   = pd.Timestamp(date + pd.Timedelta(days=1)).tz_localize('America/New_York')
    hours = int((end - start).total_seconds() / 3600)
    if hours == 23: return 'spring_forward'
    if hours == 25: return 'fall_back'
    return None


def _restore_dst_hours(true_24h, pred_24h, dst):
    """
    Undo the 24h normalization for plotting:
      spring_forward → remove interpolated 2AM → 23 points, x=[0,1,3,...,23]
      fall_back      → duplicate averaged 1AM  → 25 points, x_labels=[0,1,1,2,...,23]
    Returns (true_plot, pred_plot, x_positions, x_labels, dst_note).
    """
    if dst == 'spring_forward':
        idx = [i for i in range(24) if i != 2]
        x_labels = [str(h) for h in ([0, 1] + list(range(3, 24)))]
        return (true_24h[idx], pred_24h[idx],
                range(23), x_labels, ' [Spring Fwd]')

    if dst == 'fall_back':
        true_plot = np.insert(true_24h, 2, true_24h[1])
        pred_plot = np.insert(pred_24h, 2, pred_24h[1])
        x_labels  = ['0', '1', '1*'] + [str(h) for h in range(2, 24)]
        return (true_plot, pred_plot,
                range(25), x_labels, ' [Fall Back, *=2nd 1AM]')

    return true_24h, pred_24h, range(24), [str(h) for h in range(24)], ''


# ---------------------------------------------------------------------------
# Standalone single-day plot (public, also used by joint evaluator)
# ---------------------------------------------------------------------------

def plot_single_day(model_name, date_str, true_24h, pred_24h, ax=None, pred_color='#C44E52', save_path=None):
    """
    Plot true vs predicted load for one day.
    DST transition days are automatically detected and plotted with the correct
    number of EPT hours (23 for spring-forward, 25 for fall-back).

    ax=None  → creates its own figure and saves to save_path
    ax=<Axes> → draws into that axes (for embedding in subplots, save_path ignored)
    """
    dst  = _dst_type(pd.Timestamp(date_str).date())
    true_plot, pred_plot, x_pos, x_labels, dst_note = _restore_dst_hours(
        np.asarray(true_24h, dtype=float),
        np.asarray(pred_24h, dtype=float),
        dst,
    )

    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(x_pos, true_plot, label='True', color='#4C72B0', linewidth=2)
    ax.plot(x_pos, pred_plot, label='Pred', color=pred_color, linewidth=2, linestyle='--')
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.set_xlabel('Hour (EPT)')
    ax.set_ylabel('Load (MW)')
    ax.legend()
    ax.grid(linestyle='--', alpha=0.6)

    if standalone:
        if save_path is None:
            raise ValueError("save_path is required when calling plot_single_day without an ax.")
        mape = mean_absolute_percentage_error(true_24h, pred_24h) * 100
        ax.set_title(f'{model_name} | {date_str}{dst_note} | MAPE: {mape:.2f}%', fontsize=12)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Single-day plot saved to: {save_path}")


# ---------------------------------------------------------------------------
# EvalUtils — stateless utility class for metrics and plots
# ---------------------------------------------------------------------------

class EvalUtils:

    @staticmethod
    def build_detailed_df(model_name, y_true_np, y_pred_np, timestamps):
        dates = pd.to_datetime(timestamps).repeat(24)
        hours_arr = np.tile(np.arange(24), len(timestamps))
        datetimes = dates + pd.to_timedelta(hours_arr, unit='h')
        df = pd.DataFrame({
            'datetime': pd.to_datetime(datetimes),
            'true_load': y_true_np.flatten(),
            f'{model_name}_pred': y_pred_np.flatten(),
        })
        df['signed_error'] = df[f'{model_name}_pred'] - df['true_load']
        df['abs_error'] = df['signed_error'].abs()
        df['mape_pct'] = df['abs_error'] / df['true_load'] * 100
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_month'] = df['datetime'].dt.day
        df['date'] = df['datetime'].dt.date
        return df

    @staticmethod
    def plot_error_distributions(model_name, hourly_mape, dow_mape, dom_mape, result_dir):
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

    @staticmethod
    def plot_best_worst_days(model_name, detailed_df, daily_mape, result_dir):
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

    @staticmethod
    def plot_error_histogram(model_name, detailed_df, result_dir):
        signed_err = detailed_df['signed_error'].values
        signed_pct = (detailed_df['signed_error'] / detailed_df['true_load'] * 100).values

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{model_name} — Error Distribution  (+) = over-predict', fontsize=14, fontweight='bold')

        for ax, vals, label, color in [
            (axes[0], signed_err, 'Signed Error (MW)', '#4C72B0'),
            (axes[1], signed_pct, 'Signed Percentage Error (%)', '#C44E52'),
        ]:
            ax.hist(vals, bins=50, color=color, edgecolor='black', alpha=0.8)
            ax.axvline(0, color='black', linewidth=1.2, linestyle='--')
            ax.set_xlabel(label)
            ax.set_ylabel('Count')
            ax.set_title(label, fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        save_path = os.path.join(result_dir, f'{model_name}_error_histogram.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Error histogram saved to: {save_path}")

    @staticmethod
    def plot_error_cdf(model_name, detailed_df, result_dir):
        signed_err = detailed_df['signed_error'].values
        signed_pct = (detailed_df['signed_error'] / detailed_df['true_load'] * 100).values

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{model_name} — Error CDF  (+) = over-predict', fontsize=14, fontweight='bold')

        for ax, vals, label, color in [
            (axes[0], signed_err, 'Signed Error (MW)', '#4C72B0'),
            (axes[1], signed_pct, 'Signed Percentage Error (%)', '#C44E52'),
        ]:
            sorted_vals = np.sort(vals)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) * 100
            ax.plot(sorted_vals, cdf, color=color, linewidth=2)
            ax.axvline(0, color='black', linewidth=1.2, linestyle='--', label='Zero bias')
            ax.set_xlabel(label)
            ax.set_ylabel('Cumulative % of Hours')
            ax.set_title(f'CDF of {label}', fontsize=12)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
            ax.grid(linestyle='--', alpha=0.7)

            for pct in [10, 50, 90]:
                threshold = np.percentile(sorted_vals, pct)
                ax.axvline(threshold, linestyle=':', linewidth=1, alpha=0.7,
                           label=f'P{pct}: {threshold:+.2f}')
            ax.legend(fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(result_dir, f'{model_name}_error_cdf.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Error CDF saved to: {save_path}")

    @staticmethod
    def plot_signed_error_vs_load(model_name, detailed_df, result_dir, train_df=None):
        rows = [('Test', detailed_df)] + ([('Train', train_df)] if train_df is not None else [])
        n_rows = len(rows)
        fig, axes_grid = plt.subplots(n_rows, 2, figsize=(16, 5 * n_rows), squeeze=False)
        fig.suptitle(f'{model_name} — Signed Error vs Load  (+) = over-predict',
                     fontsize=14, fontweight='bold')

        for row_idx, (split_label, df) in enumerate(rows):
            ax_sc, ax_bar = axes_grid[row_idx]

            true_load  = df['true_load'].values
            signed_err = df['signed_error'].values
            hours      = df['datetime'].dt.hour.values

            sc = ax_sc.scatter(true_load, signed_err, c=hours, cmap='twilight_shifted',
                               s=4, alpha=0.4, rasterized=True)
            ax_sc.axhline(0, color='black', linewidth=1.0, linestyle='--')
            fig.colorbar(sc, ax=ax_sc).set_label('Hour of Day (EPT)')
            ax_sc.set_xlabel('True Load (MW)')
            ax_sc.set_ylabel('Signed Error (MW)\n(+) = over-predict')
            ax_sc.set_title(f'{split_label} — Scatter by Hour of Day')
            ax_sc.grid(linestyle='--', alpha=0.5)

            hourly_me = df.groupby(df['datetime'].dt.hour)['signed_error'].mean()
            colors = ['#C44E52' if v > 0 else '#4C72B0' for v in hourly_me.values]
            ax_bar.bar(hourly_me.index, hourly_me.values, color=colors, edgecolor='black', alpha=0.8)
            ax_bar.axhline(0, color='black', linewidth=1.0)
            ax_bar.set_xlabel('Hour of Day (EPT)')
            ax_bar.set_ylabel('Mean Signed Error (MW)')
            ax_bar.set_title(f'{split_label} — Mean Bias by Hour')
            ax_bar.set_xticks(range(24))
            ax_bar.grid(axis='y', linestyle='--', alpha=0.6)

        plt.tight_layout()
        save_path = os.path.join(result_dir, f'{model_name}_signed_error_vs_load.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Signed error scatter saved to: {save_path}")

    @staticmethod
    def plot_residuals_vs_predicted(model_name, detailed_df, result_dir, train_df=None):
        pred_col = f'{model_name}_pred'
        rows = [('Test', detailed_df)] + ([('Train', train_df)] if train_df is not None else [])
        n_rows = len(rows)
        fig, axes_grid = plt.subplots(n_rows, 2, figsize=(16, 5 * n_rows), squeeze=False)
        fig.suptitle(f'{model_name} — Residuals vs Predicted  (+) = over-predict',
                     fontsize=14, fontweight='bold')

        for row_idx, (split_label, df) in enumerate(rows):
            ax_sc, ax_bar = axes_grid[row_idx]

            pred_vals = df[pred_col].values
            residuals = df['signed_error'].values
            hours     = df['datetime'].dt.hour.values

            sc = ax_sc.scatter(pred_vals, residuals, c=hours, cmap='twilight_shifted',
                               s=4, alpha=0.4, rasterized=True)
            ax_sc.axhline(0, color='black', linewidth=1.0, linestyle='--')
            fig.colorbar(sc, ax=ax_sc).set_label('Hour of Day (EPT)')
            ax_sc.set_xlabel('Predicted Load (MW)')
            ax_sc.set_ylabel('Residual (MW)\n(+) = over-predict')
            ax_sc.set_title(f'{split_label} — Scatter by Hour of Day')
            ax_sc.grid(linestyle='--', alpha=0.5)

            pred_decile = pd.qcut(df[pred_col], q=10, labels=False, duplicates='drop')
            bin_centers = df.groupby(pred_decile)[pred_col].mean()
            bin_resid   = df.groupby(pred_decile)['signed_error'].mean()
            colors = ['#C44E52' if v > 0 else '#4C72B0' for v in bin_resid.values]
            ax_bar.bar(range(len(bin_resid)), bin_resid.values, color=colors, edgecolor='black', alpha=0.8)
            ax_bar.axhline(0, color='black', linewidth=1.0)
            ax_bar.set_xticks(range(len(bin_centers)))
            ax_bar.set_xticklabels([f'{v/1000:.1f}k' for v in bin_centers.values], fontsize=8)
            ax_bar.set_xlabel('Predicted Load Decile (MW)')
            ax_bar.set_ylabel('Mean Residual (MW)')
            ax_bar.set_title(f'{split_label} — Mean Bias by Predicted-Value Decile')
            ax_bar.grid(axis='y', linestyle='--', alpha=0.6)

        plt.tight_layout()
        save_path = os.path.join(result_dir, f'{model_name}_residuals_vs_predicted.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Residuals vs predicted saved to: {save_path}")

    @staticmethod
    def evaluate_one(model_name, y_true_np, y_pred_np, timestamps, result_dir, train_df=None):
        os.makedirs(result_dir, exist_ok=True)
        print(f"\n====== {model_name} Error Analysis ======")

        flat_true = y_true_np.flatten()
        flat_pred = y_pred_np.flatten()
        mape = mean_absolute_percentage_error(flat_true, flat_pred) * 100
        mae  = mean_absolute_error(flat_true, flat_pred)
        rmse = np.sqrt(mean_squared_error(flat_true, flat_pred))
        me   = float(np.mean(flat_pred - flat_true))
        bias_dir = 'over' if me > 0 else 'under'
        print(f"Global -> MAPE: {mape:.2f}% | MAE: {mae:.2f} | RMSE: {rmse:.2f} | ME: {me:+.2f} MW ({bias_dir})")

        hourly_mape = [
            mean_absolute_percentage_error(y_true_np[:, h], y_pred_np[:, h]) * 100
            for h in range(24)
        ]
        detailed_df = EvalUtils.build_detailed_df(model_name, y_true_np, y_pred_np, timestamps)
        dow_mape   = detailed_df.groupby('day_of_week')['mape_pct'].mean()
        dom_mape   = detailed_df.groupby('day_of_month')['mape_pct'].mean()
        daily_mape = detailed_df.groupby('date')['mape_pct'].mean().sort_values(ascending=False)

        csv_path = os.path.join(result_dir, f'{model_name}_detailed_errors.csv')
        detailed_df.drop(columns=['date']).to_csv(csv_path, index=False)
        print(f"Detailed errors saved to: {csv_path}")

        EvalUtils.plot_error_distributions(model_name, hourly_mape, dow_mape, dom_mape, result_dir)
        EvalUtils.plot_error_histogram(model_name, detailed_df, result_dir)
        EvalUtils.plot_error_cdf(model_name, detailed_df, result_dir)
        EvalUtils.plot_signed_error_vs_load(model_name, detailed_df, result_dir, train_df)
        EvalUtils.plot_residuals_vs_predicted(model_name, detailed_df, result_dir, train_df)
        EvalUtils.plot_best_worst_days(model_name, detailed_df, daily_mape, result_dir)
