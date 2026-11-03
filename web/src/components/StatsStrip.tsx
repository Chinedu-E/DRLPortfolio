import type { BacktestResponse } from '../api/types'
import styles from './StatsStrip.module.css'

function pick(metrics: Record<string, number>, keys: string[]): number | null {
  for (const k of keys) {
    if (k in metrics && Number.isFinite(metrics[k])) return metrics[k]
  }
  return null
}

function fmt(v: number | null, digits = 3): string {
  if (v == null || !Number.isFinite(v)) return '—'
  return v.toFixed(digits)
}

type Props = { data: BacktestResponse }

export function StatsStrip({ data }: Props) {
  const blocks = [
    { key: 'ppo', label: 'PPO', className: styles.ppo, metrics: data.metrics.ppo },
    { key: 'mvo', label: 'MVO', className: styles.mvo, metrics: data.metrics.mvo },
  ] as const

  return (
    <section className={styles.strip}>
      {blocks.map((b) => (
        <article key={b.key} className={`${styles.card} ${b.className}`}>
          <div className={styles.head}>
            <span className={styles.name}>{b.label}</span>
            <span className={styles.badge}>
              {b.key === 'ppo' ? data.drl_status : 'benchmark'}
            </span>
          </div>
          <div className={styles.grid}>
            <div>
              <span className={styles.metricLabel}>Sharpe</span>
              <div className={styles.metricValue}>
                {fmt(pick(b.metrics, ['Sharpe ratio', 'Sharpe Ratio']))}
              </div>
            </div>
            <div>
              <span className={styles.metricLabel}>Max DD</span>
              <div className={styles.metricValue}>
                {fmt(pick(b.metrics, ['Max drawdown', 'Max Drawdown']))}
              </div>
            </div>
            <div>
              <span className={styles.metricLabel}>Mean Δp_w</span>
              <div className={styles.metricValue}>
                {fmt(pick(b.metrics, ['Mean Δp_w', 'Mean delta_pw']))}
              </div>
            </div>
          </div>
        </article>
      ))}
    </section>
  )
}
