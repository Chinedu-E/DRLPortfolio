import styles from './Controls.module.css'

type Props = {
  start: string
  end: string
  presets: Record<string, { start: string; end: string }>
  activePreset: string | null
  loading: boolean
  onStartChange: (v: string) => void
  onEndChange: (v: string) => void
  onPreset: (name: string) => void
  onRun: () => void
}

export function Controls({
  start,
  end,
  presets,
  activePreset,
  loading,
  onStartChange,
  onEndChange,
  onPreset,
  onRun,
}: Props) {
  return (
    <header className={styles.bar}>
      <div className={styles.brandBlock}>
        <h1 className={styles.brand}>RLPortfolio</h1>
        <p className={styles.tagline}>Quant research console · PPO vs MVO vs S&amp;P 500</p>
      </div>
      <div className={styles.controls}>
        <label className={styles.field}>
          <span className={styles.label}>Start</span>
          <input
            className={styles.input}
            type="date"
            value={start}
            onChange={(e) => onStartChange(e.target.value)}
          />
        </label>
        <label className={styles.field}>
          <span className={styles.label}>End</span>
          <input
            className={styles.input}
            type="date"
            value={end}
            onChange={(e) => onEndChange(e.target.value)}
          />
        </label>
        <div className={styles.field}>
          <span className={styles.label}>Presets</span>
          <div className={styles.presets}>
            {Object.keys(presets).map((name) => (
              <button
                key={name}
                type="button"
                className={`${styles.chip} ${activePreset === name ? styles.chipActive : ''}`}
                onClick={() => onPreset(name)}
              >
                {name}
              </button>
            ))}
          </div>
        </div>
        <button type="button" className={styles.run} onClick={onRun} disabled={loading}>
          {loading ? 'Running…' : 'Run backtest'}
        </button>
      </div>
    </header>
  )
}
