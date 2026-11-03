import { useMemo } from 'react'
import type { BacktestResponse } from '../api/types'
import styles from './WeightStream.module.css'

type Props = {
  data: BacktestResponse
  hoverIndex: number | null
}

const MAX_HEAT_COLS = 48

function shortLabel(name: string): string {
  if (name === 'Cash') return 'Cash'
  if (name === 'InformationTechnology') return 'Info Tech'
  if (name === 'ConsumerDiscretionary') return 'Cons. Disc.'
  if (name === 'ConsumerStaples') return 'Cons. Staples'
  if (name === 'CommunicationServices') return 'Comm. Serv.'
  if (name === 'HealthCare') return 'Health Care'
  if (name === 'RealEstate') return 'Real Estate'
  return name
}

/** Map weight to teal using the series max so equal ~9% portfolios still read clearly. */
function heatColor(weight: number, maxWeight: number): string {
  const denom = Math.max(maxWeight, 1e-6)
  const t = Math.max(0, Math.min(1, weight / denom))
  // Dark steel base → bright teal
  const r = Math.round(24 + t * (45 - 24))
  const g = Math.round(36 + t * (212 - 36))
  const b = Math.round(54 + t * (191 - 54))
  return `rgb(${r}, ${g}, ${b})`
}

function sampleIndices(n: number, maxCols: number): number[] {
  if (n <= 0) return []
  if (n <= maxCols) return Array.from({ length: n }, (_, i) => i)
  const out: number[] = []
  for (let c = 0; c < maxCols; c++) {
    out.push(Math.round((c * (n - 1)) / (maxCols - 1)))
  }
  return out
}

export function WeightStream({ data, hoverIndex }: Props) {
  const idx =
    hoverIndex != null && hoverIndex >= 0 && hoverIndex < data.dates.length
      ? hoverIndex
      : Math.max(0, data.dates.length - 1)

  const scrubDate = data.dates[idx] ?? '—'
  const dayWeights = data.weights[idx] ?? {}

  const ranked = useMemo(() => {
    return [...data.weight_columns]
      .map((col) => ({
        name: col,
        value: Number(dayWeights[col] ?? 0),
      }))
      .sort((a, b) => b.value - a.value)
  }, [data.weight_columns, dayWeights])

  const heat = useMemo(() => {
    const cols = sampleIndices(data.dates.length, MAX_HEAT_COLS)
    let maxWeight = 0
    for (const ci of cols) {
      const row = data.weights[ci]
      if (!row) continue
      for (const col of data.weight_columns) {
        maxWeight = Math.max(maxWeight, Number(row[col] ?? 0))
      }
    }
    return {
      cols,
      maxWeight: maxWeight > 0 ? maxWeight : 1,
      labels: cols.map((i) => data.dates[i] ?? ''),
    }
  }, [data.dates, data.weight_columns, data.weights])

  const activeColPos = heat.cols.findIndex((ci) => ci === idx)

  return (
    <section className={styles.panel} key={`${data.start}-${data.end}`}>
      <div className={styles.titleRow}>
        <h2 className={styles.title}>Portfolio weights</h2>
        <span className={styles.scrub}>{scrubDate}</span>
      </div>

      <div className={styles.layout}>
        <div className={styles.snapshot}>
          <h3 className={styles.subhead}>Allocation on selected day</h3>
          <div className={styles.bars}>
            {ranked.map((row) => (
              <div key={row.name} className={styles.barRow}>
                <span className={styles.barLabel} title={row.name}>
                  {shortLabel(row.name)}
                </span>
                <div className={styles.track}>
                  <div
                    className={`${styles.fill} ${row.name === 'Cash' ? styles.fillCash : ''}`}
                    style={{ width: `${Math.max(row.value * 100, row.value > 0 ? 1.5 : 0)}%` }}
                  />
                </div>
                <span className={styles.barValue}>{(row.value * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
          <p className={styles.hint}>Hover the equity chart to scrub this snapshot.</p>
        </div>

        <div className={styles.heatBlock}>
          <h3 className={styles.subhead}>
            Weight intensity over time
            <span className={styles.scaleNote}>
              {' '}
              · scaled to max {(heat.maxWeight * 100).toFixed(0)}%
            </span>
          </h3>

          <div className={styles.heatScroll}>
            <div className={styles.heatTable}>
              <div
                className={styles.heatHeader}
                style={{
                  gridTemplateColumns: `7.2rem repeat(${heat.cols.length}, minmax(12px, 1fr))`,
                }}
              >
                <div />
                {heat.cols.map((ci, j) => (
                  <div
                    key={`c-${ci}`}
                    className={`${styles.colLabel} ${j === activeColPos ? styles.colActive : ''}`}
                  >
                    {j === 0 || j === heat.cols.length - 1 || j % 8 === 0
                      ? heat.labels[j]?.slice(5, 10)
                      : ''}
                  </div>
                ))}
              </div>

              {data.weight_columns.map((col) => (
                <div
                  key={col}
                  className={styles.heatRow}
                  style={{
                    gridTemplateColumns: `7.2rem repeat(${heat.cols.length}, minmax(12px, 1fr))`,
                  }}
                >
                  <div className={styles.rowLabel} title={col}>
                    {shortLabel(col)}
                  </div>
                  {heat.cols.map((ci, j) => {
                    const w = Number(data.weights[ci]?.[col] ?? 0)
                    return (
                      <div
                        key={`${col}-${ci}`}
                        className={`${styles.cell} ${j === activeColPos ? styles.cellActive : ''}`}
                        title={`${data.dates[ci]} · ${col}: ${(w * 100).toFixed(1)}%`}
                        style={{ background: heatColor(w, heat.maxWeight) }}
                      />
                    )
                  })}
                </div>
              ))}
            </div>
          </div>

          <div className={styles.legend}>
            <span>Low</span>
            <div className={styles.legendBar} />
            <span>High</span>
          </div>
        </div>
      </div>
    </section>
  )
}
