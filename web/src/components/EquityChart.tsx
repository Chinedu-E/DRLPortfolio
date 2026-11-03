import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { BacktestResponse } from '../api/types'
import styles from './EquityChart.module.css'

type Props = {
  data: BacktestResponse
  hoverIndex: number | null
  onHoverIndex: (i: number | null) => void
}

export function EquityChart({ data, hoverIndex, onHoverIndex }: Props) {
  const rows = data.dates.map((date, i) => ({
    date,
    ppo: data.equity.ppo[i],
    mvo: data.equity.mvo[i],
    spx: data.equity.spx[i],
  }))

  return (
    <section className={styles.panel}>
      <h2 className={styles.title}>Normalized equity</h2>
      <div className={styles.chartWrap}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={rows}
            onMouseMove={(state) => {
              // Recharts 3 may expose activeTooltipIndex as string, or only activeLabel.
              const raw = state?.activeTooltipIndex
              if (typeof raw === 'number' && Number.isFinite(raw)) {
                onHoverIndex(raw)
                return
              }
              if (typeof raw === 'string' && raw !== '' && !Number.isNaN(Number(raw))) {
                onHoverIndex(Number(raw))
                return
              }
              const label = state?.activeLabel
              if (typeof label === 'string') {
                const found = rows.findIndex((r) => r.date === label)
                onHoverIndex(found >= 0 ? found : null)
                return
              }
            }}
            onMouseLeave={() => onHoverIndex(null)}
          >
            <CartesianGrid stroke="rgba(148,163,184,0.12)" vertical={false} />
            <XAxis
              dataKey="date"
              tick={{ fill: '#8b9bb4', fontSize: 11, fontFamily: 'IBM Plex Mono' }}
              minTickGap={48}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: '#8b9bb4', fontSize: 11, fontFamily: 'IBM Plex Mono' }}
              axisLine={false}
              tickLine={false}
              domain={['auto', 'auto']}
              width={48}
            />
            <Tooltip
              contentStyle={{
                background: '#121a2b',
                border: '1px solid rgba(148,163,184,0.2)',
                borderRadius: 8,
                fontFamily: 'IBM Plex Mono',
                fontSize: 12,
              }}
              labelStyle={{ color: '#8b9bb4' }}
              content={(props) => {
                const label = props?.label
                if (typeof label === 'string' && props?.active) {
                  const found = rows.findIndex((r) => r.date === label)
                  if (found >= 0 && found !== hoverIndex) {
                    // Keep scrubber in sync even when mouse-move index is missing.
                    queueMicrotask(() => onHoverIndex(found))
                  }
                }
                if (!props?.active || !props?.payload?.length) return null
                return (
                  <div
                    style={{
                      background: '#121a2b',
                      border: '1px solid rgba(148,163,184,0.2)',
                      borderRadius: 8,
                      padding: '8px 10px',
                      fontFamily: 'IBM Plex Mono',
                      fontSize: 12,
                    }}
                  >
                    <div style={{ color: '#8b9bb4', marginBottom: 4 }}>{String(label ?? '')}</div>
                    {props.payload.map((p) => (
                      <div key={String(p.dataKey)} style={{ color: p.color }}>
                        {p.name}:{' '}
                        {typeof p.value === 'number' ? p.value.toFixed(3) : String(p.value)}
                      </div>
                    ))}
                  </div>
                )
              }}
            />
            <Legend
              wrapperStyle={{ fontFamily: 'IBM Plex Sans', fontSize: 12, color: '#8b9bb4' }}
            />
            {data.regime_hints.map((band) => (
              <ReferenceArea
                key={`${band.start}-${band.end}`}
                x1={band.start}
                x2={band.end}
                fill="rgba(251, 113, 133, 0.12)"
                strokeOpacity={0}
              />
            ))}
            <Area
              type="monotone"
              dataKey="spx"
              name="S&P 500"
              stroke="#94a3b8"
              fill="rgba(148,163,184,0.08)"
              strokeWidth={1.5}
              dot={false}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="mvo"
              name="MVO"
              stroke="#f5c14a"
              strokeWidth={2}
              dot={false}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="ppo"
              name="PPO"
              stroke="#2dd4bf"
              strokeWidth={2.25}
              dot={false}
              connectNulls
            />
            {hoverIndex != null && rows[hoverIndex] ? (
              <ReferenceArea
                x1={rows[hoverIndex].date}
                x2={rows[hoverIndex].date}
                stroke="#2dd4bf"
                strokeOpacity={0.35}
                fillOpacity={0}
              />
            ) : null}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}
