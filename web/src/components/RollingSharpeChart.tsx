import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { BacktestResponse } from '../api/types'
import styles from './RollingSharpeChart.module.css'

type Props = { data: BacktestResponse }

export function RollingSharpeChart({ data }: Props) {
  const rows = data.dates.map((date, i) => ({
    date,
    ppo: data.rolling_sharpe.ppo[i],
    mvo: data.rolling_sharpe.mvo[i],
  }))

  return (
    <section className={styles.panel}>
      <h2 className={styles.title}>Rolling Sharpe (63d)</h2>
      <div className={styles.chartWrap}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows}>
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
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="mvo" name="MVO" stroke="#f5c14a" dot={false} connectNulls />
            <Line type="monotone" dataKey="ppo" name="PPO" stroke="#2dd4bf" dot={false} connectNulls />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}
