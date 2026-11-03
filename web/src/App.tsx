import { useMutation, useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'
import { fetchMeta, runBacktest } from './api/client'
import type { BacktestResponse } from './api/types'
import { Controls } from './components/Controls'
import { EquityChart } from './components/EquityChart'
import { RollingSharpeChart } from './components/RollingSharpeChart'
import { StatsStrip } from './components/StatsStrip'
import { WeightStream } from './components/WeightStream'
import styles from './App.module.css'

export default function App() {
  const metaQuery = useQuery({ queryKey: ['meta'], queryFn: fetchMeta })
  const [start, setStart] = useState('2012-01-01')
  const [end, setEnd] = useState('2012-12-31')
  const [activePreset, setActivePreset] = useState<string | null>('2012')
  const [hoverIndex, setHoverIndex] = useState<number | null>(null)
  const [result, setResult] = useState<BacktestResponse | null>(null)

  useEffect(() => {
    if (!metaQuery.data) return
    const preset = metaQuery.data.presets['2012']
    if (preset) {
      setStart(preset.start)
      setEnd(preset.end)
    }
  }, [metaQuery.data])

  const mutation = useMutation({
    mutationFn: () => runBacktest(start, end),
    onSuccess: (data) => setResult(data),
  })

  useEffect(() => {
    if (metaQuery.isSuccess && !result && !mutation.isPending && !mutation.isError) {
      mutation.mutate()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [metaQuery.isSuccess])

  const presets = metaQuery.data?.presets ?? {
    '2012': { start: '2012-01-01', end: '2012-12-31' },
    '2020': { start: '2020-01-01', end: '2020-12-31' },
    'Full OOS': { start: '2017-01-01', end: '2021-12-31' },
  }

  const statusLine = useMemo(() => {
    if (!result) return 'Select a range and run a live historical backtest.'
    const modelNote =
      result.drl_status === 'ok'
        ? `PPO window ${result.window_index ?? '—'} · ${result.model_paths.length} checkpoint(s)`
        : 'PPO unavailable for this range — showing MVO + S&P 500'
    return `${result.dates.length} sessions · ${modelNote}`
  }, [result])

  return (
    <div className={styles.shell}>
      <div className={styles.hero}>
        <Controls
          start={start}
          end={end}
          presets={presets}
          activePreset={activePreset}
          loading={mutation.isPending}
          onStartChange={(v) => {
            setActivePreset(null)
            setStart(v)
          }}
          onEndChange={(v) => {
            setActivePreset(null)
            setEnd(v)
          }}
          onPreset={(name) => {
            const p = presets[name]
            if (!p) return
            setActivePreset(name)
            setStart(p.start)
            setEnd(p.end)
          }}
          onRun={() => mutation.mutate()}
        />
        <p className={styles.status}>{statusLine}</p>
        {mutation.isError ? (
          <p className={styles.error}>{(mutation.error as Error).message}</p>
        ) : null}
        {metaQuery.isError ? (
          <p className={styles.error}>API meta failed — is the FastAPI server running on :8000?</p>
        ) : null}

        {result ? (
          <div className={styles.main}>
            <EquityChart data={result} hoverIndex={hoverIndex} onHoverIndex={setHoverIndex} />
            <StatsStrip data={result} />
            <RollingSharpeChart data={result} />
            <WeightStream data={result} hoverIndex={hoverIndex} />
          </div>
        ) : null}
      </div>
    </div>
  )
}
