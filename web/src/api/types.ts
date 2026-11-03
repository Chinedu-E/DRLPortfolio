export type MetaResponse = {
  min_date: string
  max_date: string
  sectors: string[]
  windows: Array<{
    index: number
    train_start: string
    val_start: string
    test_start: string
    test_end: string
  }>
  models: string[]
  presets: Record<string, { start: string; end: string }>
}

export type BacktestResponse = {
  start: string
  end: string
  asset_names: string[]
  dates: string[]
  equity: {
    ppo: Array<number | null>
    mvo: Array<number | null>
    spx: Array<number | null>
  }
  rolling_sharpe: {
    ppo: Array<number | null>
    mvo: Array<number | null>
  }
  weights: Array<Record<string, number>>
  weight_columns: string[]
  metrics: {
    ppo: Record<string, number>
    mvo: Record<string, number>
    spx: Record<string, number>
  }
  regime_hints: Array<{ start: string; end: string }>
  drl_status: string
  model_paths: string[]
  window_index: number | null
}
