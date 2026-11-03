import type { BacktestResponse, MetaResponse } from './types'

async function parseError(res: Response): Promise<string> {
  try {
    const data = await res.json()
    if (typeof data?.detail === 'string') return data.detail
    if (Array.isArray(data?.detail)) {
      return data.detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join('; ')
    }
    return JSON.stringify(data)
  } catch {
    return res.statusText || 'Request failed'
  }
}

export async function fetchMeta(): Promise<MetaResponse> {
  const res = await fetch('/api/meta')
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

export async function runBacktest(start: string, end: string): Promise<BacktestResponse> {
  const res = await fetch('/api/backtest', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ start, end }),
  })
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}
