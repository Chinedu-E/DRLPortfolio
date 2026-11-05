import type { BacktestResponse, MetaResponse } from './types'

/** Production: full API origin (e.g. https://api.up.railway.app). Local: empty → Vite /api proxy. */
const API_BASE = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '') ?? ''

function apiUrl(path: string): string {
  return `${API_BASE}${path}`
}

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
  const res = await fetch(apiUrl('/api/meta'))
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

export async function runBacktest(start: string, end: string): Promise<BacktestResponse> {
  const res = await fetch(apiUrl('/api/backtest'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ start, end }),
  })
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}
