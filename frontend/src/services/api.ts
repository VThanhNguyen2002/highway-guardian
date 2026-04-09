// frontend/src/services/api.ts
// Typed axios client consuming the Highway Guardian FastAPI backend.

import axios, { type AxiosInstance } from 'axios'

const BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'
const API_PREFIX = '/api/v1'

const client: AxiosInstance = axios.create({
  baseURL: `${BASE_URL}${API_PREFIX}`,
  timeout: 15_000,
  headers: { 'Content-Type': 'application/json' },
})

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

export interface TrendPoint {
  period: string
  count: number
}

export interface TrendResponse {
  granularity: string
  data: TrendPoint[]
}

export interface ValidityResponse {
  valid_count: number
  invalid_count: number
  total: number
  valid_ratio: number
}

export interface FrequencyPoint {
  class_id: number
  class_name: string
  count: number
}

export interface FrequencyResponse {
  top_n: number
  data: FrequencyPoint[]
}

export interface DetectionRecord {
  id: number
  timestamp: string
  image_path: string
  class_id: number
  class_name: string
  confidence: number
  is_valid: boolean
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

export async function fetchTrend(
  granularity: 'daily' | 'monthly' = 'daily',
  days = 30,
): Promise<TrendResponse> {
  const { data } = await client.get<TrendResponse>('/analytics/trend', {
    params: { granularity, days },
  })
  return data
}

export async function fetchValidity(days?: number): Promise<ValidityResponse> {
  const { data } = await client.get<ValidityResponse>('/analytics/validity', {
    params: days ? { days } : {},
  })
  return data
}

export async function fetchFrequency(topN = 10, days?: number): Promise<FrequencyResponse> {
  const { data } = await client.get<FrequencyResponse>('/analytics/frequency', {
    params: { top_n: topN, ...(days ? { days } : {}) },
  })
  return data
}

export async function fetchHistory(
  limit = 50,
  offset = 0,
  isValid?: boolean,
): Promise<{ records: DetectionRecord[]; total_returned: number }> {
  const { data } = await client.get('/history', {
    params: {
      limit,
      offset,
      ...(isValid !== undefined ? { is_valid: isValid } : {}),
    },
  })
  return data
}
