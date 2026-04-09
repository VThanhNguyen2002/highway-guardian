// frontend/src/stores/analytics.ts
// Pinia store for analytics dashboard state management.

import { defineStore } from 'pinia'
import { ref } from 'vue'
import {
  fetchFrequency,
  fetchTrend,
  fetchValidity,
  type FrequencyResponse,
  type TrendResponse,
  type ValidityResponse,
} from '@/services/api'

export const useAnalyticsStore = defineStore('analytics', () => {
  // State
  const trend = ref<TrendResponse | null>(null)
  const validity = ref<ValidityResponse | null>(null)
  const frequency = ref<FrequencyResponse | null>(null)
  const loading = ref<boolean>(false)
  const error = ref<string | null>(null)

  // Current filter params (reactive, shared across components)
  const granularity = ref<'daily' | 'monthly'>('daily')
  const lookbackDays = ref<number>(30)
  const topN = ref<number>(10)

  async function loadAll(): Promise<void> {
    loading.value = true
    error.value = null
    try {
      const [trendData, validityData, frequencyData] = await Promise.all([
        fetchTrend(granularity.value, lookbackDays.value),
        fetchValidity(lookbackDays.value),
        fetchFrequency(topN.value, lookbackDays.value),
      ])
      trend.value = trendData
      validity.value = validityData
      frequency.value = frequencyData
    } catch (e: unknown) {
      error.value = e instanceof Error ? e.message : 'Failed to load analytics data.'
    } finally {
      loading.value = false
    }
  }

  return {
    trend,
    validity,
    frequency,
    loading,
    error,
    granularity,
    lookbackDays,
    topN,
    loadAll,
  }
})
