import { useCallback, useState } from 'react'
import type { Dispatch, SetStateAction } from 'react'

export function useSessionState<T>(
  key: string,
  initialValue: T,
): [T, Dispatch<SetStateAction<T>>] {
  const [entry, setEntry] = useState(() => ({
    key,
    value: readValue(key, initialValue),
  }))
  const value = entry.key === key ? entry.value : readValue(key, initialValue)
  const setValue = useCallback<Dispatch<SetStateAction<T>>>(
    (next) => {
      setEntry((current) => {
        const currentValue =
          current.key === key ? current.value : readValue(key, initialValue)
        const resolved =
          typeof next === 'function'
            ? (next as (previous: T) => T)(currentValue)
            : next
        window.sessionStorage.setItem(key, JSON.stringify(resolved))
        return { key, value: resolved }
      })
    },
    [initialValue, key],
  )
  return [value, setValue]
}

function readValue<T>(key: string, initialValue: T): T {
  const stored = window.sessionStorage.getItem(key)
  if (stored === null) return initialValue
  try {
    return JSON.parse(stored) as T
  } catch {
    window.sessionStorage.removeItem(key)
    return initialValue
  }
}
