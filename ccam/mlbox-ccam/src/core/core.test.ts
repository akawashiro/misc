import { describe, expect, it } from 'vitest'
import { compile } from './compiler'
import { formatProgram, formatValue, run } from './ccam'
import { parse } from './parser'

function execute(source: string) {
  const ast = parse(source)
  const compiled = compile(ast)
  return { ...run(compiled.program), compiled }
}

function expectLinesInOrder(log: string[], expected: string[]) {
  let previousIndex = -1
  for (const line of expected) {
    const index = log.findIndex((entry, entryIndex) => entryIndex > previousIndex && entry === line)
    expect(index).toBeGreaterThan(previousIndex)
    previousIndex = index
  }
}

describe('ML^box parser/compiler/CCAM', () => {
  it('runs integer arithmetic on the CCAM', () => {
    const result = execute('6 * 7 + 8')
    expect(formatValue(result.value)).toBe('50')
  })

  it('runs lambda application on the CCAM', () => {
    const result = execute('(fn x => x + 1) 41')
    expect(formatValue(result.value)).toBe('42')
  })

  it('evaluates a generated code block', () => {
    const result = execute('eval (code (40 + 2))')
    expect(formatValue(result.value)).toBe('42')
    expect(result.transitions.length).toBeGreaterThan(1)
  })

  it('evaluates the minimal generated code sample', () => {
    const result = execute('eval (code 1)')
    expect(formatValue(result.value)).toBe('1')
  })

  it('specializes a code variable with let cogen', () => {
    const result = execute('eval (let cogen a = lift (6 * 7) in code (a + 8) end)')
    expect(formatValue(result.value)).toBe('50')
    expect(formatProgram(result.compiled.program)).toContain('app')
    expect(formatProgram(result.compiled.program)).not.toContain('splice')
    expect(result.compiled.log.join('\n')).not.toContain('splice')
  })

  it('formats compile traces from source term to final CCAM program', () => {
    const ast = parse('(fn x => x + 1) 41')
    const compiled = compile(ast)

    expect(compiled.log[0]).toBe('[[ (fn x => x + 1) 41 ]] Ω=∅ Λ=∅')
    expectLinesInOrder(compiled.log, [
      'push',
      'push; [[ fn x => x + 1 ]] Ω=∅ Λ=∅',
      'push; Cur([[ x + 1 ]] Ω=x Λ=∅)',
      'push; Cur(push; snd; swap; \'1; cons; add); swap; \'41; cons; app',
    ])
    expect(compiled.log.some((line) => line.includes('Cur(') && line.includes('[[ 41 ]] Ω=∅ Λ=∅'))).toBe(true)
    expect(compiled.log.at(-1)).toBe(formatProgram(compiled.program))
  })

  it('expands let cogen and lift compile traces one instruction at a time', () => {
    const ast = parse('eval (let cogen a = lift (6 * 7) in code (a + 8) end)')
    const compiled = compile(ast)

    expectLinesInOrder(compiled.log, [
      'push; [[ lift (6 * 7) ]] Ω=∅ Λ=∅',
      'push; [[ 6 * 7 ]] Ω=∅ Λ=∅',
      'push; push',
      'push; push; [[ 6 ]] Ω=∅ Λ=∅',
      'push; push; \'6',
      'push; push; \'6; swap',
      'push; push; \'6; swap; [[ 7 ]] Ω=∅ Λ=∅',
      'push; push; \'6; swap; \'7',
      'push; push; \'6; swap; \'7; cons',
      'push; push; \'6; swap; \'7; cons; mul',
      'push; push; \'6; swap; \'7; cons; mul; Cur(lift; snd)',
    ])
    expect(compiled.log.at(-1)).toBe(formatProgram(compiled.program))
  })
})
