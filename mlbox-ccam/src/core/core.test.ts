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

  it('specializes a cogen binding produced from code', () => {
    const result = execute('eval (let cogen u = code 1 in code u end)')
    expect(formatValue(result.value)).toBe('1')
  })

  it('formats compile traces from source term to final CCAM program', () => {
    const ast = parse('(fn x => x + 1) 41')
    const compiled = compile(ast)

    expect(compiled.log[0]).toBe('[[ (fn x => x + 1) 41 ]] Ω=∅')
    expectLinesInOrder(compiled.log, [
      'push; [[ fn x => x + 1 ]] Ω=∅; swap; [[ 41 ]] Ω=∅; cons; app',
      'push; Cur([[ x + 1 ]] Ω=x); swap; [[ 41 ]] Ω=∅; cons; app',
      'push; Cur(push; snd; swap; \'1; cons; add); swap; \'41; cons; app',
    ])
    expect(compiled.log.some((line) => line.includes('Cur(') && line.includes('[[ 41 ]] Ω=∅'))).toBe(true)
    expect(compiled.log).not.toContain('push')
    expect(compiled.log.at(-1)).toBe(formatProgram(compiled.program))
  })

  it('expands let cogen and lift compile traces one instruction at a time', () => {
    const ast = parse('eval (let cogen a = lift (6 * 7) in code (a + 8) end)')
    const compiled = compile(ast)

    expectLinesInOrder(compiled.log, [
      'push; [[ lift (6 * 7) ]] Ω=∅; cons; [[ code (a + 8) ]] Ω=∅ Λ=a; arena; cons; app; call',
      'push; [[ 6 * 7 ]] Ω=∅; Cur(lift; snd); cons; [[ code (a + 8) ]] Ω=∅ Λ=a; arena; cons; app; call',
      'push; push; [[ 6 ]] Ω=∅; swap; [[ 7 ]] Ω=∅; cons; mul; Cur(lift; snd); cons; [[ code (a + 8) ]] Ω=∅ Λ=a; arena; cons; app; call',
      'push; push; \'6; swap; [[ 7 ]] Ω=∅; cons; mul; Cur(lift; snd); cons; [[ code (a + 8) ]] Ω=∅ Λ=a; arena; cons; app; call',
      'push; push; \'6; swap; \'7; cons; mul; Cur(lift; snd); cons; [[ code (a + 8) ]] Ω=∅ Λ=a; arena; cons; app; call',
      'push; push; \'6; swap; \'7; cons; mul; Cur(lift; snd); cons; Cur([[ a + 8 ]] Ω=∅ Λ=a; snd); arena; cons; app; call',
    ])
    expectLinesInOrder(compiled.log, [
      'push; push; \'6; swap; \'7; cons; mul; Cur(lift; snd); cons; Cur(emit(push); [[ a ]] Ω=∅ Λ=a; emit(swap); [[ 8 ]] Ω=∅ Λ=a; emit(cons); emit(add); snd); arena; cons; app; call',
      'push; push; \'6; swap; \'7; cons; mul; Cur(lift; snd); cons; Cur(emit(push); push; push; fst; snd; swap; snd; cons; app; swap; emit(swap); [[ 8 ]] Ω=∅ Λ=a; emit(cons); emit(add); snd); arena; cons; app; call',
    ])
    expect(compiled.log).not.toContain(
      'push; push; \'6; swap; \'7; cons; mul; Cur(lift; snd); cons; Cur(emit(push); push; emit(swap); [[ 8 ]] Ω=∅ Λ=a; emit(cons); emit(add); snd); arena; cons; app; call',
    )
    expect(compiled.log).not.toContain('push')
    expect(compiled.log.at(-1)).toBe(formatProgram(compiled.program))
  })

  describe('Figure 4 compilation rules', () => {
    it('compiles a value variable by selecting it from the environment', () => {
      const compiled = compile(parse('fn x => x'))

      expectLinesInOrder(compiled.log, ['Cur([[ x ]] Ω=x)', 'Cur(snd)'])
      expect(formatProgram(compiled.program)).toBe('Cur(snd)')
    })

    it('compiles a lambda as a Cur instruction over the compiled body', () => {
      const compiled = compile(parse('fn x => x + 1'))

      expectLinesInOrder(compiled.log, [
        'Cur([[ x + 1 ]] Ω=x)',
        "Cur(push; snd; swap; '1; cons; add)",
      ])
      expect(formatProgram(compiled.program)).toBe("Cur(push; snd; swap; '1; cons; add)")
    })

    it('compiles an application by evaluating function and argument before app', () => {
      const compiled = compile(parse('(fn x => x) 1'))

      expectLinesInOrder(compiled.log, [
        'push; [[ fn x => x ]] Ω=∅; swap; [[ 1 ]] Ω=∅; cons; app',
        "push; Cur(snd); swap; '1; cons; app",
      ])
      expect(formatProgram(compiled.program)).toBe("push; Cur(snd); swap; '1; cons; app")
    })

    it('compiles a code variable by activating its generator in a fresh arena', () => {
      const compiled = compile(parse('let cogen u = code 1 in u end'))

      expectLinesInOrder(compiled.log, [
        'push; [[ code 1 ]] Ω=∅; cons; [[ u ]] Ω=∅ Λ=u',
        "push; Cur(emit('1); snd); cons; snd; arena; cons; app; call",
      ])
      expect(formatProgram(compiled.program)).toBe("push; Cur(emit('1); snd); cons; snd; arena; cons; app; call")
    })

    it('compiles code as a Cur instruction around generator compilation', () => {
      const compiled = compile(parse('code 1'))

      expectLinesInOrder(compiled.log, ['Cur([[ 1 ]] Ω=∅ Λ=∅; snd)', "Cur(emit('1); snd)"])
      expect(formatProgram(compiled.program)).toBe("Cur(emit('1); snd)")
    })

    it('compiles lift by compiling the source term and wrapping lift in Cur', () => {
      const compiled = compile(parse('lift (1 + 2)'))

      expectLinesInOrder(compiled.log, [
        '[[ 1 + 2 ]] Ω=∅; Cur(lift; snd)',
        "push; '1; swap; '2; cons; add; Cur(lift; snd)",
      ])
      expect(formatProgram(compiled.program)).toBe("push; '1; swap; '2; cons; add; Cur(lift; snd)")
    })

    it('compiles let cogen by pairing the generated binding with the body environment', () => {
      const compiled = compile(parse('let cogen u = code 1 in 2 end'))

      expectLinesInOrder(compiled.log, [
        'push; [[ code 1 ]] Ω=∅; cons; [[ 2 ]] Ω=∅ Λ=u',
        "push; Cur(emit('1); snd); cons; '2",
      ])
      expect(formatProgram(compiled.program)).toBe("push; Cur(emit('1); snd); cons; '2")
    })

    it('compiles generator value variables by emitting environment selections', () => {
      const compiled = compile(parse('fn x => code x'))

      expectLinesInOrder(compiled.log, ['Cur([[ code x ]] Ω=x)', 'Cur(Cur(emit(snd); snd))'])
      expect(formatProgram(compiled.program)).toBe('Cur(Cur(emit(snd); snd))')
    })

    it('compiles generator lambdas by merging a generated Cur body', () => {
      const compiled = compile(parse('code (fn x => x)'))

      expectLinesInOrder(compiled.log, [
        'Cur([[ fn x => x ]] Ω=∅ Λ=∅; snd)',
        'Cur(merge(Cur(snd)); snd)',
      ])
      expect(formatProgram(compiled.program)).toBe('Cur(merge(Cur(snd)); snd)')
    })

    it('compiles generator applications by emitting push, swap, cons, and app', () => {
      const compiled = compile(parse('code ((fn x => x) 1)'))

      expectLinesInOrder(compiled.log, [
        'Cur(emit(push); [[ fn x => x ]] Ω=∅ Λ=∅; emit(swap); [[ 1 ]] Ω=∅ Λ=∅; emit(cons); emit(app); snd)',
        "Cur(emit(push); emit(Cur(snd)); emit(swap); emit('1); emit(cons); emit(app); snd)",
      ])
      expect(formatProgram(compiled.program)).toBe(
        "Cur(emit(push); emit(Cur(snd)); emit(swap); emit('1); emit(cons); emit(app); snd)",
      )
    })

    it('compiles generator code variables in Omega by emitting future generator activation', () => {
      const compiled = compile(parse('code (let cogen u = code 1 in u end)'))

      expectLinesInOrder(compiled.log, [
        'Cur([[ let cogen u = code 1 in u end ]] Ω=∅ Λ=∅; snd)',
        "Cur(emit(push); emit(Cur(emit('1); snd)); emit(cons); emit(snd); emit(arena); emit(cons); emit(app); emit(call); snd)",
      ])
      expect(formatProgram(compiled.program)).toBe(
        "Cur(emit(push); emit(Cur(emit('1); snd)); emit(cons); emit(snd); emit(arena); emit(cons); emit(app); emit(call); snd)",
      )
    })

    it('compiles generator code variables in Lambda by substituting the captured generator', () => {
      const compiled = compile(parse('let cogen u = code 1 in code u end'))

      expect(compiled.log.some((line) => line.includes('Cur([[ u ]] Ω=∅ Λ=u; snd)'))).toBe(true)
      expect(compiled.log.some((line) => line.includes('Cur(push; push; fst; snd; swap; snd; cons; app; swap; snd)'))).toBe(
        true,
      )
      expect(formatProgram(compiled.program)).toBe(
        "push; Cur(emit('1); snd); cons; Cur(push; push; fst; snd; swap; snd; cons; app; swap; snd)",
      )
    })

    it('compiles nested generator code by merging a lifted generator closure', () => {
      const compiled = compile(parse('code (code 1)'))

      expectLinesInOrder(compiled.log, [
        'Cur([[ code 1 ]] Ω=∅ Λ=∅; snd)',
        "Cur(merge(Cur(emit('1); snd)); snd)",
      ])
      expect(formatProgram(compiled.program)).toBe("Cur(merge(Cur(emit('1); snd)); snd)")
    })

    it('compiles generator lift by evaluating the source term into the current block', () => {
      const compiled = compile(parse('code (lift (1 + 2))'))

      expectLinesInOrder(compiled.log, [
        'Cur([[ lift (1 + 2) ]] Ω=∅ Λ=∅; snd)',
        "Cur(lift[push; '1; swap; '2; cons; add]; snd)",
      ])
      expect(formatProgram(compiled.program)).toBe("Cur(lift[push; '1; swap; '2; cons; add]; snd)")
    })

    it('compiles generator let cogen by emitting the normal let cogen compilation', () => {
      const compiled = compile(parse('code (let cogen u = code 1 in 2 end)'))

      expectLinesInOrder(compiled.log, [
        'Cur([[ let cogen u = code 1 in 2 end ]] Ω=∅ Λ=∅; snd)',
        "Cur(emit(push); emit(Cur(emit('1); snd)); emit(cons); emit('2); snd)",
      ])
      expect(formatProgram(compiled.program)).toBe(
        "Cur(emit(push); emit(Cur(emit('1); snd)); emit(cons); emit('2); snd)",
      )
    })
  })
})
