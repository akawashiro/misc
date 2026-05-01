import { describe, expect, it } from 'vitest'
import { compile, compileGenerator, compileNormalTerm } from './compiler'
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
    const result = execute('let cogen result = code (40 + 2) in result end')
    expect(formatValue(result.value)).toBe('42')
    expect(result.transitions.length).toBeGreaterThan(1)
  })

  it('evaluates the minimal generated code sample', () => {
    const result = execute('let cogen result = code 1 in result end')
    expect(formatValue(result.value)).toBe('1')
  })

  it('specializes a code variable with let cogen', () => {
    const result = execute('let cogen result = (let cogen a = lift (6 * 7) in code (a + 8) end) in result end')
    expect(formatValue(result.value)).toBe('50')
    expect(formatProgram(result.compiled.program)).toContain('app')
    expect(formatProgram(result.compiled.program)).not.toContain('splice')
    expect(result.compiled.log.join('\n')).not.toContain('splice')
  })

  it('specializes a cogen binding produced from code', () => {
    const result = execute('let cogen result = (let cogen u = code 1 in code u end) in result end')
    expect(formatValue(result.value)).toBe('1')
  })

  it('rejects eval because it is not an ML^box primitive', () => {
    expect(() => parse('eval (code 1)')).toThrow('Expected expression')
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
    const ast = parse('let cogen result = (let cogen a = lift (6 * 7) in code (a + 8) end) in result end')
    const compiled = compile(ast)

    expectLinesInOrder(compiled.log, [
      'push; [[ let cogen a = lift (6 * 7) in code (a + 8) end ]] Ω=∅; cons; [[ result ]] Ω=∅ Λ=result',
      'push; push; [[ lift (6 * 7) ]] Ω=∅; cons; [[ code (a + 8) ]] Ω=∅ Λ=a; cons; [[ result ]] Ω=∅ Λ=result',
      'push; push; [[ 6 * 7 ]] Ω=∅; Cur(lift; snd); cons; [[ code (a + 8) ]] Ω=∅ Λ=a; cons; [[ result ]] Ω=∅ Λ=result',
      'push; push; push; [[ 6 ]] Ω=∅; swap; [[ 7 ]] Ω=∅; cons; mul; Cur(lift; snd); cons; [[ code (a + 8) ]] Ω=∅ Λ=a; cons; [[ result ]] Ω=∅ Λ=result',
      'push; push; push; \'6; swap; [[ 7 ]] Ω=∅; cons; mul; Cur(lift; snd); cons; [[ code (a + 8) ]] Ω=∅ Λ=a; cons; [[ result ]] Ω=∅ Λ=result',
      'push; push; push; \'6; swap; \'7; cons; mul; Cur(lift; snd); cons; [[ code (a + 8) ]] Ω=∅ Λ=a; cons; [[ result ]] Ω=∅ Λ=result',
      'push; push; push; \'6; swap; \'7; cons; mul; Cur(lift; snd); cons; Cur([[ a + 8 ]] Ω=∅ Λ=a; snd); cons; [[ result ]] Ω=∅ Λ=result',
    ])
    expectLinesInOrder(compiled.log, [
      'push; push; push; \'6; swap; \'7; cons; mul; Cur(lift; snd); cons; Cur(emit(push); [[ a ]] Ω=∅ Λ=a; emit(swap); [[ 8 ]] Ω=∅ Λ=a; emit(cons); emit(add); snd); cons; [[ result ]] Ω=∅ Λ=result',
      'push; push; push; \'6; swap; \'7; cons; mul; Cur(lift; snd); cons; Cur(emit(push); push; push; fst; snd; swap; snd; cons; app; swap; emit(swap); [[ 8 ]] Ω=∅ Λ=a; emit(cons); emit(add); snd); cons; [[ result ]] Ω=∅ Λ=result',
      'push; push; push; \'6; swap; \'7; cons; mul; Cur(lift; snd); cons; Cur(emit(push); push; push; fst; snd; swap; snd; cons; app; swap; emit(swap); emit(\'8); emit(cons); emit(add); snd); cons; snd; arena; cons; app; call',
    ])
    expect(compiled.log).not.toContain(
      'push; push; push; \'6; swap; \'7; cons; mul; Cur(lift; snd); cons; Cur(emit(push); push; emit(swap); [[ 8 ]] Ω=∅ Λ=a; emit(cons); emit(add); snd); cons; snd; arena; cons; app; call',
    )
    expect(compiled.log.join('\n')).not.toContain('eval')
    expect(compiled.log).not.toContain('push')
    expect(compiled.log.at(-1)).toBe(formatProgram(compiled.program))
  })

  describe('Figure 4 compilation rules', () => {
    it('compiles a value variable by selecting it from the environment', () => {
      const compiled = compileNormalTerm(parse('x'), [{ name: 'x', isCode: false }])
      expect(formatProgram(compiled.program)).toBe('snd')
    })

    it('compiles a lambda as a Cur instruction over the compiled body', () => {
      const compiled = compileNormalTerm(parse('fn x => x + 1'), [])
      expect(formatProgram(compiled.program)).toBe("Cur(push; snd; swap; '1; cons; add)")
    })

    it('compiles an application by evaluating function and argument before app', () => {
      const compiled = compileNormalTerm(parse('(fn x => x) 1'), [])
      expect(formatProgram(compiled.program)).toBe("push; Cur(snd); swap; '1; cons; app")
    })

    it('compiles a code variable by activating its generator in a fresh arena', () => {
      const compiled = compileNormalTerm(parse('u'), [{ name: 'u', isCode: true }])
      expect(formatProgram(compiled.program)).toBe('snd; arena; cons; app; call')
    })

    it('compiles code as a Cur instruction around generator compilation', () => {
      const compiled = compileNormalTerm(parse('code 1'), [])
      expect(formatProgram(compiled.program)).toBe("Cur(emit('1); snd)")
    })

    it('compiles lift by compiling the source term and wrapping lift in Cur', () => {
      const compiled = compileNormalTerm(parse('lift (1 + 2)'), [])
      expect(formatProgram(compiled.program)).toBe("push; '1; swap; '2; cons; add; Cur(lift; snd)")
    })

    it('compiles let cogen by pairing the generated binding with the body environment', () => {
      const compiled = compileNormalTerm(parse('let cogen u = code 1 in 2 end'), [])
      expect(formatProgram(compiled.program)).toBe("push; Cur(emit('1); snd); cons; '2")
    })

    it('compiles generator value variables by emitting environment selections', () => {
      const compiled = compileGenerator(parse('x'), [{ name: 'x', isCode: false }])
      expect(formatProgram(compiled.program)).toBe('emit(snd)')
    })

    it('compiles generator lambdas by generating the body in a fresh arena before merge', () => {
      const compiled = compileGenerator(parse('fn x => x'))
      expect(formatProgram(compiled.program)).toBe('push; fst; arena; emit(snd); snd; swap; id; cons; merge(Cur(snd))')
    })

    it('compiles generator applications by emitting push, swap, cons, and app', () => {
      const compiled = compileGenerator(parse('(fn x => x) 1'))
      expect(formatProgram(compiled.program)).toBe(
        "emit(push); emit(Cur(snd)); emit(swap); emit('1); emit(cons); emit(app)",
      )
    })

    it('compiles generator code variables in Omega by emitting future generator activation', () => {
      const compiled = compileGenerator(parse('let cogen u = code 1 in u end'))
      expect(formatProgram(compiled.program)).toBe(
        "emit(push); emit(Cur(emit('1); snd)); emit(cons); emit(snd); emit(arena); emit(cons); emit(app); emit(call)",
      )
    })

    it('compiles generator code variables in Lambda by substituting the captured generator', () => {
      const compiled = compileGenerator(parse('u'), [{ name: 'u', isCode: true }], ['u'])
      expect(formatProgram(compiled.program)).toBe('push; push; fst; snd; swap; snd; cons; app; swap')
    })

    it('compiles nested generator code by lifting and applying a generator closure', () => {
      const compiled = compileGenerator(parse('code 1'))
      expect(formatProgram(compiled.program)).toBe(
        "push; fst; push; '(); Cur(fst; Cur(emit('1); snd)); swap; snd; cons; lift; snd; swap; id; cons; app",
      )
    })

    it('compiles generator lift by evaluating the source term into the current block', () => {
      const compiled = compileGenerator(parse('lift 1'))
      expect(formatProgram(compiled.program)).toBe("'1; merge(Cur(fst; arena; lift; snd; id))")
    })

    it('compiles generator let cogen by emitting the normal let cogen compilation', () => {
      const compiled = compileGenerator(parse('let cogen u = code 1 in 2 end'))
      expect(formatProgram(compiled.program)).toBe(
        "emit(push); emit(Cur(emit('1); snd)); emit(cons); emit('2)",
      )
    })
  })
})
