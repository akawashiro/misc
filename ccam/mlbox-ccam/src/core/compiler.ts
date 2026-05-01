import type { ContextEntry, Expr } from './ast'
import type { Instruction, Value } from './ccam'
import { formatProgram } from './ccam'

export type CompileResult = {
  program: Instruction[]
  log: string[]
}

export function compile(expr: Expr): CompileResult {
  const log: string[] = []
  const program = compileCore(expr, [], log, 'term')
  return { program, log }
}

function compileCore(expr: Expr, ctx: ContextEntry[], log: string[], label: string): Instruction[] {
  const emitLog = (program: Instruction[], rule: string) => {
    log.push(`${label}: ${rule} => ${formatProgram(program)}`)
    return program
  }

  switch (expr.type) {
    case 'int':
      return emitLog([{ op: 'quote', value: intValue(expr.value) }], `[${expr.value}]`)
    case 'var': {
      const entry = lookup(ctx, expr.name)
      if (entry.isCode) {
        const program: Instruction[] = [...envPath(ctx, expr.name), { op: 'arena' }, { op: 'cons' }, { op: 'app' }, { op: 'call' }]
        return emitLog(program, `[${expr.name}] code variable`)
      }
      return emitLog(envPath(ctx, expr.name), `[${expr.name}] value variable`)
    }
    case 'lambda': {
      const body = compileCore(expr.body, [...ctx, { name: expr.param, isCode: false }], log, `${label}.${expr.param}`)
      return emitLog([{ op: 'cur', program: body }], `[fn ${expr.param} => M]`)
    }
    case 'app': {
      const program: Instruction[] = [
        { op: 'push' },
        ...compileCore(expr.fn, ctx, log, `${label}.fn`),
        { op: 'swap' },
        ...compileCore(expr.arg, ctx, log, `${label}.arg`),
        { op: 'cons' },
        { op: 'app' },
      ]
      return emitLog(program, `[M N]`)
    }
    case 'binary': {
      const op = expr.op === '+' ? 'add' : expr.op === '-' ? 'sub' : 'mul'
      const program: Instruction[] = [
        { op: 'push' },
        ...compileCore(expr.left, ctx, log, `${label}.left`),
        { op: 'swap' },
        ...compileCore(expr.right, ctx, log, `${label}.right`),
        { op: 'cons' },
        { op },
      ]
      return emitLog(program, `[M ${expr.op} N]`)
    }
    case 'code': {
      const body = compileGenerator(expr.body, ctx, ctx.filter((entry) => entry.isCode).map((entry) => entry.name), log, `${label}.code`)
      return emitLog([{ op: 'cur', program: [...body, { op: 'snd' }] }], `[code M]`)
    }
    case 'lift': {
      const body = compileCore(expr.body, ctx, log, `${label}.lift`)
      return emitLog([...body, { op: 'cur', program: [{ op: 'lift' }, { op: 'snd' }] }], `[lift M]`)
    }
    case 'letCogen': {
      const program: Instruction[] = [
        { op: 'push' },
        ...compileCore(expr.generator, ctx, log, `${label}.cogen`),
        { op: 'cons' },
        ...compileCore(expr.body, [...ctx, { name: expr.name, isCode: true }], log, `${label}.body`),
      ]
      return emitLog(program, `[let cogen ${expr.name} = M in N end]`)
    }
    case 'eval': {
      const program: Instruction[] = [
        ...compileCore(expr.body, ctx, log, `${label}.eval`),
        { op: 'arena' },
        { op: 'cons' },
        { op: 'app' },
        { op: 'call' },
      ]
      return emitLog(program, `[eval M]`)
    }
  }
}

function compileGenerator(
  expr: Expr,
  capturedCtx: ContextEntry[],
  codeVars: string[],
  log: string[],
  label: string,
): Instruction[] {
  const emitLog = (program: Instruction[], rule: string) => {
    log.push(`${label}: ${rule} => ${formatProgram(program)}`)
    return program
  }

  switch (expr.type) {
    case 'int':
      return emitLog([{ op: 'emit', instruction: { op: 'quote', value: intValue(expr.value) } }], `emit ${expr.value}`)
    case 'var':
      if (codeVars.includes(expr.name)) {
        return emitLog([{ op: 'splice', path: envPath(capturedCtx, expr.name) }], `specialize code variable ${expr.name}`)
      }
      return emitLog(
        envPath(capturedCtx, expr.name).map((instruction) => ({ op: 'emit', instruction }) as Instruction),
        `emit value variable ${expr.name}`,
      )
    case 'lambda': {
      const body = compileGenerator(
        expr.body,
        [...capturedCtx, { name: expr.param, isCode: false }],
        codeVars,
        log,
        `${label}.${expr.param}`,
      )
      const bodyProgram = emittedProgram(body)
      return emitLog([{ op: 'merge', program: bodyProgram }], `merge lambda ${expr.param}`)
    }
    case 'app': {
      const program = emitSequence([
        { op: 'push' },
        ...emittedProgram(compileGenerator(expr.fn, capturedCtx, codeVars, log, `${label}.fn`)),
        { op: 'swap' },
        ...emittedProgram(compileGenerator(expr.arg, capturedCtx, codeVars, log, `${label}.arg`)),
        { op: 'cons' },
        { op: 'app' },
      ])
      return emitLog(program, 'emit application')
    }
    case 'binary': {
      const op = expr.op === '+' ? 'add' : expr.op === '-' ? 'sub' : 'mul'
      const program = [
        ...emitSequence([{ op: 'push' }]),
        ...compileGenerator(expr.left, capturedCtx, codeVars, log, `${label}.left`),
        ...emitSequence([{ op: 'swap' }]),
        ...compileGenerator(expr.right, capturedCtx, codeVars, log, `${label}.right`),
        ...emitSequence([{ op: 'cons' }, { op }]),
      ] as Instruction[]
      return emitLog(program, `emit ${expr.op}`)
    }
    case 'lift':
      return emitLog([{ op: 'evalLift', program: compileCore(expr.body, capturedCtx, log, `${label}.lift`) }], 'lift into current block')
    case 'code': {
      const nested = compileGenerator(expr.body, capturedCtx, codeVars, log, `${label}.nested`)
      return emitLog([{ op: 'merge', program: [...nested, { op: 'snd' }] }], 'lift nested code generator')
    }
    case 'letCogen': {
      const generated = compileCore(expr, capturedCtx, log, `${label}.let`)
      return emitLog(emitSequence(generated), 'emit let cogen for later stage')
    }
    case 'eval': {
      const generated = compileCore(expr, capturedCtx, log, `${label}.eval`)
      return emitLog(emitSequence(generated), 'emit eval for later stage')
    }
  }
}

function emittedProgram(program: Instruction[]): Instruction[] {
  return program.flatMap((instruction) => {
    if (instruction.op === 'emit') return [instruction.instruction]
    if (instruction.op === 'merge') return [{ op: 'cur', program: instruction.program } as Instruction]
    return [instruction]
  })
}

function emitSequence(program: Instruction[]): Instruction[] {
  return program.map((instruction) => ({ op: 'emit', instruction }) as Instruction)
}

function envPath(ctx: ContextEntry[], name: string): Instruction[] {
  const index = ctx.findIndex((entry) => entry.name === name)
  if (index < 0) throw new Error(`Unbound variable: ${name}`)
  const depthFromRight = ctx.length - index - 1
  const path: Instruction[] = Array.from({ length: depthFromRight }, () => ({ op: 'fst' }) as Instruction)
  path.push({ op: 'snd' })
  return path
}

function lookup(ctx: ContextEntry[], name: string): ContextEntry {
  const entry = [...ctx].reverse().find((item) => item.name === name)
  if (!entry) throw new Error(`Unbound variable: ${name}`)
  return entry
}

function intValue(value: number): Value {
  return { type: 'int', value }
}
