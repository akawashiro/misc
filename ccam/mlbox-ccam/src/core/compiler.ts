import type { ContextEntry, Expr } from './ast'
import type { Instruction, Value } from './ccam'
import { formatProgram } from './ccam'

export type CompileResult = {
  program: Instruction[]
  log: string[]
}

type Compiled = {
  program: Instruction[]
  trace: string[]
}

type ChildTrace = {
  placeholder: string
  trace: string[]
}

export function compile(expr: Expr): CompileResult {
  const compiled = compileCore(expr, [])
  return { program: compiled.program, log: compiled.trace }
}

function compileCore(expr: Expr, ctx: ContextEntry[]): Compiled {
  const judgement = formatJudgement(expr, ctx, [])

  switch (expr.type) {
    case 'int':
      return leaf(judgement, [{ op: 'quote', value: intValue(expr.value) }])
    case 'var': {
      const entry = lookup(ctx, expr.name)
      const path = envPath(ctx, expr.name)
      if (entry.isCode) {
        return leaf(judgement, [...path, { op: 'arena' }, { op: 'cons' }, { op: 'app' }, { op: 'call' }])
      }
      return leaf(judgement, path)
    }
    case 'lambda': {
      const bodyCtx = [...ctx, { name: expr.param, isCode: false }]
      const body = compileCore(expr.body, bodyCtx)
      const program: Instruction[] = [{ op: 'cur', program: body.program }]
      return composite(judgement, `Cur(${formatJudgement(expr.body, bodyCtx, [])})`, program, [
        { placeholder: formatJudgement(expr.body, bodyCtx, []), trace: body.trace },
      ])
    }
    case 'app': {
      const fn = compileCore(expr.fn, ctx)
      const arg = compileCore(expr.arg, ctx)
      const program: Instruction[] = [
        { op: 'push' },
        ...fn.program,
        { op: 'swap' },
        ...arg.program,
        { op: 'cons' },
        { op: 'app' },
      ]
      return composite(
        judgement,
        `push; ${formatJudgement(expr.fn, ctx, [])}; swap; ${formatJudgement(expr.arg, ctx, [])}; cons; app`,
        program,
        [
          { placeholder: formatJudgement(expr.fn, ctx, []), trace: fn.trace },
          { placeholder: formatJudgement(expr.arg, ctx, []), trace: arg.trace },
        ],
      )
    }
    case 'binary': {
      const op = expr.op === '+' ? 'add' : expr.op === '-' ? 'sub' : 'mul'
      const left = compileCore(expr.left, ctx)
      const right = compileCore(expr.right, ctx)
      const program: Instruction[] = [
        { op: 'push' },
        ...left.program,
        { op: 'swap' },
        ...right.program,
        { op: 'cons' },
        { op },
      ]
      return composite(
        judgement,
        `push; ${formatJudgement(expr.left, ctx, [])}; swap; ${formatJudgement(expr.right, ctx, [])}; cons; ${op}`,
        program,
        [
          { placeholder: formatJudgement(expr.left, ctx, []), trace: left.trace },
          { placeholder: formatJudgement(expr.right, ctx, []), trace: right.trace },
        ],
      )
    }
    case 'code': {
      const codeVars = ctx.filter((entry) => entry.isCode).map((entry) => entry.name)
      const body = compileGenerator(expr.body, ctx, codeVars)
      const program: Instruction[] = [{ op: 'cur', program: [...body.program, { op: 'snd' }] }]
      return composite(judgement, `Cur(${formatJudgement(expr.body, ctx, codeVars)}; snd)`, program, [
        { placeholder: formatJudgement(expr.body, ctx, codeVars), trace: body.trace },
      ])
    }
    case 'lift': {
      const body = compileCore(expr.body, ctx)
      const program: Instruction[] = [...body.program, { op: 'cur', program: [{ op: 'lift' }, { op: 'snd' }] }]
      return composite(judgement, `${formatJudgement(expr.body, ctx, [])}; Cur(lift; snd)`, program, [
        { placeholder: formatJudgement(expr.body, ctx, []), trace: body.trace },
      ])
    }
    case 'letCogen': {
      const generator = compileCore(expr.generator, ctx)
      const bodyCtx = [...ctx, { name: expr.name, isCode: true }]
      const body = compileCore(expr.body, bodyCtx)
      const program: Instruction[] = [{ op: 'push' }, ...generator.program, { op: 'cons' }, ...body.program]
      return composite(
        judgement,
        `push; ${formatJudgement(expr.generator, ctx, [])}; cons; ${formatJudgement(expr.body, bodyCtx, [])}`,
        program,
        [
          { placeholder: formatJudgement(expr.generator, ctx, []), trace: generator.trace },
          { placeholder: formatJudgement(expr.body, bodyCtx, []), trace: body.trace },
        ],
      )
    }
    case 'eval': {
      const body = compileCore(expr.body, ctx)
      const program: Instruction[] = [...body.program, { op: 'arena' }, { op: 'cons' }, { op: 'app' }, { op: 'call' }]
      return composite(judgement, `${formatJudgement(expr.body, ctx, [])}; arena; cons; app; call`, program, [
        { placeholder: formatJudgement(expr.body, ctx, []), trace: body.trace },
      ])
    }
  }
}

function compileGenerator(expr: Expr, capturedCtx: ContextEntry[], codeVars: string[]): Compiled {
  const judgement = formatJudgement(expr, capturedCtx, codeVars)

  switch (expr.type) {
    case 'int':
      return leaf(judgement, [{ op: 'emit', instruction: { op: 'quote', value: intValue(expr.value) } }])
    case 'var':
      if (codeVars.includes(expr.name)) {
        return leaf(judgement, [{ op: 'splice', path: envPath(capturedCtx, expr.name) }])
      }
      return leaf(
        judgement,
        envPath(capturedCtx, expr.name).map((instruction) => ({ op: 'emit', instruction }) as Instruction),
      )
    case 'lambda': {
      const bodyCtx = [...capturedCtx, { name: expr.param, isCode: false }]
      const body = compileGenerator(expr.body, bodyCtx, codeVars)
      const bodyProgram = emittedProgram(body.program)
      const program: Instruction[] = [{ op: 'merge', program: bodyProgram }]
      return composite(judgement, `merge(Cur(${formatJudgement(expr.body, bodyCtx, codeVars)}))`, program, [
        { placeholder: formatJudgement(expr.body, bodyCtx, codeVars), trace: body.trace },
      ])
    }
    case 'app': {
      const fn = compileGenerator(expr.fn, capturedCtx, codeVars)
      const arg = compileGenerator(expr.arg, capturedCtx, codeVars)
      const program = emitSequence([
        { op: 'push' },
        ...emittedProgram(fn.program),
        { op: 'swap' },
        ...emittedProgram(arg.program),
        { op: 'cons' },
        { op: 'app' },
      ])
      return composite(
        judgement,
        `emit(push); ${formatJudgement(expr.fn, capturedCtx, codeVars)}; emit(swap); ${formatJudgement(expr.arg, capturedCtx, codeVars)}; emit(cons); emit(app)`,
        program,
        [
          { placeholder: formatJudgement(expr.fn, capturedCtx, codeVars), trace: fn.trace },
          { placeholder: formatJudgement(expr.arg, capturedCtx, codeVars), trace: arg.trace },
        ],
      )
    }
    case 'binary': {
      const op = expr.op === '+' ? 'add' : expr.op === '-' ? 'sub' : 'mul'
      const left = compileGenerator(expr.left, capturedCtx, codeVars)
      const right = compileGenerator(expr.right, capturedCtx, codeVars)
      const program = [
        ...emitSequence([{ op: 'push' }]),
        ...left.program,
        ...emitSequence([{ op: 'swap' }]),
        ...right.program,
        ...emitSequence([{ op: 'cons' }, { op }]),
      ] as Instruction[]
      return composite(
        judgement,
        `emit(push); ${formatJudgement(expr.left, capturedCtx, codeVars)}; emit(swap); ${formatJudgement(expr.right, capturedCtx, codeVars)}; emit(cons); emit(${op})`,
        program,
        [
          { placeholder: formatJudgement(expr.left, capturedCtx, codeVars), trace: left.trace },
          { placeholder: formatJudgement(expr.right, capturedCtx, codeVars), trace: right.trace },
        ],
      )
    }
    case 'lift': {
      const body = compileCore(expr.body, capturedCtx)
      const program: Instruction[] = [{ op: 'evalLift', program: body.program }]
      return composite(judgement, `lift[${formatJudgement(expr.body, capturedCtx, [])}]`, program, [
        { placeholder: formatJudgement(expr.body, capturedCtx, []), trace: body.trace },
      ])
    }
    case 'code': {
      const nested = compileGenerator(expr.body, capturedCtx, codeVars)
      const program: Instruction[] = [{ op: 'merge', program: [...nested.program, { op: 'snd' }] }]
      return composite(judgement, `merge(Cur(${formatJudgement(expr.body, capturedCtx, codeVars)}; snd))`, program, [
        { placeholder: formatJudgement(expr.body, capturedCtx, codeVars), trace: nested.trace },
      ])
    }
    case 'letCogen': {
      const generated = compileCore(expr, capturedCtx)
      return composite(judgement, `emit(${formatJudgement(expr, capturedCtx, [])})`, emitSequence(generated.program), [
        { placeholder: formatJudgement(expr, capturedCtx, []), trace: generated.trace },
      ])
    }
    case 'eval': {
      const generated = compileCore(expr, capturedCtx)
      return composite(judgement, `emit(${formatJudgement(expr, capturedCtx, [])})`, emitSequence(generated.program), [
        { placeholder: formatJudgement(expr, capturedCtx, []), trace: generated.trace },
      ])
    }
  }
}

function leaf(judgement: string, program: Instruction[]): Compiled {
  return { program, trace: [judgement, formatProgram(program)] }
}

function composite(judgement: string, expansion: string, program: Instruction[], children: ChildTrace[]): Compiled {
  const trace = [judgement]
  let current = expansion
  trace.push(current)

  for (const child of children) {
    for (const replacement of child.trace.slice(1)) {
      current = replaceOnce(current, child.placeholder, replacement)
      trace.push(current)
    }
  }

  const finalProgram = formatProgram(program)
  if (trace.at(-1) !== finalProgram) trace.push(finalProgram)
  return { program, trace: dedupeAdjacent(trace) }
}

function replaceOnce(input: string, search: string, replacement: string): string {
  const index = input.indexOf(search)
  if (index < 0) return input
  return `${input.slice(0, index)}${replacement}${input.slice(index + search.length)}`
}

function dedupeAdjacent(lines: string[]): string[] {
  return lines.filter((line, index) => index === 0 || line !== lines[index - 1])
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

function formatExpr(expr: Expr): string {
  switch (expr.type) {
    case 'int':
      return String(expr.value)
    case 'var':
      return expr.name
    case 'lambda':
      return `fn ${expr.param} => ${formatExpr(expr.body)}`
    case 'app':
      return `${parenthesize(expr.fn)} ${parenthesize(expr.arg)}`
    case 'binary':
      return `${parenthesize(expr.left)} ${expr.op} ${parenthesize(expr.right)}`
    case 'code':
      return `code ${parenthesize(expr.body)}`
    case 'lift':
      return `lift ${parenthesize(expr.body)}`
    case 'letCogen':
      return `let cogen ${expr.name} = ${formatExpr(expr.generator)} in ${formatExpr(expr.body)} end`
    case 'eval':
      return `eval ${parenthesize(expr.body)}`
  }
}

function parenthesize(expr: Expr): string {
  if (expr.type === 'int' || expr.type === 'var') return formatExpr(expr)
  return `(${formatExpr(expr)})`
}

function formatContexts(ctx: ContextEntry[], codeVars: string[]): string {
  const omega = ctx.filter((entry) => !entry.isCode).map((entry) => entry.name)
  const lambda = ctx.filter((entry) => entry.isCode || codeVars.includes(entry.name)).map((entry) => entry.name)
  return `Ω=${formatContext(omega)} Λ=${formatContext(lambda)}`
}

function formatJudgement(expr: Expr, ctx: ContextEntry[], codeVars: string[]): string {
  return `[[ ${formatExpr(expr)} ]] ${formatContexts(ctx, codeVars)}`
}

function formatContext(names: string[]): string {
  return names.length === 0 ? '∅' : names.join(',')
}
