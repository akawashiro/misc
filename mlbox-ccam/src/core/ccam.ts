export type Value =
  | { type: 'unit' }
  | { type: 'int'; value: number }
  | { type: 'pair'; left: Value; right: Value }
  | { type: 'closure'; env: Value; program: Instruction[] }
  | { type: 'block'; program: Instruction[] }

export type Instruction =
  | { op: 'id' }
  | { op: 'fst' }
  | { op: 'snd' }
  | { op: 'push' }
  | { op: 'swap' }
  | { op: 'cons' }
  | { op: 'app' }
  | { op: 'quote'; value: Value }
  | { op: 'cur'; program: Instruction[] }
  | { op: 'emit'; instruction: Instruction }
  | { op: 'lift' }
  | { op: 'arena' }
  | { op: 'merge'; program: Instruction[] }
  | { op: 'call' }
  | { op: 'evalLift'; program: Instruction[] }
  | { op: 'add' }
  | { op: 'sub' }
  | { op: 'mul' }

export type Transition = {
  step: number
  stack: string
  program: string
}

export const unit: Value = { type: 'unit' }

export function run(program: Instruction[], env: Value = unit, limit = 300): { value: Value; transitions: Transition[] } {
  const stack: Value[] = [env]
  let instructions = [...program]
  const transitions: Transition[] = []
  let step = 0

  while (instructions.length > 0) {
    if (step >= limit) throw new Error(`CCAM step limit exceeded (${limit})`)
    transitions.push({ step, stack: formatStack(stack), program: formatProgram(instructions) })
    const instruction = instructions.shift()
    if (!instruction) break
    execute(instruction, stack, (extra) => {
      instructions = [...extra, ...instructions]
    })
    step += 1
  }

  transitions.push({ step, stack: formatStack(stack), program: '.' })
  if (stack.length === 0) throw new Error('CCAM stack is empty')
  return { value: stack[0], transitions }
}

function execute(instruction: Instruction, stack: Value[], prepend: (program: Instruction[]) => void): void {
  switch (instruction.op) {
    case 'id':
      return
    case 'fst':
      stack[0] = asPair(stack[0]).left
      return
    case 'snd':
      stack[0] = asPair(stack[0]).right
      return
    case 'push':
      stack.unshift(stack[0])
      return
    case 'swap': {
      const first = stack[0]
      stack[0] = stack[1]
      stack[1] = first
      return
    }
    case 'cons': {
      const right = stack.shift()
      const left = stack.shift()
      if (!left || !right) throw new Error('cons requires two stack values')
      stack.unshift({ type: 'pair', left, right })
      return
    }
    case 'app': {
      const pair = asPair(stack.shift())
      const closure = asClosure(pair.left)
      stack.unshift({ type: 'pair', left: closure.env, right: pair.right })
      prepend(closure.program)
      return
    }
    case 'quote':
      stack[0] = cloneValue(instruction.value)
      return
    case 'cur':
      stack[0] = { type: 'closure', env: stack[0], program: instruction.program }
      return
    case 'emit':
      currentBlock(stack[0]).program.push(instruction.instruction)
      return
    case 'lift': {
      const env = asPair(stack[0])
      asBlock(env.right).program.push({ op: 'quote', value: cloneValue(env.left) })
      return
    }
    case 'arena':
      stack.unshift({ type: 'block', program: [] })
      return
    case 'merge':
      currentBlock(stack[0]).program.push({ op: 'cur', program: instruction.program })
      return
    case 'call': {
      const block = asBlock(stack.shift())
      prepend(block.program)
      return
    }
    case 'evalLift': {
      const env = asPair(stack[0])
      const result = run(instruction.program, env.left).value
      asBlock(env.right).program.push({ op: 'quote', value: result })
      return
    }
    case 'add':
    case 'sub':
    case 'mul': {
      const pair = asPair(stack[0])
      const left = asInt(pair.left)
      const right = asInt(pair.right)
      const value =
        instruction.op === 'add' ? left + right : instruction.op === 'sub' ? left - right : left * right
      stack[0] = { type: 'int', value }
      return
    }
  }
}

function currentBlock(value: Value): Extract<Value, { type: 'block' }> {
  return asBlock(asPair(value).right)
}

function asPair(value: Value | undefined): Extract<Value, { type: 'pair' }> {
  if (!value || value.type !== 'pair') throw new Error(`Expected pair, got ${formatValue(value)}`)
  return value
}

function asClosure(value: Value): Extract<Value, { type: 'closure' }> {
  if (value.type !== 'closure') throw new Error(`Expected closure, got ${formatValue(value)}`)
  return value
}

function asBlock(value: Value | undefined): Extract<Value, { type: 'block' }> {
  if (!value || value.type !== 'block') throw new Error(`Expected code block, got ${formatValue(value)}`)
  return value
}

function asInt(value: Value): number {
  if (value.type !== 'int') throw new Error(`Expected int, got ${formatValue(value)}`)
  return value.value
}

function cloneValue(value: Value): Value {
  if (value.type === 'pair') return { type: 'pair', left: cloneValue(value.left), right: cloneValue(value.right) }
  if (value.type === 'closure') return { type: 'closure', env: cloneValue(value.env), program: value.program }
  if (value.type === 'block') return { type: 'block', program: [...value.program] }
  return { ...value }
}

export function formatProgram(program: Instruction[]): string {
  if (program.length === 0) return '.'
  return program.map(formatInstruction).join('; ')
}

export function formatInstruction(instruction: Instruction): string {
  switch (instruction.op) {
    case 'quote':
      return `'${formatValue(instruction.value)}`
    case 'cur':
      return `Cur(${formatProgram(instruction.program)})`
    case 'emit':
      return `emit(${formatInstruction(instruction.instruction)})`
    case 'merge':
      return `merge(Cur(${formatProgram(instruction.program)}))`
    case 'evalLift':
      return `lift[${formatProgram(instruction.program)}]`
    default:
      return instruction.op
  }
}

export function formatStack(stack: Value[]): string {
  return `[${stack.map(formatValue).join(' :: ')}]`
}

export function formatValue(value: Value | undefined): string {
  if (!value) return '<empty>'
  switch (value.type) {
    case 'unit':
      return '()'
    case 'int':
      return String(value.value)
    case 'pair':
      return `(${formatValue(value.left)}, ${formatValue(value.right)})`
    case 'closure':
      return `[${formatValue(value.env)} : ${formatProgram(value.program)}]`
    case 'block':
      return `{${formatProgram(value.program)}}`
  }
}
