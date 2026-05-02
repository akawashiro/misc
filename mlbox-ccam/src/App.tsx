import { useMemo, useState } from 'react'
import './App.css'
import { compile } from './core/compiler'
import { formatValue, run } from './core/ccam'
import { parse } from './core/parser'
import {
  assembleRv32,
  createRv32Machine,
  disassembleRv32Word,
  formatRv32Instruction,
  formatRv32Register,
  formatRv32Word,
  stepRv32,
} from './core/riscv'
import type { Rv32State, Rv32StepResult } from './core/riscv'

const samples = [
  {
    name: 'minimal code run',
    source: `let cogen result = code 1 in
  result
end`,
  },
  {
    name: 'let cogen + lift',
    source: `let cogen result = (
  let cogen a = lift (6 * 7) in
    code (a + 8)
  end
) in
  result
end`,
  },
  {
    name: 'let cogen + code',
    source: `let cogen result = (
  let cogen u = code 1 in
    code u
  end
) in
  result
end`,
  },
  {
    name: 'arithmetic',
    source: `6 * 7 + 8`,
  },
  {
    name: 'lambda application',
    source: `(fn x => x * x + 1) 9`,
  },
  {
    name: 'code run',
    source: `let cogen result = code ((20 + 1) * 2) in
  result
end`,
  },
  {
    name: 'generated function',
    source: `(let cogen f = code (fn x => x + 10) in
  f
end) 32`,
  },
  {
    name: 'code substitution',
    source: `let cogen result = (
  let cogen base = lift 40 in
    let cogen delta = lift 2 in
      code (base + delta)
    end
  end
) in
  result
end`,
  },
  {
    name: 'Figure 4: normal variable',
    source: `fn x => x`,
  },
  {
    name: 'Figure 4: normal lambda',
    source: `fn x => x + 1`,
  },
  {
    name: 'Figure 4: normal application',
    source: `(fn x => x) 1`,
  },
  {
    name: 'Figure 4: normal code variable',
    source: `let cogen u = code 1 in
  u
end`,
  },
  {
    name: 'Figure 4: normal code',
    source: `code 1`,
  },
  {
    name: 'Figure 4: normal lift',
    source: `lift (1 + 2)`,
  },
  {
    name: 'Figure 4: normal let cogen',
    source: `let cogen u = code 1 in
  2
end`,
  },
  {
    name: 'Figure 4: generator variable',
    source: `(fn x => code x) 1`,
  },
  {
    name: 'Figure 4: generator lambda',
    source: `code (fn x => x)`,
  },
  {
    name: 'Figure 4: generator application',
    source: `code ((fn x => x) 1)`,
  },
  {
    name: 'Figure 4: generator code variable in Omega',
    source: `code (let cogen u = code 1 in
  u
end)`,
  },
  {
    name: 'Figure 4: generator code variable in Lambda',
    source: `let cogen u = code 1 in
  code u
end`,
  },
  {
    name: 'Figure 4: generator nested code',
    source: `code (code 1)`,
  },
  {
    name: 'Figure 4: generator lift',
    source: `code (lift 1)`,
  },
  {
    name: 'Figure 4: generator let cogen',
    source: `code (let cogen u = code 1 in
  2
end)`,
  },
]

const sortedSamples = [...samples].sort((left, right) => left.name.localeCompare(right.name))

const rv32MemorySize = 64 * 1024
const rv32InitialSp = rv32MemorySize
const rv32DisassemblyRadius = 3

const rv32Samples = [
  {
    name: 'Load/store round trip',
    source: `addi x1, x0, 42
addi x2, x2, -16
sw x1, 12(x2)
lw x3, 12(x2)
ecall`,
  },
  {
    name: 'Fib(10) loop',
    source: `# Fib(10): result is x6 = 55
addi x5, x0, 10
addi x6, x0, 0
addi x7, x0, 1
add x8, x6, x7
add x6, x7, x0
add x7, x8, x0
addi x5, x5, -1
bne x5, x0, -16
ecall`,
  },
  {
    name: 'Stack machine 1 + 2',
    source: `# Stack machine style: computes 1 + 2
addi x2, x2, -4
addi x5, x0, 1
sw x5, 0(x2)
addi x2, x2, -4
addi x5, x0, 2
sw x5, 0(x2)
lw x5, 0(x2)
addi x2, x2, 4
lw x6, 0(x2)
addi x2, x2, 4
add x7, x6, x5
addi x2, x2, -4
sw x7, 0(x2)
ecall`,
  },
]

const sortedRv32Samples = [...rv32Samples].sort((left, right) => left.name.localeCompare(right.name))
const defaultRv32Source = rv32Samples[0].source

type Rv32UiState = {
  machine: Rv32State | null
  steps: Rv32StepResult[]
  error: string | null
}

type Rv32DisassemblyRow = {
  address: number
  word: number | null
  text: string
  current: boolean
}

type Rv32MemoryRow = {
  address: number
  bytes: string[]
  includesSp: boolean
}

function createRv32UiState(source: string): Rv32UiState {
  try {
    const program = wordsFromBytes(assembleRv32(source))
    const regs = new Uint32Array(32)
    regs[2] = rv32InitialSp
    return {
      machine: createRv32Machine(program, { regs, memorySize: rv32MemorySize }),
      steps: [],
      error: null,
    }
  } catch (error) {
    return {
      machine: null,
      steps: [],
      error: error instanceof Error ? error.message : String(error),
    }
  }
}

function wordsFromBytes(bytes: Uint8Array): number[] {
  const words: number[] = []
  for (let index = 0; index < bytes.length; index += 4) {
    words.push(readUint32(bytes, index))
  }
  return words
}

function rv32DisassemblyRows(machine: Rv32State): Rv32DisassemblyRow[] {
  const pc = machine.pc
  const center = Math.floor(pc / 4) * 4
  const start = Math.max(0, center - rv32DisassemblyRadius * 4)
  const rows: Rv32DisassemblyRow[] = []

  for (let address = start; address <= center + rv32DisassemblyRadius * 4; address += 4) {
    if (address > machine.memory.length - 4) {
      rows.push({ address, word: null, text: 'outside memory', current: address === center })
      continue
    }

    const word = readUint32(machine.memory, address)
    try {
      rows.push({ address, word, text: disassembleRv32Word(word, address), current: address === center })
    } catch (error) {
      rows.push({
        address,
        word,
        text: error instanceof Error ? error.message : String(error),
        current: address === center,
      })
    }
  }

  return rows
}

function rv32StackRows(machine: Rv32State): Rv32MemoryRow[] {
  const sp = machine.regs[2]
  const rowSize = 16
  const start = alignDown(Math.min(sp, machine.memory.length - 1), rowSize)
  const clampedStart = Math.max(0, Math.min(machine.memory.length - rowSize, start - rowSize * 2))
  const rows: Rv32MemoryRow[] = []

  for (let address = clampedStart; address < clampedStart + rowSize * 5 && address < machine.memory.length; address += rowSize) {
    const bytes: string[] = []
    for (let offset = 0; offset < rowSize && address + offset < machine.memory.length; offset += 1) {
      bytes.push(machine.memory[address + offset].toString(16).padStart(2, '0'))
    }
    rows.push({ address, bytes, includesSp: sp >= address && sp < address + rowSize })
  }

  return rows
}

function readUint32(bytes: Uint8Array, address: number): number {
  return (bytes[address] | (bytes[address + 1] << 8) | (bytes[address + 2] << 16) | (bytes[address + 3] << 24)) >>> 0
}

function alignDown(value: number, alignment: number): number {
  return value - (value % alignment)
}

function App() {
  const [source, setSource] = useState(samples[0].source)
  const [rv32Source, setRv32Source] = useState(defaultRv32Source)
  const [rv32State, setRv32State] = useState<Rv32UiState>(() => createRv32UiState(defaultRv32Source))

  const result = useMemo(() => {
    try {
      const ast = parse(source)
      const compiled = compile(ast)
      const executed = run(compiled.program)
      return { ast, compiled, executed, error: null }
    } catch (error) {
      return { ast: null, compiled: null, executed: null, error: error instanceof Error ? error.message : String(error) }
    }
  }, [source])

  const rv32Machine = rv32State.machine
  const rv32Trap = rv32Machine?.trap ?? rv32State.steps.at(-1)?.trap
  const rv32LastStep = rv32State.steps.at(-1)
  const rv32Disassembly = rv32Machine ? rv32DisassemblyRows(rv32Machine) : []
  const rv32Stack = rv32Machine ? rv32StackRows(rv32Machine) : []

  function resetRv32(): void {
    setRv32State(createRv32UiState(rv32Source))
  }

  function selectRv32Sample(name: string): void {
    const sample = sortedRv32Samples.find((item) => item.name === name)
    if (!sample) return
    setRv32Source(sample.source)
    setRv32State(createRv32UiState(sample.source))
  }

  function stepRv32Once(): void {
    setRv32State((current) => {
      if (!current.machine || current.machine.halted) return current
      const step = stepRv32(current.machine)
      return { ...current, steps: [...current.steps, step], error: null }
    })
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>ML^box to CCAM</h1>
          <p>
            Modal ML terms are parsed, compiled, and executed by a client-side CCAM simulator.{' '}
            <a href="https://dl.acm.org/doi/10.1145/277652.277727" target="_blank" rel="noreferrer">
              Run-time Code Generation and Modal-ML
            </a>
          </p>
        </div>
      </header>

      <section className="workspace">
        <section className="panel input-panel">
          <div className="panel-header">
            <h2>ML^box Input</h2>
            <label className="sample-picker">
              <span>Sample</span>
              <select
                value={samples.find((sample) => sample.source === source)?.name ?? 'custom'}
                onChange={(event) => {
                  const sample = sortedSamples.find((item) => item.name === event.target.value)
                  if (sample) setSource(sample.source)
                }}
              >
                {samples.find((sample) => sample.source === source) ? null : <option value="custom">Custom</option>}
                {sortedSamples.map((sample) => (
                  <option key={sample.name} value={sample.name}>
                    {sample.name}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <textarea
            value={source}
            onChange={(event) => setSource(event.target.value)}
            spellCheck={false}
            aria-label="ML box source"
          />
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>Compile Trace</h2>
          </div>
          {result.error ? (
            <pre className="error">{result.error}</pre>
          ) : (
            <ol className="trace-list">
              {result.compiled!.log.map((line, index) => (
                <li key={`${index}-${line}`}>
                  <code>{line}</code>
                </li>
              ))}
            </ol>
          )}
        </section>

        <section className="panel run-panel">
          <div className="panel-header">
            <h2>CCAM Run</h2>
            {!result.error && <span>{result.executed!.transitions.length - 1} transitions</span>}
          </div>
          {result.error ? (
            <pre className="error">{result.error}</pre>
          ) : (
            <>
              <div className="summary">
                <span>result</span>
                <code>{formatValue(result.executed!.value)}</code>
              </div>
              <div className="transition-table">
                <div className="row heading">
                  <span>#</span>
                  <span>Stack</span>
                  <span>Program</span>
                </div>
                {result.executed!.transitions.map((transition) => (
                  <div className="row" key={transition.step}>
                    <span>{transition.step}</span>
                    <code>{transition.stack}</code>
                    <code>{transition.program}</code>
                  </div>
                ))}
              </div>
            </>
          )}
        </section>

        <section className="panel riscv-panel">
          <div className="panel-header">
            <h2>RISC-V Emulator</h2>
            <span>{rv32Machine ? `${rv32State.steps.length} steps` : 'not loaded'}</span>
          </div>
          <div className="riscv-layout">
            <div className="riscv-editor">
              <div className="riscv-toolbar">
                <label className="sample-picker riscv-sample-picker">
                  <span>Sample</span>
                  <select
                    value={rv32Samples.find((sample) => sample.source === rv32Source)?.name ?? 'custom'}
                    onChange={(event) => selectRv32Sample(event.target.value)}
                  >
                    {rv32Samples.find((sample) => sample.source === rv32Source) ? null : <option value="custom">Custom</option>}
                    {sortedRv32Samples.map((sample) => (
                      <option key={sample.name} value={sample.name}>
                        {sample.name}
                      </option>
                    ))}
                  </select>
                </label>
                <button type="button" onClick={resetRv32}>
                  Reset
                </button>
                <button type="button" onClick={stepRv32Once} disabled={!rv32Machine || rv32Machine.halted}>
                  Step
                </button>
              </div>
              <textarea
                className="riscv-source"
                value={rv32Source}
                onChange={(event) => setRv32Source(event.target.value)}
                spellCheck={false}
                aria-label="RISC-V assembly source"
              />
              {rv32State.error && <pre className="error">{rv32State.error}</pre>}
            </div>

            <div className="riscv-state">
              <div className="riscv-status">
                <div>
                  <span>PC</span>
                  <code>{rv32Machine ? formatRv32Word(rv32Machine.pc) : '-'}</code>
                </div>
                <div>
                  <span>SP</span>
                  <code>{rv32Machine ? formatRv32Word(rv32Machine.regs[2]) : '-'}</code>
                </div>
                <div>
                  <span>Status</span>
                  <code>{rv32Machine?.halted ? rv32Trap?.reason ?? 'halted' : 'running'}</code>
                </div>
                <div>
                  <span>Last</span>
                  <code>{rv32LastStep ? formatRv32Instruction(rv32LastStep.instruction) : '-'}</code>
                </div>
              </div>
              {rv32Trap && <pre className="trap">{rv32Trap.message}</pre>}

              <div className="riscv-grid">
                <section className="riscv-block">
                  <h3>Registers</h3>
                  <div className="register-grid">
                    {rv32Machine
                      ? Array.from(rv32Machine.regs).map((value, index) => (
                          <div className="register-cell" key={index}>
                            <span>{formatRv32Register(index)}</span>
                            <code>{formatRv32Word(value)}</code>
                          </div>
                        ))
                      : null}
                  </div>
                </section>

                <section className="riscv-block">
                  <h3>Disassembly</h3>
                  <div className="disassembly-list">
                    {rv32Disassembly.map((row) => (
                      <div className={row.current ? 'disassembly-row current' : 'disassembly-row'} key={row.address}>
                        <span>{row.current ? 'PC' : ''}</span>
                        <code>{formatRv32Word(row.address)}</code>
                        <code>{row.word === null ? '--------' : formatRv32Word(row.word)}</code>
                        <code>{row.text}</code>
                      </div>
                    ))}
                  </div>
                </section>
              </div>

              <section className="riscv-block">
                <h3>Stack Memory</h3>
                <div className="memory-dump">
                  {rv32Stack.map((row) => (
                    <div className={row.includesSp ? 'memory-row current' : 'memory-row'} key={row.address}>
                      <span>{row.includesSp ? 'SP' : ''}</span>
                      <code>{formatRv32Word(row.address)}</code>
                      <code>{row.bytes.join(' ')}</code>
                    </div>
                  ))}
                </div>
              </section>
            </div>
          </div>
        </section>
      </section>
    </main>
  )
}

export default App
