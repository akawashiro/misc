import { describe, expect, it } from 'vitest'
import { assembleRv32, assembleRv32Line, disassembleRv32, disassembleRv32Word } from './riscv'

const allInstructions = [
  'lui x1, 305418240',
  'auipc x2, 4096',
  'jal x3, 8',
  'jalr x4, -4(x5)',
  'beq x1, x2, 8',
  'bne x1, x2, -4',
  'blt x1, x2, 6',
  'bge x1, x2, -6',
  'bltu x1, x2, 10',
  'bgeu x1, x2, -10',
  'lb x6, -8(x7)',
  'lh x6, -6(x7)',
  'lw x6, 0(x7)',
  'lbu x6, 6(x7)',
  'lhu x6, 8(x7)',
  'sb x8, -8(x9)',
  'sh x8, -6(x9)',
  'sw x8, 0(x9)',
  'addi x10, x11, -2048',
  'slti x10, x11, 2047',
  'sltiu x10, x11, -1',
  'xori x10, x11, 123',
  'ori x10, x11, -123',
  'andi x10, x11, 255',
  'slli x12, x13, 0',
  'srli x12, x13, 31',
  'srai x12, x13, 31',
  'add x14, x15, x16',
  'sub x14, x15, x16',
  'sll x14, x15, x16',
  'slt x14, x15, x16',
  'sltu x14, x15, x16',
  'xor x14, x15, x16',
  'srl x14, x15, x16',
  'sra x14, x15, x16',
  'or x14, x15, x16',
  'and x14, x15, x16',
  'fence',
  'ecall',
  'ebreak',
]

function roundTrip(instructions1: string): { bytes2: Uint8Array; instructions3: string; bytes4: Uint8Array } {
  const bytes2 = assembleRv32(instructions1)
  const instructions3 = disassembleRv32(bytes2)
  const bytes4 = assembleRv32(instructions3)
  expect(instructions3).toBe(instructions1)
  expect(Array.from(bytes4)).toEqual(Array.from(bytes2))
  return { bytes2, instructions3, bytes4 }
}

describe('RV32I assembler/disassembler', () => {
  it('round-trips every supported RV32I mnemonic', () => {
    const instructions1 = allInstructions.join('\n')
    roundTrip(instructions1)
  })

  it('emits little-endian bytes', () => {
    expect(Array.from(assembleRv32('addi x1, x0, 1'))).toEqual([0x93, 0x00, 0x10, 0x00])
  })

  it('round-trips boundary immediates', () => {
    roundTrip(
      [
        'addi x1, x2, -2048',
        'addi x1, x2, 2047',
        'sw x1, -2048(x2)',
        'sw x1, 2047(x2)',
        'beq x1, x2, -4096',
        'beq x1, x2, 4094',
        'jal x1, -1048576',
        'jal x1, 1048574',
        'lui x1, -2147483648',
        'lui x1, 2147479552',
        'slli x1, x2, 0',
        'slli x1, x2, 31',
      ].join('\n'),
    )
  })

  it('canonicalizes whitespace and immediate syntax through disassembly', () => {
    const bytes2 = assembleRv32('addi   x1,   x0,   0x2a\njalr x2, -0x4(x3)')
    const instructions3 = disassembleRv32(bytes2)
    const bytes4 = assembleRv32(instructions3)

    expect(instructions3).toBe('addi x1, x0, 42\njalr x2, -4(x3)')
    expect(Array.from(bytes4)).toEqual(Array.from(bytes2))
  })

  it('ignores hash and semicolon comments in assembly source', () => {
    const bytes = assembleRv32(
      [
        '# full-line comment',
        'addi x1, x0, 1 # trailing comment',
        '; another full-line comment',
        'addi x2, x1, 2 ; semicolon trailing comment',
      ].join('\n'),
    )

    expect(disassembleRv32(bytes)).toBe('addi x1, x0, 1\naddi x2, x1, 2')
  })

  it('disassembles single words', () => {
    const word = assembleRv32Line('sub x1, x2, x3')
    expect(disassembleRv32Word(word)).toBe('sub x1, x2, x3')
  })

  it('rejects unsupported assembly syntax and invalid operands', () => {
    expect(() => assembleRv32Line('unknown x1, x2, x3')).toThrow('Unknown RV32I mnemonic')
    expect(() => assembleRv32Line('addi ra, x0, 1')).toThrow('Invalid RV32I register')
    expect(() => assembleRv32Line('addi x1, x0, 2048')).toThrow('immediate out of range')
    expect(() => assembleRv32Line('beq x1, x2, 3')).toThrow('2-byte aligned')
    expect(() => assembleRv32Line('jal x1, 3')).toThrow('2-byte aligned')
    expect(() => assembleRv32Line('target: addi x1, x0, 1')).toThrow('labels are not supported')
    expect(() => assembleRv32Line('li x1, 1')).toThrow('Unknown RV32I mnemonic')
  })

  it('rejects invalid disassembly input', () => {
    expect(() => disassembleRv32(new Uint8Array([1, 2, 3]))).toThrow('multiple of 4')
    expect(() => disassembleRv32(new Uint8Array([0xff, 0xff, 0xff, 0xff]))).toThrow('Illegal RV32I instruction')
  })
})
