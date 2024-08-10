module MN-CORE2

  // syntax Program ::
  syntax PackedInstruction ::= "Packed"
  syntax Color ::= Yellow() | Blue()
  syntax Fruit ::= Banana() | Blueberry()
  syntax Color ::= colorOf(Fruit) [function]

  rule colorOf(Banana()) => colorOf(Blueberry())
  rule colorOf(Blueberry()) => Blue()

endmodule
