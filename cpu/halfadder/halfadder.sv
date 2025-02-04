module HalfAdder (
    input logic a, b,
    output logic sum, carry
);
    assign sum = a ^ b;
    assign carry = a & b;
endmodule
