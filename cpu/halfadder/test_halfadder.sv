`include "halfadder.sv"

module test_halfadder;
    reg a, b;
    wire sum, carry;

    HalfAdder ha(a, b, sum, carry);

    initial begin
        a = 0; b = 0;
        #10 $display("a=%b, b=%b, sum=%b, carry=%b", a, b, sum, carry);
        a = 0; b = 1;
        #10 $display("a=%b, b=%b, sum=%b, carry=%b", a, b, sum, carry);
        a = 1; b = 0;
        #10 $display("a=%b, b=%b, sum=%b, carry=%b", a, b, sum, carry);
        a = 1; b = 1;
        #10 $display("a=%b, b=%b, sum=%b, carry=%b", a, b, sum, carry);
    end
endmodule
