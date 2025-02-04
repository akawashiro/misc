`include "flop.sv"

module test_flop;
    logic clk;
    logic [3:0] d;
    logic [3:0] q;

    flop uut (
        .clk(clk),
        .d(d),
        .q(q)
    );

    initial begin
        clk = 0;
        d = 3'b101;
        clk = 1;
        #10 $display("q = %b", q);
        clk = 0;
        d = 3'b010;
        clk = 1;
        #10 $display("q = %b", q);
    end
endmodule
