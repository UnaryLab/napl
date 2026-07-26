`timescale 1ns/1ps
`default_nettype none
// GEN_SCALE / GEN_WIDTH / GEN_ENTRY are emitted by gen/gen_add_any.py from the op
// config (= the test's add_any_config + reduction dim), so the DUT parameters are
// inherited from the Python model. iverilog resolves this include relative to the
// compile cwd (imp/).
`include "add_any/vec/add_any_params.vh"
//==============================================================================
// Self-checking testbench for add_any (unipolar + bipolar variants).
//
// Reads golden vectors produced by gen/gen_add_any.py (from the napl Python
// model) and asserts both RTL modules reproduce them cycle-for-cycle. add_any is
// stateful, so each vector row is one clock: the input partial sum is driven one
// value per posedge i_clk and o_out is sampled in the same cycle (combinational
// from state, pp_delay=0). Each row carries a `rst` flag: rst=1 marks a cycle the
// model was reset() immediately before, so the tb pulses i_rst_n low (async clears
// the accumulators to 0) before driving that row. This covers both the opening
// reset and a MID-STREAM reset (the dirtied-state reset-equivalence proof).
// Prints "PASS ..." iff every vector matches both modules; the Makefile greps for
// that line.
//
// SCALE/WIDTH/ENTRY come from the param header so the verified hardware tracks the
// simulator's configuration exactly.
//
// Run (from src/napl/imp/):
//   make test OP=add_any
//==============================================================================
module add_any_tb;
    // input port width: partial sum in [0, ENTRY] -> need clog2(ENTRY+1) bits.
    localparam integer IN_W = clog2(`GEN_ENTRY + 1);

    function integer clog2;
        input integer value;
        integer v;
        begin
            v = value - 1;
            for (clog2 = 0; v > 0; clog2 = clog2 + 1)
                v = v >> 1;
        end
    endfunction

    reg              i_clk;
    reg              i_rst_n;
    reg  [IN_W-1:0]  i_input;
    wire             o_uni;
    wire             o_bi;

    add_any_unipolar #(
        .SCALE(`GEN_SCALE),
        .WIDTH(`GEN_WIDTH),
        .ENTRY(`GEN_ENTRY)
    ) dut_uni (
        .i_clk  (i_clk),
        .i_rst_n(i_rst_n),
        .i_input   (i_input),
        .o_out  (o_uni)
    );

    add_any_bipolar #(
        .SCALE(`GEN_SCALE),
        .WIDTH(`GEN_WIDTH),
        .ENTRY(`GEN_ENTRY)
    ) dut_bi (
        .i_clk  (i_clk),
        .i_rst_n(i_rst_n),
        .i_input   (i_input),
        .o_out  (o_bi)
    );

    // 10ns clock
    initial i_clk = 1'b0;
    always #5 i_clk = ~i_clk;

    integer fd, code, n, fails;
    integer part;
    reg rst_flag, exp_uni, exp_bi;

    initial begin
        fd = $fopen("vec/add_any.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/add_any.vec (run `make vectors` first)");
            $finish;
        end

        i_input    = {IN_W{1'b0}};
        i_rst_n = 1'b1;

        n = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %d %b %b\n", rst_flag, part, exp_uni, exp_bi);
            if (code == 4) begin
                // rst=1: model was reset() right before this cycle. Pulse i_rst_n
                // low across a posedge so the async reset clears the accumulators
                // to 0, then release on the negedge before driving this row.
                if (rst_flag) begin
                    i_rst_n = 1'b0;
                    @(posedge i_clk);   // async reset fires here (acc <= 0)
                    @(negedge i_clk);
                    i_rst_n = 1'b1;
                end
                // Drive the input just after a falling edge so it is stable at the
                // rising edge; o_out is combinational from the accumulators.
                i_input = part[IN_W-1:0];
                #1;                     // let the combinational outputs settle
                n = n + 1;
                if (o_uni !== exp_uni) begin
                    $display("FAIL cycle %0d unipolar: i_input=%0d got %b exp %b", n, part, o_uni, exp_uni);
                    fails = fails + 1;
                end
                if (o_bi !== exp_bi) begin
                    $display("FAIL cycle %0d bipolar: i_input=%0d got %b exp %b", n, part, o_bi, exp_bi);
                    fails = fails + 1;
                end
                @(posedge i_clk);       // advance the accumulator state
                @(negedge i_clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS add_any: %0d/%0d vectors (unipolar + bipolar)", n, n);
        else
            $display("FAIL add_any: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
