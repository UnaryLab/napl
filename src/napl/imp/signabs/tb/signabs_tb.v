`timescale 1ns/1ps
`default_nettype none
// GEN_WIDTH is emitted by gen/gen_signabs.py from the op config (= the test's
// signabs_config), so the DUT parameter is inherited from the Python model.
// iverilog resolves this include relative to the compile cwd (imp/).
`include "signabs/vec/signabs_params.vh"
//==============================================================================
// Self-checking testbench for signabs.
//
// Reads golden vectors produced by gen/gen_signabs.py (from the napl Python
// model) and asserts the saturating-accumulator sign/abs logic reproduces them
// cycle by cycle.
//
// Timing contract (matches the Python forward()): at timestep t the model folds
// i_in into the accumulator and returns sign/abs from the UPDATED value, all in
// one call. In RTL acc is registered and o_sign/o_abs are combinational from the
// current acc + i_in, so per cycle we (1) drive i_in, (2) check o_sign/o_abs
// against the settled combinational outputs, then (3) pulse one posedge i_clk to
// commit acc_next. i_rst_n is held low first so the co-sim starts from the exact
// post-reset() state (acc = ACC_MED).
//
// Prints "PASS ..." iff every vector matches; the Makefile greps for that line.
//
// Run (from src/napl/imp/):
//   make test OP=signabs
//==============================================================================
module signabs_tb;
    reg  clk;
    reg  rst_n;
    reg  in_bit;
    wire sign_bit;
    wire abs_bit;

    signabs #(.WIDTH(`GEN_WIDTH)) dut (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_in    (in_bit),
        .o_sign  (sign_bit),
        .o_abs   (abs_bit)
    );

    integer fd, code, n, fails;
    reg rflag, a, exp_sign, exp_abs;

    // Free-running clock: 10ns period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        in_bit = 1'b0;

        // Assert active-low reset across a clock edge to load acc = ACC_MED, then
        // settle on a negedge with reset released so the first checked state is
        // the exact post-reset() accumulator.
        rst_n = 1'b0;
        @(negedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);

        fd = $fopen("vec/signabs.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/signabs.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b %b\n", rflag, a, exp_sign, exp_abs);
            if (code == 4) begin
                // Mid-stream reset: when rflag is set, the model called reset()
                // BEFORE this timestep, so re-load acc = ACC_MED here to prove
                // the RTL recovers the post-reset() state from a dirtied acc.
                if (rflag) begin
                    rst_n = 1'b0;
                    @(posedge clk);   // async reset loads acc = ACC_MED
                    @(negedge clk);
                    rst_n = 1'b1;
                end
                // Drive this cycle's input; o_sign/o_abs settle combinationally
                // from the current acc + i_in (the UPDATED-acc outputs the model
                // returns this timestep). Check, then posedge to commit acc_next.
                in_bit = a;
                #1;   // let the combinational outputs settle
                n = n + 1;
                if (sign_bit !== exp_sign || abs_bit !== exp_abs) begin
                    $display("FAIL cyc=%0d in=%b : got sign=%b abs=%b exp sign=%b abs=%b",
                             n, a, sign_bit, abs_bit, exp_sign, exp_abs);
                    fails = fails + 1;
                end
                @(posedge clk);   // commit acc_next
                @(negedge clk);   // settle for the next check
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS signabs: %0d/%0d vectors", n, n);
        else
            $display("FAIL signabs: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
