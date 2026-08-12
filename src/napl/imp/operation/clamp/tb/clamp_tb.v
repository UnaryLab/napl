`timescale 1ns/1ps
`default_nettype none
// Generated parameters mirror the Python model configuration.
`include "clamp/vec/clamp_params.vh"
// Golden rows are <rst> <unipolar input/output> <bipolar input/output>. Reset
// precedes the marked row; outputs are checked before the posedge updates state.
// Co-sim: make test OP=clamp


module clamp_tb;
    reg  i_clk;
    reg  i_rst_n;
    reg  i_input_uni;
    reg  i_input_bi;
    wire o_uni;
    wire o_bi;

    clamp_unipolar #(
        .LO(`GEN_LO_UNI),
        .HI(`GEN_HI_UNI),
        .FRACWIDTH(`GEN_FRACWIDTH),
        .TIMESTEP(`GEN_TIMESTEP)
    ) dut_uni (
        .i_clk  (i_clk),
        .i_rst_n(i_rst_n),
        .i_input   (i_input_uni),
        .o_output  (o_uni)
    );

    clamp_bipolar #(
        .LO(`GEN_LO_BI),
        .HI(`GEN_HI_BI),
        .FRACWIDTH(`GEN_FRACWIDTH),
        .TIMESTEP(`GEN_TIMESTEP)
    ) dut_bi (
        .i_clk  (i_clk),
        .i_rst_n(i_rst_n),
        .i_input   (i_input_bi),
        .o_output  (o_bi)
    );

    // 10 ns clock period.
    initial i_clk = 1'b0;
    always #5 i_clk = ~i_clk;

    integer fd, code, n, fails;
    reg rst_flag, exp_uni, exp_bi;

    initial begin
        fd = $fopen("vec/clamp.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/clamp.vec (run `make vectors` first)");
            $finish;
        end

        i_input_uni = 1'b0;
        i_input_bi  = 1'b0;
        i_rst_n = 1'b1;

        n = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(
                fd, "%b %b %b %b %b\n",
                rst_flag, i_input_uni, exp_uni, i_input_bi, exp_bi
            );
            if (code == 5) begin
                if (rst_flag) begin
                    i_rst_n = 1'b0;
                    @(posedge i_clk);   // async reset fires here (ts/acc/em <= 0)
                    @(negedge i_clk);
                    i_rst_n = 1'b1;
                end
                #1;                     // let the combinational outputs settle
                n = n + 1;
                if (o_uni !== exp_uni) begin
                    $display("FAIL cycle %0d unipolar: i_input=%b got %b exp %b", n, i_input_uni, o_uni, exp_uni);
                    fails = fails + 1;
                end
                if (o_bi !== exp_bi) begin
                    $display("FAIL cycle %0d bipolar: i_input=%b got %b exp %b", n, i_input_bi, o_bi, exp_bi);
                    fails = fails + 1;
                end
                @(posedge i_clk);       // advance the counter state
                @(negedge i_clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS clamp: %0d/%0d vectors (unipolar + bipolar)", n, n);
        else
            $display("FAIL clamp: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
