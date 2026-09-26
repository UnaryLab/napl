`timescale 1ns/1ps
`default_nettype none
`include "mul_unibi_mux/vec/mul_unibi_mux_params.vh"
// Golden rows are <rst> <input_u> <input_b> <output>, produced by the Python
// model. The output is combinational, so it is checked before the posedge that
// advances the toggle flip-flop; a marked row resets that flip-flop first.
// Co-sim: make test OP=mul_unibi_mux


module mul_unibi_mux_tb;
    reg  i_clk;
    reg  i_rst_n;
    reg  i_input_u;
    reg  i_input_b;
    wire o_output;

    mul_unibi_mux dut (
        .i_clk    (i_clk),
        .i_rst_n  (i_rst_n),
        .i_input_u(i_input_u),
        .i_input_b(i_input_b),
        .o_output (o_output)
    );

    // 10 ns clock period.
    initial i_clk = 1'b0;
    always #5 i_clk = ~i_clk;

    // LINE_BYTES sizes the line buffer, which holds LINE_BYTES bytes, so a
    // newline-terminated row carries at most LINE_BYTES-1 payload bytes, and
    // DATA_COLS is the column count every data row carries. Neither constant
    // can be set wrong and still let a malformed row through, because the
    // trailing sentinel makes the column check two-sided. A DATA_COLS set too
    // large fatals on the first clean row. A LINE_BYTES set too large costs
    // guard precision rather than correctness, because the length assertion
    // can then fire only at the buffer boundary instead of near the true row
    // width, so both constants are measured from the vec file rather than
    // guessed.
    localparam LINE_BYTES = 16;
    localparam DATA_COLS  = 4;

    integer fd, code, chars, n, fails;
    reg rst_flag, in_u_s, in_b_s, exp_out;
    reg [8*8-1:0] extra;
    reg [8*8-1:0] tok_rst_flag;
    reg [8*8-1:0] tok_in_u_s;
    reg [8*8-1:0] tok_in_b_s;
    reg [8*8-1:0] tok_exp_out;
    reg [8*LINE_BYTES-1:0] line;

    initial begin
        fd = $fopen("vec/mul_unibi_mux.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/mul_unibi_mux.vec (run `make vectors` first)");
            $finish;
        end

        i_input_u = 1'b0;
        i_input_b = 1'b0;
        i_rst_n = 1'b1;

        n = 0;
        fails = 0;
        // The output is a multiplexer over the current inputs and the held state.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL mul_unibi_mux: observed latency 0, expected pp_delay %0d",
                     `GEN_PP_DELAY);
            fails = fails + 1;
        end
        // A row carrying any other column count, a row that fills the line
        // buffer, or an x or z data field ends the run with a fatal at that row,
        // so a row can never borrow a column from its neighbour. A blank line is
        // benign only at end of file.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "mul_unibi_mux: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                // The field count of this format string must match DATA_COLS.
                code = $sscanf(line, "%s %s %s %s %s", tok_rst_flag, tok_in_u_s, tok_in_b_s, tok_exp_out, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "mul_unibi_mux: row %0d is blank", n + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "mul_unibi_mux: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    // Input columns: a token wider than one bit is a corrupt vec file,
                    // so it is rejected here rather than truncated into stimulus.
                    if (tok_rst_flag !== "0" && tok_rst_flag !== "1")
                        $fatal(1, "mul_unibi_mux: row %0d input field rst is \"%0s\", expected a single 0 or 1", n + 1, tok_rst_flag);
                    rst_flag = (tok_rst_flag == "1");
                    if (tok_in_u_s !== "0" && tok_in_u_s !== "1")
                        $fatal(1, "mul_unibi_mux: row %0d input field i_input_u is \"%0s\", expected a single 0 or 1", n + 1, tok_in_u_s);
                    in_u_s = (tok_in_u_s == "1");
                    if (tok_in_b_s !== "0" && tok_in_b_s !== "1")
                        $fatal(1, "mul_unibi_mux: row %0d input field i_input_b is \"%0s\", expected a single 0 or 1", n + 1, tok_in_b_s);
                    in_b_s = (tok_in_b_s == "1");
                    // Expected-output columns: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the column as characters instead.
                    if (tok_exp_out !== "0" && tok_exp_out !== "1")
                        $fatal(1, "mul_unibi_mux: row %0d expected field exp_out is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_out);
                    exp_out = (tok_exp_out == "1");
                    i_input_u = in_u_s;
                    i_input_b = in_b_s;
                    if (rst_flag) begin
                        i_rst_n = 1'b0;
                        @(posedge i_clk);   // the async reset clears q here
                        @(negedge i_clk);
                        i_rst_n = 1'b1;
                    end
                    #1;                     // let the combinational output settle
                    n = n + 1;
                    if (o_output !== exp_out) begin
                        $display("FAIL cycle %0d: i_input_u=%b i_input_b=%b got %b exp %b",
                                 n, i_input_u, i_input_b, o_output, exp_out);
                        fails = fails + 1;
                    end
                    @(posedge i_clk);       // advance the toggle flip-flop
                    @(negedge i_clk);
                end
            end
        end
        $fclose(fd);

        // Every generated row must have been compared, so a vec file that lost
        // rows fails instead of passing on the rows it kept.
        if (n != `GEN_VECTORS) begin
            $display("FAIL mul_unibi_mux: compared %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS mul_unibi_mux: %0d/%0d vectors", n, n);
        else
            $display("FAIL mul_unibi_mux: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
