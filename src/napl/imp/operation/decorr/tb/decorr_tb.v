`timescale 1ns/1ps
`default_nettype none
// Generated sizes mirror the Python model configuration.
`include "decorr/vec/decorr_params.vh"
// Python golden rows are <rst> <in_0> <in_1> <out_0> <out_1>.
// Both outputs are read before the posedge that updates the buffers and the
// sequence counter; rst=1 opens a block by reloading the reset state.
// Co-sim: make test OP=decorr


module decorr_tb;
    reg  i_clk;
    reg  i_rst_n;
    reg  i_input_0;
    reg  i_input_1;
    wire o_output_0;
    wire o_output_1;

    decorr #(
        .DEPTH   (`GEN_DEPTH),
        .IDX_W   (`GEN_IDX_W),
        .SEQ_LEN (`GEN_SEQ_LEN),
        .SEQ_W   (`GEN_SEQ_W)
    ) dut (
        .i_clk     (i_clk),
        .i_rst_n   (i_rst_n),
        .i_input_0 (i_input_0),
        .i_input_1 (i_input_1),
        .o_output_0   (o_output_0),
        .o_output_1   (o_output_1)
    );

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
    localparam LINE_BYTES = 64;
    localparam DATA_COLS  = 5;

    integer fd, code, chars, n, fails;
    reg rst, a, b, exp_0, exp_1;
    // The columns are scanned as text so a token outside {0, 1} always reaches
    // the guard below; each reg is wider than a legal token so an over-wide
    // token reads back over-wide instead of saturating.
    reg [8*8-1:0] tok_rst;
    reg [8*8-1:0] tok_a;
    reg [8*8-1:0] tok_b;
    reg [8*8-1:0] tok_exp_0;
    reg [8*8-1:0] tok_exp_1;
    reg [8*8-1:0] extra;
    reg [8*LINE_BYTES-1:0] line;

    initial begin
        i_clk     = 1'b0;
        i_rst_n   = 1'b1;
        i_input_0 = 1'b0;
        i_input_1 = 1'b0;

        fd = $fopen("vec/decorr.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/decorr.vec (run `make vectors` first)");
            $finish;
        end

        // HEADER GUARD: the header is read through the same buffer as a data
        // row, so it takes the same truncation check.
        chars = $fgets(line, fd);
        if (chars == 0)
            $fatal(1, "decorr: vec file has no header line");
        if (chars == LINE_BYTES && line[7:0] !== "\n")
            $fatal(1, "decorr: header line fills the %0d byte line buffer", LINE_BYTES);

        n = 0;
        fails = 0;
        // Each row below is compared in the cycle its inputs are applied, with no
        // clock edge in between, so only a pp_delay of 0 can match the vectors.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL decorr: pp_delay %0d, but the vector rows compare at zero latency",
                     `GEN_PP_DELAY);
            fails = fails + 1;
        end

        // One line holds one row of exactly DATA_COLS binary columns, after a
        // single header line of column names. A line carrying any other column
        // count, a line that fills the line buffer, or an x or z data field ends
        // the run with a fatal at that row, so a row can never borrow a column
        // from its neighbour. A blank line is benign only at end of file.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "decorr: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                // The field count in this format string must match DATA_COLS.
                code = $sscanf(line, "%s %s %s %s %s %s", tok_rst, tok_a, tok_b, tok_exp_0, tok_exp_1, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "decorr: row %0d is blank", n + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "decorr: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    // INPUT COLUMN GUARDS: a stimulus column outside {0, 1} is a
                    // corrupt file, and is named as one rather than truncated
                    // into a bit the DUT would accept as stimulus.
                    if (tok_rst !== "0" && tok_rst !== "1")
                        $fatal(1, "decorr: row %0d input field rst is \"%0s\", expected a single 0 or 1", n + 1, tok_rst);
                    rst = (tok_rst == "1");
                    if (tok_a !== "0" && tok_a !== "1")
                        $fatal(1, "decorr: row %0d input field in_0 is \"%0s\", expected a single 0 or 1", n + 1, tok_a);
                    a = (tok_a == "1");
                    if (tok_b !== "0" && tok_b !== "1")
                        $fatal(1, "decorr: row %0d input field in_1 is \"%0s\", expected a single 0 or 1", n + 1, tok_b);
                    b = (tok_b == "1");
                    // EXPECTED COLUMN GUARDS: a golden column outside {0, 1} is
                    // a corrupt answer, and is named as one rather than
                    // truncated into a bit that can match the DUT.
                    if (tok_exp_0 !== "0" && tok_exp_0 !== "1")
                        $fatal(1, "decorr: row %0d expected field exp_0 is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_0);
                    exp_0 = (tok_exp_0 == "1");
                    if (tok_exp_1 !== "0" && tok_exp_1 !== "1")
                        $fatal(1, "decorr: row %0d expected field exp_1 is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_1);
                    exp_1 = (tok_exp_1 == "1");

                    // block boundary: reload the buffers and the counter before this cycle
                    if (rst == 1) begin
                        i_rst_n = 1'b0;
                        #1;
                        i_rst_n = 1'b1;
                        #1;
                    end

                    i_input_0 = a;
                    i_input_1 = b;
                    #1;

                    n = n + 1;
                    if (o_output_0 !== exp_0 || o_output_1 !== exp_1) begin
                        $display("FAIL n=%0d in=(%b,%b) : got (%b,%b) exp (%b,%b)",
                                 n, a, b, o_output_0, o_output_1, exp_0, exp_1);
                        fails = fails + 1;
                    end

                    // clock edge stores this cycle's inputs and advances the counter
                    i_clk = 1'b1; #1;
                    i_clk = 1'b0; #1;
                end
            end
        end
        $fclose(fd);

        // Every generated row must have been compared, so a vec file that lost
        // rows fails instead of passing on the rows it kept.
        if (n != `GEN_VECTORS) begin
            $display("FAIL decorr: compared %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS decorr: %0d/%0d vectors", n, n);
        else
            $display("FAIL decorr: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
