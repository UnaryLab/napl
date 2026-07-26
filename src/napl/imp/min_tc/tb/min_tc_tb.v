// Self-checking testbench for min_tc.
// Reads vec/min_tc.vec (golden vectors from the napl Python model), applies
// each (in_0, in_1) combinationally, and compares o_out to the expected column.
// Prints "PASS" only on a full bit-exact match.
`timescale 1ns / 1ps
module min_tc_tb;

    reg  r_in_0;
    reg  r_in_1;
    wire w_out;

    integer fd;
    integer rc;
    integer n_vec;
    integer n_err;
    reg [127:0] hdr0, hdr1, hdr2;
    integer vin0, vin1, vexp;

    min_tc dut (
        .i_input_0 (r_in_0),
        .i_input_1 (r_in_1),
        .o_out  (w_out)
    );

    initial begin
        n_vec = 0;
        n_err = 0;

        fd = $fopen("vec/min_tc.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/min_tc.vec");
            $finish;
        end

        // consume the header line (3 column names)
        rc = $fscanf(fd, "%s %s %s\n", hdr0, hdr1, hdr2);

        while (!$feof(fd)) begin
            rc = $fscanf(fd, "%d %d %d\n", vin0, vin1, vexp);
            if (rc == 3) begin
                r_in_0 = vin0[0];
                r_in_1 = vin1[0];
                #1; // settle combinational logic
                if (w_out !== vexp[0]) begin
                    n_err = n_err + 1;
                    $display("MISMATCH vec %0d: in_0=%0d in_1=%0d exp=%0d got=%0b",
                             n_vec, vin0, vin1, vexp, w_out);
                end
                n_vec = n_vec + 1;
            end
        end
        $fclose(fd);

        if (n_err == 0)
            $display("PASS: min_tc %0d/%0d vectors match", n_vec, n_vec);
        else
            $display("FAIL: min_tc %0d/%0d vectors mismatched", n_err, n_vec);

        $finish;
    end

endmodule
