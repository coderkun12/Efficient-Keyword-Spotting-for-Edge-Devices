// ---------------------------------------------------------------------------
// byte_ram -- one byte-wide simple-dual-port RAM: one write port, one read
//             port, read data registered.
//
// WHY THIS MODULE EXISTS
// layer_top's weight memory would not map to block RAM while it was declared
// as a reg array inside a generate block. Four attempts at coaxing inference
// failed, each costing a ~20 minute synthesis run:
//
//   1. split the M*8-bit array into M byte-wide lanes      no change
//   2. replace the computed write address with a bit-slice no change
//   3. add (* ramstyle = "M9K" *)                          silently ignored
//   4. round the depth 1152 -> 2048                        no change; +114,688
//                                                          registers
//
// A probe (fpga/memcheck/wmem_only/wmem_probe.sv) synthesising four variants
// side by side showed the one shape that DID infer with a non-power-of-two
// depth was an array at MODULE scope -- not inside a generate. So the array
// moves here, and layer_top instantiates this module M times. The generate
// then contains module instances, which is not the same thing as a generate
// containing array declarations.
//
// AND AN ESCAPE HATCH
// If module scope still is not enough, `define USE_ALTSYNCRAM stops asking
// the tool to recognise a pattern and instantiates the primitive directly.
// That cannot fail to map, because it is not inference. It is off by default
// because it is Altera-specific and would not survive the move to an ASIC
// flow, where this array becomes a compiled SRAM macro instead.
//
// TIMING CONTRACT, identical either way:
//   write : synchronous, gated by we
//   read  : address presented on cycle n, data valid on cycle n+1
//   read-during-write to the same address returns OLD data
//
// The host writes every weight before compute starts, so same-address
// read-during-write never arises in this design. The contract is stated so
// the two implementations cannot quietly diverge on it.
// ---------------------------------------------------------------------------

`default_nettype none

module byte_ram #(
    parameter int DEPTH = 2048,
    parameter int AW    = 11
) (
    input  wire          clk,
    input  wire          we,
    input  wire [AW-1:0] waddr,
    input  wire [7:0]    wdata,
    input  wire [AW-1:0] raddr,
    output wire [7:0]    q
);

`ifdef USE_ALTSYNCRAM
    // Explicit primitive: guaranteed to map, no inference involved. The
    // parameters mirror what Quartus chose for band_sram's arrays, which are
    // already confirmed in M9K -- DUAL_PORT, registered address on port B,
    // unregistered output, OLD_DATA on a mixed-port collision.
    altsyncram #(
        .operation_mode                     ("DUAL_PORT"),
        .width_a                            (8),
        .widthad_a                          (AW),
        .numwords_a                         (DEPTH),
        .width_b                            (8),
        .widthad_b                          (AW),
        .numwords_b                         (DEPTH),
        .address_reg_b                      ("CLOCK0"),
        .outdata_reg_b                      ("UNREGISTERED"),
        .address_aclr_b                     ("NONE"),
        .outdata_aclr_b                     ("NONE"),
        .indata_aclr_a                      ("NONE"),
        .wrcontrol_aclr_a                   ("NONE"),
        .read_during_write_mode_mixed_ports ("OLD_DATA"),
        .lpm_type                           ("altsyncram"),
        .intended_device_family             ("Cyclone IV GX")
    ) u_ram (
        .clock0         (clk),
        .wren_a         (we),
        .address_a      (waddr),
        .data_a         (wdata),
        .address_b      (raddr),
        .q_b            (q),
        .aclr0          (1'b0),
        .aclr1          (1'b0),
        .addressstall_a (1'b0),
        .addressstall_b (1'b0),
        .byteena_a      (1'b1),
        .byteena_b      (1'b1),
        .clock1         (1'b1),
        .clocken0       (1'b1),
        .clocken1       (1'b1),
        .clocken2       (1'b1),
        .clocken3       (1'b1),
        .data_b         (8'hFF),
        .eccstatus      (),
        .q_a            (),
        .rden_a         (1'b1),
        .rden_b         (1'b1),
        .wren_b         (1'b0)
    );
`else
    // Behavioural: portable, simulates in Icarus, and is what an ASIC flow
    // maps onto a compiled SRAM macro. The array is at MODULE scope, which
    // is the distinction this module exists to create.
    //
    // No reset on the output register: an async clear on a block RAM's
    // output stops the tool packing it into the memory.
    reg [7:0] mem [0:DEPTH-1];
    reg [7:0] q_r;

    always @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        q_r <= mem[raddr];
    end

    assign q = q_r;
`endif

endmodule

`default_nettype wire
