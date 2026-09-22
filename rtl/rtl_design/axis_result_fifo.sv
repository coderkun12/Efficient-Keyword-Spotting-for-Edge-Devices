// ---------------------------------------------------------------------------
// axis_result_fifo -- elastic buffer between the array and the result stream.
//
// WHY THIS EXISTS:
// mac_array cannot stall. It is a fixed-latency systolic pipeline with no
// back-pressure path, which is exactly what gives it one MAC per PE per cycle
// and avoids the 50% utilisation the reference BF16 design lost to its per-tile
// FSM. But AXI4-Stream consumers are allowed to deassert tready whenever they
// like. Something has to reconcile those, and it cannot be the array.
//
// So results land in this FIFO the cycle they are produced, and drain at
// whatever rate the consumer accepts. If the consumer stalls long enough to
// fill the FIFO, data is genuinely lost -- the array has already moved on. That
// is reported rather than hidden: `overflow` latches high and stays high until
// reset, and tile_top exposes it in the STATUS register. A silent drop here
// would look exactly like an arithmetic bug at the system level, which is the
// worst possible failure mode to debug.
//
// Sizing note for M3: DEPTH 16 covers short consumer hiccups. The real fix is
// for the host interface to guarantee drain rate, since at one result vector
// per cycle no practical FIFO survives a sustained stall.
// ---------------------------------------------------------------------------

`default_nettype none

module axis_result_fifo #(
    parameter int WIDTH = 512,          // M * ACC_W
    parameter int DEPTH = 16            // must be a power of two
) (
    input  wire                  clk,
    input  wire                  rst_n,

    // Producer side: unconditional write, no back-pressure offered.
    input  wire                  wr_en,
    input  wire [WIDTH-1:0]      wr_data,

    // Consumer side: AXI4-Stream
    output wire                  m_axis_tvalid,
    input  wire                  m_axis_tready,
    output wire [WIDTH-1:0]      m_axis_tdata,

    output wire                  empty,
    output wire                  full,
    output reg                   overflow
);

    localparam int PTR_W = $clog2(DEPTH);

    reg [WIDTH-1:0] mem [0:DEPTH-1];
    reg [PTR_W:0]   wr_ptr;             // one extra bit distinguishes full/empty
    reg [PTR_W:0]   rd_ptr;

    assign empty = (wr_ptr == rd_ptr);
    assign full  = (wr_ptr[PTR_W] != rd_ptr[PTR_W]) &&
                   (wr_ptr[PTR_W-1:0] == rd_ptr[PTR_W-1:0]);

    // First-word fall-through: the head is visible without a read cycle, so a
    // result is available to the consumer the cycle after the array emits it.
    assign m_axis_tvalid = !empty;
    assign m_axis_tdata  = mem[rd_ptr[PTR_W-1:0]];

    wire do_write = wr_en && !full;
    wire do_read  = m_axis_tvalid && m_axis_tready;

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr   <= {(PTR_W+1){1'b0}};
            rd_ptr   <= {(PTR_W+1){1'b0}};
            overflow <= 1'b0;
            for (i = 0; i < DEPTH; i = i + 1)
                mem[i] <= {WIDTH{1'b0}};
        end else begin
            if (do_write) begin
                mem[wr_ptr[PTR_W-1:0]] <= wr_data;
                wr_ptr <= wr_ptr + 1'b1;
            end
            if (do_read)
                rd_ptr <= rd_ptr + 1'b1;

            // Sticky: a dropped result must never be silently absorbed.
            if (wr_en && full)
                overflow <= 1'b1;
        end
    end

endmodule

`default_nettype wire
