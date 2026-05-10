#ifndef TRITONCPUOPT_CONVERSION_PASSES
#define TRITONCPUOPT_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertUnsupportedOps : Pass<"triton-cpu-add-casts-for-unsupported-ops", "mlir::ModuleOp"> {
    let summary = "Convert operations on unsupported types.";
    let description = [{
        This pass converts various operations on data types that are not supported
        by the target natively. Operations are converted to a supported data type
        with casts added for inputs and the result.
    }];

    let options = [
        Option<"promoteBf16ToFp32", "promote-bf16-to-fp32",
               "bool", /*default*/"false",
               "Convert BF16 operations to FP32.">,
        Option<"convertMixedPrecisionMatmul", "convert-mixed-precision-matmul",
               "bool", /*default*/"false",
               "Convert inputs of a mixed-precision matmul to a destination type.">,
        Option<"promoteLibMathToFp32", "promote-lib-math-to-fp32",
               "bool", /*default*/"true",
               "Promote FP8, FP16, BF16 math operations mapped to libm function to FP32.">,
    ];

    let constructor = "mlir::triton::cpu::createConvertUnsupportedOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def DecomposeFpConversions : Pass<"triton-cpu-decompose-fp-conversions", "mlir::ModuleOp"> {
    let summary = "Decompose fp conversion ops.";
    let description = [{
        This pass is used for targets lacking native instructions to convert FP
        vectors. By default, LLVM would decompose them using scalar FP conversion
        intrinsics. This pass transforms such conversions into vector code
        instead.
    }];

    let options = [
        Option<"decomposeBf16Conversions", "decompose-bf16-conversions",
               "bool", /*default*/"false",
               "Lower BF16 conversions to arith operations.">,
        Option<"decomposeFp8Conversions", "decompose-fp8-conversions",
               "bool", /*default*/"false",
               "Lower FP8 conversions to arith operations.">,
    ];

    let constructor = "mlir::triton::cpu::createDecomposeFpConversions()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def OptimizeMasks : Pass<"triton-cpu-optimize-masks", "mlir::ModuleOp"> {
    let summary = "Optimize masked memory accesses.";
    let description = [{
        This pass tries to detect masked memory accesses with mask values that
        can be proven to be all-ones or all-zeros.
    }];

    let options = [
    ];

    let constructor = "mlir::triton::cpu::createOptimizeMasks()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertDotProduct : Pass<"triton-cpu-convert-dot-product", "mlir::ModuleOp"> {
    let summary = "Convert dot product op.";
    let description = [{
        This pass is used for indentifying dot product pattern
        (for example, elementwise mul followed by a sum) and
        converting it to dot product intrinsics like bfdot.
    }];

    let options = [
        Option<"useHorizontalSum", "use-horizontal-sum",
               "bool", /*default*/"true",
               "Use Horizontal Sum kernel for the dot product (gemv). Otherwise use a kernel with packing.">,
    ];

    let constructor = "mlir::triton::cpu::createConvertDotProduct()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertDotToAMX : Pass<"triton-cpu-convert-dot-to-amx", "mlir::ModuleOp"> {
    let summary = "Convert dot product op to AMX dialect.";
    let description = [{
        This pass is used to lower matmul operations to amx dialect.
    }];

    let options = [
        Option<"convertInt8", "convert-i8",
               "bool", /*default*/"false",
               "Use AMX extensions for int8 type.">,
        Option<"convertFp16", "convert-fp16",
               "bool", /*default*/"false",
               "Use AMX extensions for ifp16 type.">,
        Option<"convertBf16", "convert-bf16",
               "bool", /*default*/"false",
               "Use AMX extensions for bf16 type.">,
    ];

    let constructor = "mlir::triton::cpu::createConvertDotToAMX()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::amx::AMXDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertDotToFMA : Pass<"triton-cpu-convert-dot-to-fma", "mlir::ModuleOp"> {
    let summary = "Decompose dot product op to a series of FMA operations.";
    let description = [{ }];

    let constructor = "mlir::triton::cpu::createConvertDotToFMA()";
    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertDotOpToUkernelOps : Pass<"triton-cpu-convert-dot-to-ukernels", "mlir::ModuleOp"> {
    let summary = "Convert dot product op to ukernel ops.";
    let description = [{
        This pass is used to lower DotOp operations to ukernel ops.
    }];

    let options = [
        Option<"ukernels", "ukernels",
               "mlir::triton::cpu::Ukernels", /*default*/"mlir::triton::cpu::Ukernels::OneDNN",
               "Ukernels provider to be used for replacement of dot op (OneDNN/XSMM/etc).",
               [{::llvm::cl::values(
               clEnumValN(mlir::triton::cpu::Ukernels::OneDNN, "oneDNN",
                "Use OneDNN as a ukernels provider"),
               clEnumValN(mlir::triton::cpu::Ukernels::XSMM, "XSMM",
                "Use XSMM as a ukernels provider")
              )}]>,
    ];

    let constructor = "mlir::triton::cpu::createConvertDotOpToUkernelOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertDotGeneric : Pass<"triton-cpu-convert-dot-generic", "mlir::ModuleOp"> {
    let summary = "Generic convertion of dot product op.";
    let description = [{
        This pass is used to lower matmul operations to generic vector code.
    }];

    let constructor = "mlir::triton::cpu::createConvertDotGeneric()";

    let dependentDialects = ["mlir::vector::VectorDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def Canonicalize : Pass<"triton-cpu-canonicalize", "mlir::ModuleOp"> {
    let summary = "Canonicalization pass.";
    let description = [{
        This pass applies various foldings to simplify analysis and transformations
        in optimization passes.
    }];

    let constructor = "mlir::triton::cpu::createCanonicalize()";

    let dependentDialects = ["mlir::vector::VectorDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def SPMTensorPlacement : Pass<"triton-cpu-spm-tensor-placement", "mlir::ModuleOp"> {
    let summary = "Analyze tensor placement tiers for SPM-aware lowering.";
    let description = [{
        This pass is the analysis hook for the three-tier data placement pipeline.
        It runs before ConvertMemoryToSPM and will annotate eligible kernel
        arguments with the selected SPM placement tier.

        M3 intentionally leaves the pass as a no-op skeleton; later milestones add
        scalar-reuse analysis, tier attributes, and sidecar export.
    }];

    let options = [
        Option<"enableReductions", "enable-reductions",
               "bool", /*default*/"true",
               "Emit tier sidecar entries for reduction/streaming-only tensors.">,
    ];

    let constructor = "mlir::triton::cpu::createSPMTensorPlacement()";

    let dependentDialects = ["mlir::func::FuncDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def SplitLargeContract : Pass<"triton-cpu-split-large-contract", "mlir::ModuleOp"> {
    let summary = "Split large vector.contract ops into register-friendly micro-tiles.";
    let description = [{
        Splits vector.contract operations whose accumulator M dimension exceeds
        `microM` into ceil(M / microM) smaller contracts.  This reduces vector
        register pressure on targets with limited physical registers (e.g.
        RISC-V RVV VLEN=256 with 32 vregs).

        Must run AFTER ConvertMemoryToSPM (so DMA granularity is unaffected)
        and BEFORE LLVM lowering.
    }];

    let options = [
        Option<"microM", "micro-m",
               "int64_t", /*default*/"8",
               "Maximum number of accumulator rows per micro-tile.">,
    ];

    let constructor = "mlir::triton::cpu::createSplitLargeContract()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::vector::VectorDialect"];
}

def ConvertMemoryToSPM : Pass<"triton-cpu-convert-memory-to-spm", "mlir::ModuleOp"> {
    let summary = "Convert tiled DRAM loads to DMA-based SPM transfers.";
    let description = [{
        This pass analyzes memory access patterns in tiled loops and converts
        eligible DRAM loads into DMA transfers to scratchpad memory (SPM).

        For GEMM-like patterns (K-loop with two tiled loads feeding a dot product),
        the pass inserts double-buffered DMA transfers: while the current tiles are
        being computed, the next tiles are prefetched asynchronously.

        For reduction patterns (single loop accumulating over a dimension), the pass
        inserts double-buffered DMA prefetch for the next chunk.

        Loads that don't match SPM-eligible patterns (scalar loads, irregular access,
        low-reuse elementwise ops) are left unchanged to use the cache hierarchy.
    }];

    let options = [
        Option<"spmBase", "spm-base",
               "int64_t", /*default*/"0x40000000",
               "Base address of the scratchpad memory region.">,
        Option<"spmSize", "spm-size",
               "int64_t", /*default*/"262144",
               "Size of the scratchpad memory in bytes.">,
        Option<"microM", "micro-m",
               "int64_t", /*default*/"8",
               "Micro-tile M dimension for fused SPM GEMM lowering.">,
        Option<"windowK", "window-k",
               "int64_t", /*default*/"8",
               "Maximum K-iteration window for resident B tiles.">,
        Option<"enableReductions", "enable-reductions",
               "bool", /*default*/"true",
               "Enable reduction/streaming loop lowering through SPM.">,
        Option<"enableRowResidentReductions", "enable-row-resident-reductions",
               "bool", /*default*/"false",
               "Enable the D2 opt-in row-resident reduction prototype.">,
        Option<"rowResidentMaxBytes", "row-resident-max-bytes",
               "int64_t", /*default*/"4096",
               "Maximum row bytes accepted by the row-resident reduction prototype.">,
        Option<"rowResidentProducerPass", "row-resident-producer-pass",
               "std::string", /*default*/"\"fill_on_first_pass\"",
               "Row-resident reduction producer schedule: fill_on_first_pass, producer_store, or dma_prefetch.">,
        Option<"enablePromotionProfitability", "enable-promotion-profitability",
               "bool", /*default*/"false",
               "Enable the D3 conservative promotion profitability model.">,
        Option<"promotionReport", "promotion-report",
               "bool", /*default*/"false",
               "Write SPM promotion records and rejection reasons to the AOT auxiliary directory.">,
    ];

    let constructor = "mlir::triton::cpu::createConvertMemoryToSPM()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

#endif
