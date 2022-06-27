module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16,  GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, VectorAnyINTEL, VectorComputeINTEL, FunctionFloatControlINTEL], [SPV_EXT_shader_atomic_float_add, SPV_INTEL_vector_compute, SPV_INTEL_float_controls2]>, {}>}{
    func @linkage_attr_test() {
        %c1 = arith.constant 1 : index
        gpu.launch_func @linkage_attr_test_module::@linkage_attr_test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args()
        return
    }

    // SPIR-V module
    spv.module @__spv__linkage_attr_test_module Physical64 OpenCL attributes {spv.entry_point_abi = {local_size = dense<0> : vector<3xi32>}} {
        spv.func @linkage_attr_test_kernel()  "DontInline"  attributes {}  {
            %uchar_0 = spv.Constant 0 : i8
            %ushort_1 = spv.Constant 1 : i16
            %uint_0 = spv.Constant 0 : i32
            spv.FunctionCall @outside.func.with.linkage(%uchar_0):(i8) -> ()
            spv.Return
        }
        // Outside SPIR-V func with Import LinkageAttributes
        spv.func @outside.func.with.linkage(%arg0 : i8) -> () "Pure" attributes {LinkageAttributes=["outside.func", "Import"], VectorComputeFunctionINTEL} //{}
        spv.func @inside.func() -> () "Pure" attributes {} {spv.Return}
    }

    // GPU module
    gpu.module @linkage_attr_test_module {
        gpu.func @linkage_attr_test_kernel() kernel attributes {spv.entry_point_abi = {local_size = dense<0> : vector<3xi32>}} {
            gpu.return
        }
    }

}
