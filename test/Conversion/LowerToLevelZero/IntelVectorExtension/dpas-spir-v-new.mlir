module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, VectorAnyINTEL, VectorComputeINTEL, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, VectorAnyINTEL, FunctionFloatControlINTEL], [SPV_EXT_shader_atomic_float_add, SPV_INTEL_vector_compute, SPV_INTEL_float_controls2]>, {}>}  {

  // function to setup the launch and launch the kernel
  // args: size_t systolic_depth, size_t repeat_cnt, size_t N
  // @ISSUE:SPIR-V type does not support bf16, hence passing vector 1, and vector 2 as f16, will load bf16 from this vector using the intel vc-intrinsics
  // func @dpas_test(%arg_sys_dpth: index, %arg_rpt_cnt: index, %arg_N: index, %arg_v0 : memref<?xf32>, %arg_v1 : memref<?xbf16>, %arg_v2 : memref<?xbf16>)
  func @dpas_gpu(%arg_sys_dpth: index, %arg_rpt_cnt: index, %arg_N: index, %arg_v0 : memref<?xf32>, %arg_v1 : memref<?xf16>, %arg_v2 : memref<?xf16>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    
    // Allocate vectors to be passed to the kernel
    // Allocate v0 and copy the data from arg_v0
    // %v0_size = arith.muli %arg_rpt_cnt, %arg_N : index 
    // %memref_v0 = gpu.alloc  (%v0_size) {gpu.alloc_shared} : memref<?xf32>
    // Don't need the reinterpret cast since the allocated array is already 1D, just copy is enough
    // memref.copy %arg_v0, %memref_v0 : memref<?xf32> to memref<?xf32>
    

    // %tmp_sys_dpth = arith.muli %arg_sys_dpth, %c2 : index
    // Allocate v1 and copy the data from arg_v1
    // %v1_size = arith.muli %tmp_sys_dpth, %arg_N : index

    // @ISSUE:SPIR-V type does not support bf16, hence passing vector 1, and vector 2 as f16, will load bf16 from this vector using the intel vc-intrinsics
    // %memref_v1 = gpu.alloc  (%v1_size) {gpu.alloc_shared} : memref<?xbf16>
    // memref.copy %arg_v1, %memref_v1 : memref<?xbf16> to memref<?xbf16> 
    
    // %memref_v1 = gpu.alloc  (%v1_size) {gpu.alloc_shared} : memref<?xf16>
    // memref.copy %arg_v1, %memref_v1 : memref<?xbf16> to memref<?xf16> 
    

    // Allocate v2 and copy the data from arg_v2
    // %v2_size = arith.muli %tmp_sys_dpth, %arg_rpt_cnt : index
    // @ISSUE:SPIR-V type does not support bf16, hence passing vector 1, and vector 2 as f16, will load bf16 from this vector using the intel vc-intrinsics
    // %memref_v2 = gpu.alloc  (%v2_size) {gpu.alloc_shared} : memref<?xbf16>
    // memref.copy %arg_v2, %memref_v2 : memref<?xbf16> to memref<?xbf16>

    // %memref_v2 = gpu.alloc  (%v2_size) {gpu.alloc_shared} : memref<?xf16>
    // memref.copy %arg_v2, %memref_v2 : memref<?xbf16> to memref<?xf16>

    // Launch the kernel
    // blocks in grid_size(x,y,z) , threads in block_size(x,y,z)
    // If any dimension of grid_size or block_size does not have any value, it has to be set to 1
    // So in the code I saw, in most cases they use, 2-D grid_size and 2-D block_size
    // In the test case it uses, blocks in grid_size(1,1,1) , threads in block_size(1,1,1) -> 
    // "blocks in (1,1,1) threads in (1,1,1) 
    
    
    // @Question: Understanding the Intel GPU structure
    // blocks and threads are for indexing 
    gpu.launch_func @dpas_module::@dpas_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%arg_v0 : memref<?xf32>, %arg_v1 : memref<?xf16>, %arg_v2 : memref<?xf16>) 
    // args: size_t systolic_depth, size_t repeat_cnt, size_t N
    return
  }

  // SPIR-V DPAS module, it holds the DPAS kernel
  // Add these following kernel properties and set the execution mode
    // [Added] OpMemoryModel Physical64 OpenCL
    // OpEntryPoint Kernel %13 "aaa0_closure_0"
    // OpExecutionMode %13 ContractionOff
    // OpExecutionMode %13 SharedLocalMemorySizeINTEL 0
    // OpExecutionMode %13 NamedBarrierCountINTEL 0
    // OpSource Unknown 0
  spv.module @__spv__dpas_module Physical64 OpenCL requires #spv.vce<v1.0, [Int8, Int16, Int64, Float16, Kernel, Addresses, Linkage, Vector16, VectorAnyINTEL, Float16Buffer, VectorComputeINTEL, FunctionFloatControlINTEL], [SPV_INTEL_float_controls2, SPV_INTEL_vector_compute]> {   
    // DPAS kernel 
    //@Question: attributes {spv.entry_point_abi = {local_size = dense<0> : vector<3xi32>}}, how do I interpret this  

    // @Maybe: local_size, ABI for args
    // WorkgroupSize: 8EU one shared mem, each EU has a systolic array
    // @ISSUE:SPIR-V type does not support bf16, hence passing vector 1, and vector 2 as f16, will load bf16 from this vector using the intel vc-intrinsics
    // spv.func @dpas_kernel(%arg0: !spv.ptr<f32, CrossWorkgroup>, %arg1: !spv.ptr<bf16, CrossWorkgroup>, %arg2: !spv.ptr<bf16, CrossWorkgroup>)  "DontInline"  attributes {spv.entry_point_abi = {local_size = dense<0> : vector<3xi32>}, workgroup_attributions = 0 : i64}  
    spv.func @dpas_kernel(%arg0: !spv.ptr<f32, CrossWorkgroup>, %arg1: !spv.ptr<f16, CrossWorkgroup>, %arg2: !spv.ptr<f16, CrossWorkgroup>)  "DontInline"  attributes {spv.entry_point_abi = {local_size = dense<0> : vector<3xi32>}, workgroup_attributions = 0 : i64}  {
        // %uchar_0 = spv.Constant 0 : i8
        // %ushort_1 = spv.Constant 1 : i16
        // %uint_0 = spv.Constant 0 : i32
        // %uchar_3 = spv.Constant 3 : i8
        // %uchar_8 = spv.Constant 8 : i8
        // %uchar_2 = spv.Constant 2 : i8
        // %uchar_4 = spv.Constant 4 : i8
        // %uchar_7 = spv.Constant 7 : i8
        // %uint_9 =  spv.Constant 9 :  i32
        // %uint_8 =  spv.Constant 8 :  i32
        // %uint_4 =  spv.Constant 4 :  i32
        %uchar_0 = spv.Constant 0 : i8
        %ushort_1 = spv.Constant 1 : i16
        %uint_0 = spv.Constant 0 : i32
        %uchar_3 = spv.Constant 3 : i8
        %uchar_8 = spv.Constant 8 : i8
        %uchar_2 = spv.Constant 2 : i8
        %uchar_4 = spv.Constant 4 : i8
        %uchar_7 = spv.Constant 7 : i8
        %uint_9 =  spv.Constant 9 :  i32
        %uint_8 =  spv.Constant 8 :  i32
        %uint_4 =  spv.Constant 4 :  i32        
        %true = spv.Constant true

        // test vector size
        // %xx = spv.FunctionCall @test() : () -> vector<2049xf32>
        // /////////////////
        // Cast the uchar pointers (i8 ptr) to ulongs (i64)
        %arg_0 = spv.FunctionCall @llvm_genx_address_convert_i64_p1f32(%arg0) : (!spv.ptr<f32, CrossWorkgroup>) -> i64
        %arg_1 = spv.FunctionCall @llvm_genx_address_convert_i64_p1f16(%arg1) : (!spv.ptr<f16, CrossWorkgroup>) -> i64
        %arg_2 = spv.FunctionCall @llvm_genx_address_convert_i64_p1f16(%arg2) : (!spv.ptr<f16, CrossWorkgroup>) -> i64
        
        // Load vector 0 using load stateless

        // @ISSUE: spv return type restriction: 'spv.FunctionCall' op result #0 must be void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type, but got 'vector<64xf32>'
        
        // %v0 =  spv.FunctionCall @llvm_genx_lsc_load_stateless_v64f32_i1_i64(%true, %uchar_0, %uchar_0, %uchar_0, %ushort_1, %uint_0, %uchar_3, %uchar_8, %uchar_2, %uchar_0, %arg_0, %uint_0) : (i1, i8, i8, i8, i16, i32, i8, i8, i8, i8, i64, i32) -> vector<64xf32> 
        // Signless version
        %v0 =  spv.FunctionCall @llvm_genx_lsc_load_stateless_v64f32_i1_i64(%true, %uchar_0, %uchar_0, %uchar_0, %ushort_1, %uint_0, %uchar_3, %uchar_8, %uchar_2, %uchar_0, %arg_0, %uint_0) : (i1, i8, i8, i8, i16, i32, i8, i8, i8, i8, i64, i32) -> vector<64xf32> 


        // Load vector 1 using load stateless
        // %v1 =  spv.FunctionCall @llvm_genx_lsc_load_stateless_v64i64_i1_i64(%true, %uchar_0, %uchar_0, %uchar_0, %ushort_1, %uint_0, %uchar_4, %uchar_8, %uchar_2, %uchar_0, %arg_1, %uint_0) : (i1, i8, i8, i8, i16, i32, i8, i8, i8, i8, i64, i32) -> vector<64 x i64>
        %v1 =  spv.FunctionCall @llvm_genx_lsc_load_stateless_v64i64_i1_i64(%true, %uchar_0, %uchar_0, %uchar_0, %ushort_1, %uint_0, %uchar_4, %uchar_8, %uchar_2, %uchar_0, %arg_1, %uint_0) : (i1, i8, i8, i8, i16, i32, i8, i8, i8, i8, i64, i32) -> vector<64 x i64>
        // Cast the vector v1 as i32 from i64
        %v1_uint_cast = spv.Bitcast %v1 : vector<64 x i64> to vector<128 x i32>

        // Load vector 2 using load stateless
        %v2 =  spv.FunctionCall @llvm_genx_lsc_load_stateless_v32i32_i1_i64(%true, %uchar_0, %uchar_0, %uchar_0, %ushort_1, %uint_0, %uchar_3, %uchar_7, %uchar_2, %uchar_0, %arg_2, %uint_0) : (i1, i8, i8, i8, i16, i32, i8, i8, i8, i8, i64, i32) -> vector<32 x i32>

        // Call dpas2
        %dpas_result =  spv.FunctionCall @llvm_genx_dpas2_v64f32_v64f32_v128i32_v32i32(%v0, %v1_uint_cast, %v2, %uint_9, %uint_9, %uint_8, %uint_4, %uint_0, %uint_0): (vector<64 x f32>, vector<128 x i32>, vector<32 x i32>, i32, i32, i32, i32, i32, i32) -> vector<64 x f32>
        
        // Store the result        
        // %ret_store = 
        spv.FunctionCall @llvm_genx_lsc_store_stateless_i1_i64_v64f32(%true, %uchar_4, %uchar_0, %uchar_0, %ushort_1, %uint_0, %uchar_3, %uchar_8, %uchar_2, %uchar_0, %v0, %dpas_result, %uint_0) : (i1, i8, i8, i8, i16, i32, i8, i8, i8, i8, vector<64 x f32>, vector<64 x f32>, i32) -> () // -> mlir::NoneType

        spv.Return
        // "spv.ReturnValue"(%arg) : (i32) -> (i32)
    }

    // Setting ABI attributes and Execution mode [Done in the "SPIRVLowerABIAttributes" pass]
    // spv.EntryPoint "Kernel" @dpas_kernel
    // @Question: "LocalSize" is not set on the Yijie's SPIR-V code, but set on the 'spv' code generated by Niahsnt's test case, Should we keep the this execution mode or not? 
    // spv.ExecutionMode @dpas_kernel "LocalSize", 0, 0, 0
    spv.ExecutionMode @dpas_kernel "ContractionOff"
    spv.ExecutionMode @dpas_kernel "SharedLocalMemorySizeINTEL", 0
    
    // @TODO: Current L0-runner cannot recognize this, need to work on it and add this
    // this execution mode is Intel-specific and neither upstream or mlir-extension have this currently
    // spv.ExecutionMode @dpas_kernel "NamedBarrierCountINTEL", 0


    // Don't need the function controls (e.g., pure, const)- since these function controls and function parameter attributes are taken from the funciton definition (SPIR-V Spec sec. 2.13)
    // Utility function declarations (Intel intrinsics)
    spv.func @llvm_genx_address_convert_i64_p1f32(%arg: !spv.ptr<f32, CrossWorkgroup>) -> i64 "Pure" attributes {LinkageAttributes=["llvm.genx.address.convert.i64.p1i8", "Import"], VectorComputeFunctionINTEL} //{}

    // spv.func @llvm_genx_address_convert_i64_p1f32(%arg: !spv.ptr<f32, CrossWorkgroup>) "Pure" //attributes {LinkageAttributes={"llvm.genx.address.convert.i64.p1i8", "Import"}} //{}

    // %llvm_genx_address_convert_i64_p1f32 = sp.Name "llvm.genx.address.convert.i64.p1i8"


    spv.func @llvm_genx_address_convert_i64_p1f16(%arg: !spv.ptr<f16, CrossWorkgroup>) -> i64 "Pure" attributes {LinkageAttributes=["llvm.genx.address.convert.i64.p1f16", "Import"], VectorComputeFunctionINTEL}
    // spv.func @llvm_genx_address_convert_i64_p1i8(%arg: !spv.ptr<i8, CrossWorkgroup>) -> i64 "Pure" //attributes {LinkageAttributes: "llvm.genx.address.convert.i64.p1i8", "Import"}
    
    spv.func @llvm_genx_lsc_load_stateless_v64f32_i1_i64(%arg0 : i1, %arg1 : i8, %arg2 : i8, %arg3 : i8, %arg4 : i16, %arg5 : i32, %arg6 : i8, %arg7 : i8, %arg8 : i8, %arg9 : i8, %arg10 : i64, %arg11 : i32) -> vector<64 x f32> "Const" attributes{LinkageAttributes=["llvm.genx.lsc.load.stateless.v64f32.i1.i64", "Import"], VectorComputeFunctionINTEL} //{} //attributes {}
    
    spv.func @llvm_genx_lsc_load_stateless_v64i64_i1_i64(%arg0 : i1, %arg1 : i8, %arg2 : i8, %arg3 : i8, %arg4 : i16, %arg5 : i32, %arg6 : i8, %arg7 : i8, %arg8 : i8, %arg9 : i8, %arg10 : i64, %arg11 : i32) -> vector<64 x i64> "Const" attributes{LinkageAttributes=["llvm.genx.lsc.load.stateless.v64f64.i1.i64", "Import"], VectorComputeFunctionINTEL}//attributes {}
    
    spv.func @llvm_genx_lsc_load_stateless_v32i32_i1_i64(%arg0 : i1, %arg1 : i8, %arg2 : i8, %arg3 : i8, %arg4 : i16, %arg5 : i32, %arg6 : i8, %arg7 : i8, %arg8 : i8, %arg9 : i8, %arg10 : i64, %arg11 : i32) -> vector<32 x i32> "Const" attributes{LinkageAttributes=["llvm.genx.lsc.load.stateless.v32i32.i1.i64", "Import"], VectorComputeFunctionINTEL} //attributes {}

    spv.func @llvm_genx_dpas2_v64f32_v64f32_v128i32_v32i32(%arg0 : vector<64 x f32>, %arg1 : vector<128 x i32>, %arg2 : vector<32 x i32>, %arg3 : i32, %arg4 : i32, %arg5 : i32, %arg6 : i32, %arg7 : i32, %arg8 : i32) -> vector<64 x f32> "Pure" attributes{LinkageAttributes=["llvm.genx.dpas2.v64f32.v64f32.v128i32.v32i32", "Import"], VectorComputeFunctionINTEL} //attributes {}

    spv.func @llvm_genx_lsc_store_stateless_i1_i64_v64f32(%arg0 : i1, %arg1 : i8, %arg2 : i8, %arg3 : i8, %arg4 : i16, %arg5 : i32, %arg6 : i8, %arg7 : i8, %arg8 : i8, %arg9 : i8, %arg10 : vector<64 x f32>, %arg11 : vector<64 x f32>, %arg12 : i32)  "None" attributes{LinkageAttributes=["llvm.genx.lsc.store.stateless.i1.i64.v64f32", "Import"], VectorComputeFunctionINTEL} //attributes {} -> mlir::NoneType
  }

  // GPU module, almost same as the SPIR-V module but without 'spv' specific properties
  // @Question: Why do the kernel definition inside GPU module stays intact in the lowered passes
  // Does the kernel code inside the GPU module helps in any way?
  // How would it hadle multiple kernels?
  gpu.module @dpas_module {
    gpu.func @dpas_kernel(%arg0: memref<?xf32>, %arg1: memref<?xf16>, %arg2: memref<?xf16>) kernel attributes {spv.entry_point_abi = {local_size = dense<0> : vector<3xi32>}} {
    //   %uchar_0 = "arith.Constant"() { value = 0 : i8 } : () -> i8
    //   %ushort_1 = "arith.Constant"() { value = 0 : i16 } : () -> i16
    //   %uint_0 = "arith.Constant"() { value = 0 : i32 } : () -> i32
    //   %uchar_3 = "arith.Constant"() { value = 3 : i8 } : () -> i8
    //   %uchar_8 = "arith.Constant"() { value = 8 : i8 } : () -> i8
    //   %uchar_2 = "arith.Constant"() { value = 2 : i8 } : () -> i8
    //   %uchar_4 = "arith.Constant"() { value = 4 : i8 } : () -> i8
    //   %uchar_7 = "arith.Constant"() { value = 7 : i8 } : () -> i8
    //   %uint_9 = "arith.Constant"() { value = 9 : i32 } : () -> i32
    //   %uint_8 = "arith.Constant"() { value = 8 : i32 } : () -> i32
    //   %uint_4 = "arith.Constant"() { value = 4 : i32 } : () -> i32        
    //   %true = "arith.Constant"() { value = "true" : i1 } : () -> i1
    //   // Cast the uchar pointers (i8 ptr) to ulongs (i64)
    //   %arg_0 = call @llvm_genx_address_convert_i64_p1f32(%arg0) : (memref<?xf32>) -> (i64)
    //   %arg_1 = call @llvm_genx_address_convert_i64_p1f16(%arg1) : (memref<?xf16>) -> (i64)
    //   %arg_2 = call @llvm_genx_address_convert_i64_p1f16(%arg2) : (memref<?xf16>) -> (i64)
      
    //   // Load vector 0 using load stateless
    //   %v0 =  call @llvm_genx_lsc_load_stateless_v64f32_i1_i64(%true, %uchar_0, %uchar_0, %uchar_0, %ushort_1, %uint_0, %uchar_3, %uchar_8, %uchar_2, %uchar_0, %arg_0, %uint_0) : (i1, i8, i8, i8, i16, i32, i8, i8, i8, i8, i64, i32) -> memref<64 x f32>

    //   // Load vector 1 using load stateless
    //   %v1 =  call @llvm_genx_lsc_load_stateless_v64i64_i1_i64(%true, %uchar_0, %uchar_0, %uchar_0, %ushort_1, %uint_0, %uchar_4, %uchar_8, %uchar_2, %uchar_0, %arg_1, %uint_0) : (i1, i8, i8, i8, i16, i32, i8, i8, i8, i8, i64, i32) -> memref<64xi64>
      
    //   // Cast the vector v1 as i32 from i64
    //   %v1_uint_cast = memref.cast %v1 : memref<64xi64> to memref<128xi32>

    //   // Load vector 2 using load stateless
    //   %v2 =  call @llvm_genx_lsc_load_stateless_v32i32_i1_i64(%true, %uchar_0, %uchar_0, %uchar_0, %ushort_1, %uint_0, %uchar_3, %uchar_7, %uchar_2, %uchar_0, %arg_2, %uint_0) : (i1, i8, i8, i8, i16, i32, i8, i8, i8, i8, i64, i32) -> memref<32xi32>

    //   // Call dpas2
    //   %dpas_result =  call @llvm_genx_dpas2_v64f32_v64f32_v128i32_v32i32(%v0, %v1_uint_cast, %v2, %uint_9, %uint_9, %uint_8, %uint_4, %uint_0, %uint_0): (memref<64xf32>, memref<128xi32>, memref<32xi32>, i32, i32, i32, i32, i32, i32) -> memref<64xf32>
      
    //   // Store the result        
    //   // %ret_store = 
    //   call @llvm_genx_lsc_store_stateless_i1_i64_v64f32(%true, %uchar_4, %uchar_0, %uchar_0, %ushort_1, %uint_0, %uchar_3, %uchar_8, %uchar_2, %uchar_0, %v0, %dpas_result, %uint_0) : (i1, i8, i8, i8, i16, i32, i8, i8, i8, i8, memref<64xf32>, memref<64xf32>, i32) -> () //mlir::NoneType

      gpu.return
    }

    // // Utility function declarations (Intel intrinsics)
    // // We don't need to launch it from the host, hence, they don't have to be declared kernel
    // // Instead these are only called from the device so normal function proerty is enough
    
    // gpu.func @llvm_genx_address_convert_i64_p1f32(%arg: memref<?xf32>) -> i64 attributes{} {}
    // gpu.func @llvm_genx_address_convert_i64_p1f16(%arg: memref<?xf16>) -> i64 attributes{} {}
    
    // gpu.func @llvm_genx_lsc_load_stateless_v64f32_i1_i64(%arg0 : i1, %arg1 : i8, %arg2 : i8, %arg3 : i8, %arg4 : i16, %arg5 : i32, %arg6 : i8, %arg7 : i8, %arg8 : i8, %arg9 : i8, %arg10 : i64, %arg11 : i32) -> (memref<64xf32>) {}
    
    // gpu.func @llvm_genx_lsc_load_stateless_v64i64_i1_i64(%arg0 : i1, %arg1 : i8, %arg2 : i8, %arg3 : i8, %arg4 : i16, %arg5 : i32, %arg6 : i8, %arg7 : i8, %arg8 : i8, %arg9 : i8, %arg10 : i64, %arg11 : i32) -> memref<64xi64> {}
    
    // gpu.func @llvm_genx_lsc_load_stateless_v32i32_i1_i64(%arg0 : i1, %arg1 : i8, %arg2 : i8, %arg3 : i8, %arg4 : i16, %arg5 : i32, %arg6 : i8, %arg7 : i8, %arg8 : i8, %arg9 : i8, %arg10 : i64, %arg11 : i32) -> memref<32xi32> {}

    // gpu.func @llvm_genx_dpas2_v64f32_v64f32_v128i32_v32i32(%arg0 : vector<64 x f32>, %arg1 : memref<128xi32>, %arg2 : memref<32xi32>, %arg3 : i32, %arg4 : i32, %arg5 : i32, %arg6 : i32, %arg7 : i32, %arg8 : i32) -> memref<64xf32> {}

    // gpu.func @llvm_genx_lsc_store_stateless_i1_i64_v64f32(%arg0 : i1, %arg1 : i8, %arg2 : i8, %arg3 : i8, %arg4 : i16, %arg5 : i32, %arg6 : i8, %arg7 : i8, %arg8 : i8, %arg9 : i8, %arg10 : memref<64xf32>, %arg11 : memref<64xf32>, %arg12 : i32) {}//-> mlir::NoneType //{}  
  }
  

  
  func @dpas_ref(%arg_sys_dpth: index, %arg_rpt_cnt: index, %arg_N: index, %arg_v0 : memref<?xf32>, %arg_v1 : memref<?xbf16>, %arg_v2 : memref<?xbf16>){

  }

  func @dpas_test(%arg_sys_dpth: index, %arg_rpt_cnt: index, %arg_N: index){
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.100000e+00 : bf16
    %cst_2 = arith.constant 2.200000e+00 : bf16

    // Allocate vectors to be passed to function

    // Setting up Vector v0
    %v0_size = arith.muli %arg_rpt_cnt, %arg_N : index 
    %memref_v0_cpu = memref.alloc (%v0_size) : memref<?xf32>
    %memref_v0_gpu = gpu.alloc (%v0_size) {gpu.alloc_shared} : memref<?xf32>   
    // Initialize v0 to 0
    call @fillResource1DFloat(%memref_v0_cpu, %cst_0) : (memref<?xf32>, f32) -> ()
    call @fillResource1DFloat(%memref_v0_gpu, %cst_0) : (memref<?xf32>, f32) -> ()

    // Setting up the Vector v1
    %tmp_sys_dpth = arith.muli %arg_sys_dpth, %c2 : index
    %v1_size = arith.muli %tmp_sys_dpth, %arg_N : index

    %memref_v1_cpu = memref.alloc (%v1_size) : memref<?xbf16>
    // @ISSUE:SPIR-V type does not support bf16, hence passing vector 1, and vector 2 as f16, will load bf16 from this vector using the intel vc-intrinsic    
    %memref_v1_gpu = gpu.alloc (%v1_size) {gpu.alloc_shared} : memref<?xf16>
    // Initialize it to 1.1
    call @fillResource1Dbf16(%memref_v1_cpu, %cst_1) : (memref<?xbf16>, bf16) -> ()
    call @fillResource1Df16(%memref_v1_gpu, %cst_1) : (memref<?xf16>, f16) -> ()

    // Setting up the Vector v2
    %v2_size = arith.muli %tmp_sys_dpth, %arg_rpt_cnt : index
    
    %memref_v2_cpu = memref.alloc  (%v2_size) : memref<?xbf16>
    // @ISSUE:SPIR-V type does not support bf16, hence passing vector 1, and vector 2 as f16, will load bf16 from this vector using the intel vc-intrinsic    
    %memref_v2_gpu = gpu.alloc (%v2_size) {gpu.alloc_shared} : memref<?xf16>
    // Initialize it to 2.2
    call @fillResource1Dbf16(%memref_v2_cpu, %cst_2) : (memref<?xbf16>, bf16) -> ()
    call @fillResource1Df16(%memref_v2_gpu, %cst_2) : (memref<?xf16>, f16) -> ()

    // Calling the reference function/CPU version
    call @dpas_ref(%arg_sys_dpth, %arg_rpt_cnt,  %arg_N, %memref_v0_cpu, %memref_v1_cpu, %memref_v2_cpu) : (index, index, index, memref<?xf32>, memref<?xbf16>, memref<?xbf16>) -> ()

    // Calling the GPU version
    call @dpas_gpu(%arg_sys_dpth, %arg_rpt_cnt,  %arg_N, %memref_v0_gpu, %memref_v1_gpu, %memref_v2_gpu) : (index, index, index, memref<?xf32>, memref<?xbf16>, memref<?xbf16>) -> ()

  }

  // main function
  func @main() {
    %cst_sys_dpth = arith.constant 8 : index
    %cst_rpt_cnt = arith.constant 4 : index
    %cst_N = arith.constant 16 : index

    call dpas_test(%cst_sys_dpth, %cst_rpt_cnt, %cst_N) : (index, index, index) -> ()

    // %memref = gpu.alloc  () {gpu.alloc_shared} : memref<8xf32>
    // %memref_2 = gpu.alloc  () {gpu.alloc_shared} : memref<8xf32>
    // %memref_3 = gpu.alloc  () {gpu.alloc_shared} : memref<8xf32>
    // %0 = memref.cast %memref : memref<8xf32> to memref<?xf32>
    // %1 = memref.cast %memref_2 : memref<8xf32> to memref<?xf32>
    // %2 = memref.cast %memref_3 : memref<8xf32> to memref<?xf32>
    // call @fillResource1DFloat(%0, %cst_0) : (memref<?xf32>, f32) -> ()
    // call @fillResource1DFloat(%1, %cst_1) : (memref<?xf32>, f32) -> ()
    // call @fillResource1DFloat(%2, %cst) : (memref<?xf32>, f32) -> ()
    
    // %memref_main_v0 = gpu.alloc  () {gpu.alloc_shared} : memref<8xf32>
    // %memref_main_v1
    // %memref_main_v2

    return
  }
    
}
  

  
