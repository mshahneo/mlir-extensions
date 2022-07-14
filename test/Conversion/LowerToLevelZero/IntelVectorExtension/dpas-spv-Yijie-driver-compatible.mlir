module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, VectorAnyINTEL, VectorComputeINTEL, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, VectorAnyINTEL, FunctionFloatControlINTEL], [SPV_EXT_shader_atomic_float_add, SPV_INTEL_vector_compute, SPV_INTEL_float_controls2]>, {}>}  {

  // function to setup the launch and launch the kernel
  // args: size_t systolic_depth, size_t repeat_cnt, size_t N
  func.func @dpas_gpu(%arg_sys_dpth: index, %arg_rpt_cnt: index, %arg_N: index, %arg_v0 : memref<?xi8>, %arg_v1 : memref<?xi8>, %arg_v2 : memref<?xi8>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    
    // @Question: Understanding the Intel GPU structure
    // blocks and threads are for indexing 
    gpu.launch_func @dpas_module::@dpas_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%arg_v0 : memref<?xi8>, %arg_v1 : memref<?xi8>, %arg_v2 : memref<?xi8>) 
    return
  }

  // SPIR-V DPAS module, it holds the DPAS kernel
  spv.module @__spv__dpas_module Physical64 OpenCL requires #spv.vce<v1.0, [Int8, Int16, Int64, Float16, Kernel, Addresses, Linkage, Vector16, VectorAnyINTEL, Float16Buffer, VectorComputeINTEL, FunctionFloatControlINTEL], [SPV_INTEL_float_controls2, SPV_INTEL_vector_compute]> {   
    // DPAS kernel 
    spv.func @dpas_kernel(%arg0: !spv.ptr<i8, CrossWorkgroup>, %arg1: !spv.ptr<i8, CrossWorkgroup>, %arg2: !spv.ptr<i8, CrossWorkgroup>)  "DontInline"  attributes {spv.entry_point_abi = {local_size = dense<0> : vector<3xi32>}, workgroup_attributions = 0 : i64, VectorComputeFunctionINTEL}  {
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

        // Cast the uchar pointers (i8 ptr) to ulongs (i64)
        %arg_0 = spv.FunctionCall @llvm_genx_address_convert_i64_p1i8(%arg0) : (!spv.ptr<i8, CrossWorkgroup>) -> i64
        %arg_1 = spv.FunctionCall @llvm_genx_address_convert_i64_p1i8(%arg1) : (!spv.ptr<i8, CrossWorkgroup>) -> i64
        %arg_2 = spv.FunctionCall @llvm_genx_address_convert_i64_p1i8(%arg2) : (!spv.ptr<i8, CrossWorkgroup>) -> i64

        // Load vector 0 using load stateless
        // Signless version
        %v0 =  spv.FunctionCall @llvm_genx_lsc_load_stateless_v64f32_i1_i64(%true, %uchar_0, %uchar_0, %uchar_0, %ushort_1, %uint_0, %uchar_3, %uchar_8, %uchar_2, %uchar_0, %arg_0, %uint_0) : (i1, i8, i8, i8, i16, i32, i8, i8, i8, i8, i64, i32) -> vector<64xf32> 
        // Load vector 1 using load stateless
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
    }

    spv.EntryPoint "Kernel" @dpas_kernel
    spv.ExecutionMode @dpas_kernel "ContractionOff"
    spv.ExecutionMode @dpas_kernel "SharedLocalMemorySizeINTEL", 0
    // Utility function declarations (Intel intrinsics)
    spv.func @llvm_genx_address_convert_i64_p1i8(%arg: !spv.ptr<i8, CrossWorkgroup>) -> i64 "Pure" attributes {LinkageAttributes=["llvm.genx.address.convert.i64.p1i8", "Import"], VectorComputeFunctionINTEL} //{}

    spv.func @llvm_genx_lsc_load_stateless_v64f32_i1_i64(%arg0 : i1, %arg1 : i8, %arg2 : i8, %arg3 : i8, %arg4 : i16, %arg5 : i32, %arg6 : i8, %arg7 : i8, %arg8 : i8, %arg9 : i8, %arg10 : i64, %arg11 : i32) -> vector<64 x f32> "Const" attributes{LinkageAttributes=["llvm.genx.lsc.load.stateless.v64f32.i1.i64", "Import"], VectorComputeFunctionINTEL} //{} //attributes {}

    spv.func @llvm_genx_lsc_load_stateless_v64i64_i1_i64(%arg0 : i1, %arg1 : i8, %arg2 : i8, %arg3 : i8, %arg4 : i16, %arg5 : i32, %arg6 : i8, %arg7 : i8, %arg8 : i8, %arg9 : i8, %arg10 : i64, %arg11 : i32) -> vector<64 x i64> "Const" attributes{LinkageAttributes=["llvm.genx.lsc.load.stateless.v64i64.i1.i64", "Import"], VectorComputeFunctionINTEL}//attributes {}

    spv.func @llvm_genx_lsc_load_stateless_v32i32_i1_i64(%arg0 : i1, %arg1 : i8, %arg2 : i8, %arg3 : i8, %arg4 : i16, %arg5 : i32, %arg6 : i8, %arg7 : i8, %arg8 : i8, %arg9 : i8, %arg10 : i64, %arg11 : i32) -> vector<32 x i32> "Const" attributes{LinkageAttributes=["llvm.genx.lsc.load.stateless.v32i32.i1.i64", "Import"], VectorComputeFunctionINTEL} //attributes {}

    spv.func @llvm_genx_dpas2_v64f32_v64f32_v128i32_v32i32(%arg0 : vector<64 x f32>, %arg1 : vector<128 x i32>, %arg2 : vector<32 x i32>, %arg3 : i32, %arg4 : i32, %arg5 : i32, %arg6 : i32, %arg7 : i32, %arg8 : i32) -> vector<64 x f32> "Pure" attributes{LinkageAttributes=["llvm.genx.dpas2.v64f32.v64f32.v128i32.v32i32", "Import"], VectorComputeFunctionINTEL} //attributes {}

    spv.func @llvm_genx_lsc_store_stateless_i1_i64_v64f32(%arg0 : i1, %arg1 : i8, %arg2 : i8, %arg3 : i8, %arg4 : i16, %arg5 : i32, %arg6 : i8, %arg7 : i8, %arg8 : i8, %arg9 : i8, %arg10 : vector<64 x f32>, %arg11 : vector<64 x f32>, %arg12 : i32)  "None" attributes{LinkageAttributes=["llvm.genx.lsc.store.stateless.i1.i64.v64f32", "Import"], VectorComputeFunctionINTEL} //attributes {} -> mlir::NoneType
  }

  // GPU module, almost same as the SPIR-V module but without 'spv' specific properties
  gpu.module @dpas_module {
    gpu.func @dpas_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi8>) kernel attributes {spv.entry_point_abi = {local_size = dense<0> : vector<3xi32>}} {
      gpu.return
    }
  }

  func.func @dpas_ref(%arg_sys_dpth: index, %arg_rpt_cnt: index, %arg_N: index, %arg_v0 : memref<?xf32>, %arg_v1 : memref<?xbf16>, %arg_v2 : memref<?xbf16>){
    return
  }

  func.func @dpas_test(%arg_sys_dpth: index, %arg_rpt_cnt: index, %arg_N: index){
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.100000e+00 : f32
    %cst_2 = arith.constant 2.200000e+00 : f32

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    
    // Allocate vectors to be passed to function

    // Setting up Vector v0
    %v0_size = arith.muli %arg_rpt_cnt, %arg_N : index 
    // %memref_v0_cpu = memref.alloc (%v0_size) : memref<?xf32>
    %v0_size_i8 = arith.muli %v0_size, %c4 : index

    %memref_v0_i8 = gpu.alloc (%v0_size_i8) {gpu.alloc_shared} : memref<?xi8>
    %memref_v0 = memref.view %memref_v0_i8[%c0][%v0_size] : memref<?xi8> to memref<?xf32>
    // Initialize v0 to 0
    // call @fillResource1DFloat(%memref_v0_cpu, %cst_0) : (memref<?xf32>, f32) -> ()
    call @fillResource1DFloat(%memref_v0, %cst_0) : (memref<?xf32>, f32) -> ()

    // Setting up the Vector v1 & v2
    // v1 is setup slightly differently than other vectors, since v1 is
    // expected to be bf16 by the dpas instruction, but can not be passed
    // in SPIR-V (SPIR-V does not support bf16), we first allocate v1
    // as i8 and then change the type (create views) to bf16 and f16
    // This way, both views point to the same vector, but accessed 
    // differently based what view is used
    // 
    // Since, in our case, the vector is essentially bf16, but needed to
    // have a view of f16 just be passed in SPIR-V and inside DPAS 
    // reinterpreted back bf16, we can safely use this approach
    //            / bf16 (initialization)         \
    // v1 = i8 -                                   -> 
    //            \ f16 (passed to SPIR-V kernel) /
    %tmp_sys_dpth = arith.muli %arg_sys_dpth, %c2 : index
    %v1_size = arith.muli %tmp_sys_dpth, %arg_N : index

    // Since, we are allocating bf16 as i8, %v1_size * 2 is used
    // for allocation size
    %v1_size_i8 =  arith.muli %v1_size, %c2 : index
    
    %memref_v1 = memref.alloc (%v1_size_i8) : memref<?xi8>

    // Create a view of bf16 vector
    %memref_v1_bf16 = memref.view %memref_v1[%c0][%v1_size] : memref<?xi8> to memref<?xbf16>
    // Create a view of f16 vector
    %memref_v1_f16 = memref.view %memref_v1[%c0][%v1_size] : memref<?xi8> to memref<?xf16>
    
    // @RESOLVED: @ISSUE:SPIR-V type does not support bf16, hence passing vector 1, and vector 2 as f16, will load bf16 from this vector using the intel vc-intrinsic    
    // %memref_v1_gpu = gpu.alloc (%v1_size) {gpu.alloc_shared} : memref<?xf16>
    
    // Initialize it to 1.1 as bf16, since that's the original data type for v1
    call @fillResource1DBFloat16(%memref_v1_bf16, %cst_1) : (memref<?xbf16>, f32) -> ()
    // call @fillResource1Df16(%memref_v1_gpu, %cst_1) : (memref<?xf16>, f16) -> ()

    // Setting up the Vector v2
    %v2_size = arith.muli %tmp_sys_dpth, %arg_rpt_cnt : index

    // Since, we are allocating bf16 as i8, %v2_size * 2 is used
    // for allocation size
    %v2_size_i8 =  arith.muli %v2_size, %c2 : index

    %memref_v2 = memref.alloc  (%v2_size_i8) : memref<?xi8>
    // Create a view of bf16 vector
    %memref_v2_bf16 = memref.view %memref_v2[%c0][%v2_size] : memref<? x i8> to memref<? x bf16>
    // Create a view of f16 vector
    %memref_v2_f16 = memref.view %memref_v2[%c0][%v2_size] : memref<? x i8> to memref<? x f16>

    // @@RESOLVED: ISSUE:SPIR-V type does not support bf16, hence passing vector 1, and vector 2 as f16, will load bf16 from this vector using the intel vc-intrinsic    
    // %memref_v2_gpu = gpu.alloc (%v2_size) {gpu.alloc_shared} : memref<?xf16>
    
    // Initialize it to 2.2 as bf16, since that's the original data type for v2
    call @fillResource1DBFloat16(%memref_v2_bf16, %cst_2) : (memref<?xbf16>, f32) -> ()
    // call @fillResource1Df16(%memref_v2_gpu, %cst_2) : (memref<?xf16>, f16) -> ()

    // Calling the reference function/CPU version
    call @dpas_ref(%arg_sys_dpth, %arg_rpt_cnt,  %arg_N, %memref_v0, %memref_v1_bf16, %memref_v2_bf16) : (index, index, index, memref<?xf32>, memref<?xbf16>, memref<?xbf16>) -> ()

    // Calling the GPU version, using f16 view of v1 and v2 vector
    call @dpas_gpu(%arg_sys_dpth, %arg_rpt_cnt,  %arg_N, %memref_v0_i8, %memref_v1, %memref_v2) : (index, index, index, memref<?xi8>, memref<?xi8>, memref<?xi8>) -> ()

    // Print the result
    %result = memref.cast %memref_v0 : memref<?xf32> to memref<*xf32>
    call @printMemrefF32(%result) : (memref<*xf32>) -> ()

    return
  }

  // main function
  func.func @main() {
    %cst_sys_dpth = arith.constant 8 : index
    %cst_rpt_cnt = arith.constant 4 : index
    %cst_N = arith.constant 16 : index

    call @dpas_test(%cst_sys_dpth, %cst_rpt_cnt, %cst_N) : (index, index, index) -> ()
    return
  }

  // Helper functions
  func.func private @fillResource1DBFloat16(memref<?xbf16>, f32)
  func.func private @fillResource1DFloat16(memref<?xf16>, f32)
  func.func private @fillResource1DFloat(memref<?xf32>, f32)
  func.func private @printMemrefBFloat16(memref<*xbf16>)
  func.func private @printMemrefFloat16(memref<*xf16>)
  func.func private @printMemrefF32(memref<*xf32>)

}

