// RUN: IMEX_ENABLE_LARGE_REG_FILE=1 %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: IMEX_ENABLE_LARGE_REG_FILE=1 %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<?x?xf16>, %B: memref<?x?xf16>, %C: memref<?x?xf16>, %M: index, %K: index, %N: index,  %wg_M: index, %wg_K: index, %wg_N: index, %sg_M: index, %sg_K: index, %sg_N: index, %wg_X: index, %wg_Y: index, %wg_Z: index, %sg_X: index, %sg_Y: index, %sg_Z: index, %prefetch_distance: index, %barrier_distance: index) -> memref<?x?xf16> attributes {llvm.emit_c_interface} {

    %A_gpu = gpu.alloc  (%M, %K) : memref<?x?xf16>
    gpu.memcpy %A_gpu, %A : memref<?x?xf16>, memref<?x?xf16>
    %B_gpu = gpu.alloc  (%K, %N) : memref<?x?xf16>
    gpu.memcpy %B_gpu, %B : memref<?x?xf16>, memref<?x?xf16>
    %C_gpu = gpu.alloc  (%M, %N) : memref<?x?xf16>
    gpu.memcpy %C_gpu, %C : memref<?x?xf16>, memref<?x?xf16>


    gpu.launch_func  @gemm_kernels::@gemm_kernel_wgX_256_wgY_256_Kblock_32_sgX_32_sgY_64_named_barrier blocks in (%wg_X, %wg_Y, %wg_Z) threads in (%sg_X, %sg_Y, %sg_Z) args(%A_gpu : memref<?x?xf16>, %B_gpu : memref<?x?xf16>, %C_gpu : memref<?x?xf16>, %M: index, %K: index, %N: index,%wg_M: index, %wg_K: index, %wg_N: index, %sg_M: index, %sg_K: index, %sg_N: index, %sg_X: index, %sg_Y: index, %prefetch_distance : index, %barrier_distance : index)

    %C_host = memref.alloc (%M, %N) : memref<?x?xf16>
    gpu.memcpy %C_host, %C_gpu : memref<?x?xf16>, memref<?x?xf16>

    gpu.dealloc  %A_gpu : memref<?x?xf16>
    gpu.dealloc  %B_gpu : memref<?x?xf16>
    gpu.dealloc  %C_gpu : memref<?x?xf16>
    return %C_host : memref<?x?xf16>
  }
  gpu.module @gemm_kernels attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @gemm_kernel_wgX_256_wgY_256_Kblock_32_sgX_32_sgY_64_split_barrier(%A: memref<?x?xf16>, %B: memref<?x?xf16>, %C: memref<?x?xf16>, %M: index, %K: index, %N: index,  %wg_M: index, %wg_K: index, %wg_N: index, %sg_M: index, %sg_K: index, %sg_N: index, %sg_X: index, %sg_Y: index, %prefetch_distance : index, %barrier_distance : index) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // constants
      %c256 = arith.constant 256 : index
      %c512 = arith.constant 512 : index
      %c128 = arith.constant 128 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c48 = arith.constant 48 : index
      %c16 = arith.constant 16 : index
      %c24 = arith.constant 24 : index
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      // get IDs
      %wg_id_x = gpu.block_id x
      %wg_id_y = gpu.block_id y
      // %sg_id = gpu.subgroup_id : index

      // each C wg tile is 256x256 and 32 SGs update it in 8x4 layout
      // C sg tile size is 32x64
      // SG layout for one C tile update
      // |0|1|2|3|
      // |4|5|6|7|
      // .........
      // |28|29|30|31|
      // --> y means cols
      // |
      // V x means rows

      // get unique sg ID in global context
      %global_sg_id_x = gpu.global_id x
      %global_sg_id_y = gpu.global_id y
      %local_sg_id_x = arith.remui %global_sg_id_x, %sg_X : index
      %local_sg_id_y = arith.remui %global_sg_id_y, %sg_Y : index

      // compute SG C tile offsets in x and y dims
      %C_sg_tile_offset_x = arith.muli %global_sg_id_x, %sg_M : index
      %C_sg_tile_offset_y = arith.muli %global_sg_id_y, %sg_N : index

      // each SG needs to do the follwoing compute to update its 32x64 sub tile
      // (32xK)x(Kx64)=(32x64)
      // DPAS size is (8x16)x(16x16)=(8x16)
      // K loop adavances in steps of 32, so inside the compute is (32x32)x(32x64) = (32x64)
      // So we need to (4x2) A tiles of size (8x16) and (2x4) B tiles of size (16x16)
      // tiled compute for a SG is (4x2x8x16)x(2x4x16x16)=(4x4x8x16)
      // this will require 32 DPAS ops (4x2x2) inside the K loop

      // WG tiles offsets for A, B and C
      %A_wg_tile_offset_x = arith.muli %wg_id_x, %wg_M : index
      %A_wg_tile_offset_y = arith.muli %wg_id_y, %wg_K : index

      %B_wg_tile_offset_x = arith.muli %wg_id_x, %wg_K : index
      %B_wg_tile_offset_y = arith.muli %wg_id_y, %wg_N : index


      %C_wg_tile_offset_x = arith.muli %wg_id_x, %wg_M : index
      %C_wg_tile_offset_y = arith.muli %wg_id_y, %wg_N : index

      // Linearized local SG ID
      %local_sg_id_temp = arith.muli %local_sg_id_x, %sg_Y : index
      %local_sg_id = arith.addi %local_sg_id_temp, %local_sg_id_y : index

      // prefetching A and B slice within the 256x256 WG tile
      //
      // prefetch the entire 256x32 slice of A WG tile, this means each subgroups needs to prefetch 8x32 slice
      // each 1x4 row of SGs do a colloborative prefetch of 8x32 slice of the 32x32 tile
      // SG 0 -> slice 0 |
      // SG 1 -> slice 1 |
      // SG 2 -> slice 2  > SG 0,1,2,3 share data prefetch from the top 32x32 tile.
      // SG 3 -> slice 3 |
      // SG 4 -> slice 4
      // ....
      // SG 31 -> slice 31
      %A_sg_prefetch_offset_x_temp = arith.muli %local_sg_id, %c8 : index
      %A_sg_prefetch_offset_x = arith.addi %A_sg_prefetch_offset_x_temp, %A_wg_tile_offset_x : index
      // create A preftech tiles and prefetch
      // stage 1
      %A_sg_prefetch_tile_iter0 = xegpu.create_nd_tdesc %A[%A_sg_prefetch_offset_x, %c0], [%M, %K], [%K, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<8x32xf16>

      // Stage 1 to prefetch_distance
      %A_nth_prefetch_tile = scf.for %i = %c0 to %prefetch_distance step %c1 iter_args(%current_tile = %A_sg_prefetch_tile_iter0) -> !xegpu.tensor_desc<8x32xf16> {
        xegpu.prefetch_nd %current_tile {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>
        %next_tile = xegpu.update_nd_offset %current_tile, [%c0, %sg_K] : !xegpu.tensor_desc<8x32xf16>
        scf.yield %next_tile : !xegpu.tensor_desc<8x32xf16>
      }

      // prefetch the entire 32x256 slice of B WG tile, we still use the prefetch size 8x32.
      // SGs have 8x4 layout. In this case 8 subgroups must do a colloborative  prefetch of 32x64 tile.
      // this because the B tile arrangement within the 32x256 slice is as follows
      // 32x64 | 32x64 | 32x64 | 32x64
      // in terms of 8x32 slices the arrangement is,
      // 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32
      // 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32
      // 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32
      // 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32
      // So SGs 0,1,2,3,....31 prefetch in following fashion
      // | 0  | 16||  1 | 17 || 2  | 18 || 3 | 19 |
      // | 4  | 20||  5 | 21 || 6  | 22 || 7 | 23 |
      // | 8  | 24||  9 | 25 || 10 | 26 || 11| 27 |
      // | 12 | 28|| 13 | 29 || 14 | 30 || 15| 31 |
      // For example, SGs 0,4,8,12,16,20,24,28 share the data in left 32x64 tile of B slice.

      // calculate the x offsets and y offsets within the 32x256 slice
      %B_sg_prefetch_offset_x_temp0 = arith.remui %local_sg_id_x, %sg_Y : index
      %B_sg_prefetch_offset_x = arith.muli %B_sg_prefetch_offset_x_temp0, %sg_X : index
      %B_sg_prefetch_offset_y_temp0 = arith.muli %local_sg_id_y, %sg_N : index
      %B_sg_prefetch_offset_y_temp1 = arith.divui %local_sg_id_x, %sg_Y : index
      %B_sg_prefetch_offset_y_temp2 = arith.muli %B_sg_prefetch_offset_y_temp1, %sg_M : index
      %B_sg_prefetch_offset_y_temp3 = arith.addi %B_sg_prefetch_offset_y_temp0, %B_sg_prefetch_offset_y_temp2 : index
      %B_sg_prefetch_offset_y = arith.addi %B_wg_tile_offset_y, %B_sg_prefetch_offset_y_temp3 : index

      // create B prefetch tiles and prefetch
      %B_sg_prefetch_tile_iter0 = xegpu.create_nd_tdesc %B[%B_sg_prefetch_offset_x, %B_sg_prefetch_offset_y], [%K, %N], [%N, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<8x32xf16>

      // Prefetch B tiles within a scf.for loop
      %B_nth_prefetch_tile = scf.for %i = %c0 to %prefetch_distance step %c1 iter_args(%current_tile = %B_sg_prefetch_tile_iter0) -> !xegpu.tensor_desc<8x32xf16> {
        xegpu.prefetch_nd %current_tile {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>
        %next_tile = xegpu.update_nd_offset %current_tile, [%sg_K, %c0] : !xegpu.tensor_desc<8x32xf16>
        scf.yield %next_tile : !xegpu.tensor_desc<8x32xf16>
      }


      // two 32x16 = 32x32 A tiles from 256x32 WG slice
      %A_sg_init_tile_0 = xegpu.create_nd_tdesc %A[%C_sg_tile_offset_x, %c0], [%M, %K], [%K, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>

      //create B tiles, total of 32x64 = 4 B tiles of size 32x16
      %B_sg_init_tile_0 = xegpu.create_nd_tdesc %B[%c0, %C_sg_tile_offset_y], [%K, %N], [%N, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>

      %B_sg_init_tile_1 = xegpu.update_nd_offset %B_sg_init_tile_0, [%c0, %c32] :  !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>

      // init 16 C tiles of size 8x16 each is initialized to 0.0 assuming a zero C matrix
      %zero_vec = arith.constant dense<0.0> : vector<128xf32>
      %c_init_val_0_0 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_0_1 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_0_2 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_0_3 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_1_0 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_1_1 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_1_2 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_1_3 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_2_0 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_2_1 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_2_2 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_2_3 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_3_0 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_3_1 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_3_2 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_3_3 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>


      xegpu.alloc_nbarrier 16
      %nbarrier_id = arith.constant 1 : i8
      %num_threads = arith.constant 32 : i8
      %nbarrier = xegpu.init_nbarrier %nbarrier_id, %num_threads : i8, i8 -> !xegpu.nbarrier
      %barrier_nth_iter = arith.muli %sg_K, %barrier_distance : index
      // K loop advances in 32 steps
      %k_loop_result:21 = scf.for %k = %c0 to %K step %sg_K iter_args (
          %A_tile_0 = %A_sg_init_tile_0,

          %B_tile_0 = %B_sg_init_tile_0,
          %B_tile_1 = %B_sg_init_tile_1,

          %c_val_0_0 = %c_init_val_0_0,
          %c_val_0_1 = %c_init_val_0_1,
          %c_val_0_2 = %c_init_val_0_2,
          %c_val_0_3 = %c_init_val_0_3,
          %c_val_1_0 = %c_init_val_1_0,
          %c_val_1_1 = %c_init_val_1_1,
          %c_val_1_2 = %c_init_val_1_2,
          %c_val_1_3 = %c_init_val_1_3,
          %c_val_2_0 = %c_init_val_2_0,
          %c_val_2_1 = %c_init_val_2_1,
          %c_val_2_2 = %c_init_val_2_2,
          %c_val_2_3 = %c_init_val_2_3,
          %c_val_3_0 = %c_init_val_3_0,
          %c_val_3_1 = %c_init_val_3_1,
          %c_val_3_2 = %c_init_val_3_2,
          %c_val_3_3 = %c_init_val_3_3,

          %A_prefetch_tile = %A_nth_prefetch_tile,
          %B_prefetch_tile = %B_nth_prefetch_tile
          ) ->
          (!xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
          !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
          !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
          vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,
          !xegpu.tensor_desc<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
          )
          {
        // all SGs must arrive here first
        %every_nth_iter = arith.remui %k, %barrier_nth_iter : index
        %every_nth_iter_i32 = arith.index_cast %every_nth_iter : index to i32
        %every_nth_iter_cond = arith.cmpi eq, %every_nth_iter_i32, %c0_i32 : i32
        scf.if %every_nth_iter_cond  {
          xegpu.nbarrier_arrive %nbarrier : !xegpu.nbarrier
        }
        // load A tiles
        %a_val = xegpu.load_nd %A_tile_0 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>> -> vector<2x32x16xf16>
        %a_val_0 = vector.extract %a_val [0] : vector<32x16xf16> from vector<2x32x16xf16>
        %a_val_1 = vector.extract %a_val [1] : vector<32x16xf16> from vector<2x32x16xf16>

        // load B tiles
        %b_val_arr_0 = xegpu.load_nd %B_tile_0 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>> -> vector<2x16x16x2xf16>
        %b_val_arr_1 = xegpu.load_nd %B_tile_1 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>> -> vector<2x16x16x2xf16>

        %b_val_0 = vector.extract %b_val_arr_0 [0] : vector<16x16x2xf16> from vector<2x16x16x2xf16>
        %b_val_1 = vector.extract %b_val_arr_0 [1] : vector<16x16x2xf16> from vector<2x16x16x2xf16>
        %b_val_2 = vector.extract %b_val_arr_1 [0] : vector<16x16x2xf16> from vector<2x16x16x2xf16>
        %b_val_3 = vector.extract %b_val_arr_1 [1] : vector<16x16x2xf16> from vector<2x16x16x2xf16>

        xegpu.compile_hint

        // prefetch A and B tiles
        xegpu.prefetch_nd %A_prefetch_tile {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>
        xegpu.prefetch_nd %B_prefetch_tile {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>

        //
        xegpu.compile_hint

        // advance A and B prefetch tiles
        %next_A_prefetch_tile = xegpu.update_nd_offset %A_prefetch_tile, [%c0, %sg_K] : !xegpu.tensor_desc<8x32xf16>
        %next_B_prefetch_tile = xegpu.update_nd_offset %B_prefetch_tile, [%sg_K, %c0] : !xegpu.tensor_desc<8x32xf16>
        // advance A and B tiles
        %next_A_tile_0 = xegpu.update_nd_offset %A_tile_0, [%c0, %sg_K] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>
        // %next_A_tile_1 = xegpu.update_nd_offset %A_tile_1, [%c0, %c32] : !xegpu.tensor_desc<32x16xf16>

        %next_B_tile_0 = xegpu.update_nd_offset %B_tile_0, [%sg_K, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>
        %next_B_tile_1 = xegpu.update_nd_offset %B_tile_1, [%sg_K, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>
        // %next_B_tile_2 = xegpu.update_nd_offset %B_tile_2, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16>
        // %next_B_tile_3 = xegpu.update_nd_offset %B_tile_3, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16>

        xegpu.compile_hint
        %a_val_0_flat = vector.shape_cast %a_val_0 : vector<32x16xf16> to vector<512xf16>
        %a_val_1_flat = vector.shape_cast %a_val_1 : vector<32x16xf16> to vector<512xf16>
        %a_val_0_0_flat = vector.extract_strided_slice %a_val_0_flat { offsets = [0], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_0_0 = vector.shape_cast %a_val_0_0_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_1_0_flat = vector.extract_strided_slice %a_val_0_flat { offsets = [128], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_1_0 = vector.shape_cast %a_val_1_0_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_2_0_flat = vector.extract_strided_slice  %a_val_0_flat { offsets = [256], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_2_0 = vector.shape_cast %a_val_2_0_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_3_0_flat = vector.extract_strided_slice %a_val_0_flat { offsets = [384], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_3_0 = vector.shape_cast %a_val_3_0_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_0_1_flat = vector.extract_strided_slice %a_val_1_flat { offsets = [0], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_0_1 = vector.shape_cast %a_val_0_1_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_1_1_flat = vector.extract_strided_slice %a_val_1_flat {offsets = [128], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_1_1 = vector.shape_cast %a_val_1_1_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_2_1_flat = vector.extract_strided_slice %a_val_1_flat { offsets = [256], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_2_1 = vector.shape_cast %a_val_2_1_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_3_1_flat = vector.extract_strided_slice %a_val_1_flat { offsets = [384], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_3_1 = vector.shape_cast %a_val_3_1_flat : vector<128xf16> to vector<8x16xf16>


        %b_val_0_flat = vector.shape_cast %b_val_0 : vector<16x16x2xf16> to vector<512xf16>
        %b_val_1_flat = vector.shape_cast %b_val_1 : vector<16x16x2xf16> to vector<512xf16>
        %b_val_2_flat = vector.shape_cast %b_val_2 : vector<16x16x2xf16> to vector<512xf16>
        %b_val_3_flat = vector.shape_cast %b_val_3 : vector<16x16x2xf16> to vector<512xf16>
        %b_val_0_0_flat = vector.extract_strided_slice %b_val_0_flat { offsets = [0], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_0_0 = vector.shape_cast %b_val_0_0_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_1_0_flat = vector.extract_strided_slice %b_val_0_flat { offsets = [256], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_1_0 = vector.shape_cast %b_val_1_0_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_0_1_flat = vector.extract_strided_slice %b_val_1_flat { offsets = [0], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_0_1 = vector.shape_cast %b_val_0_1_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_1_1_flat = vector.extract_strided_slice %b_val_1_flat { offsets = [256], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_1_1 = vector.shape_cast %b_val_1_1_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_0_2_flat = vector.extract_strided_slice %b_val_2_flat { offsets = [0], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_0_2 = vector.shape_cast %b_val_0_2_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_1_2_flat = vector.extract_strided_slice %b_val_2_flat { offsets = [256], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_1_2 = vector.shape_cast %b_val_1_2_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_0_3_flat = vector.extract_strided_slice  %b_val_3_flat { offsets = [0], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_0_3 = vector.shape_cast %b_val_0_3_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_1_3_flat = vector.extract_strided_slice %b_val_3_flat {offsets = [256], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_1_3 = vector.shape_cast %b_val_1_3_flat : vector<256xf16> to vector<8x16x2xf16>

        // do DPAS
        xegpu.compile_hint
        %new_c_val_0_0_temp = xegpu.dpas %a_val_0_0, %b_val_0_0, %c_val_0_0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_0 = xegpu.dpas %a_val_0_1, %b_val_1_0, %new_c_val_0_0_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_0_temp = xegpu.dpas %a_val_1_0, %b_val_0_0, %c_val_1_0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_0 = xegpu.dpas %a_val_1_1, %b_val_1_0, %new_c_val_1_0_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_0_temp = xegpu.dpas %a_val_2_0, %b_val_0_0, %c_val_2_0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_0 = xegpu.dpas %a_val_2_1, %b_val_1_0, %new_c_val_2_0_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_0_temp = xegpu.dpas %a_val_3_0, %b_val_0_0, %c_val_3_0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_0 = xegpu.dpas %a_val_3_1, %b_val_1_0, %new_c_val_3_0_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        %new_c_val_0_1_temp = xegpu.dpas %a_val_0_0, %b_val_0_1, %c_val_0_1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_1 = xegpu.dpas %a_val_0_1, %b_val_1_1, %new_c_val_0_1_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_1_temp = xegpu.dpas %a_val_1_0, %b_val_0_1, %c_val_1_1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_1 = xegpu.dpas %a_val_1_1, %b_val_1_1, %new_c_val_1_1_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_1_temp = xegpu.dpas %a_val_2_0, %b_val_0_1, %c_val_2_1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_1 = xegpu.dpas %a_val_2_1, %b_val_1_1, %new_c_val_2_1_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_1_temp = xegpu.dpas %a_val_3_0, %b_val_0_1, %c_val_3_1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_1 = xegpu.dpas %a_val_3_1, %b_val_1_1, %new_c_val_3_1_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        %new_c_val_0_2_temp = xegpu.dpas %a_val_0_0, %b_val_0_2, %c_val_0_2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_2 = xegpu.dpas %a_val_0_1, %b_val_1_2, %new_c_val_0_2_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_2_temp = xegpu.dpas %a_val_1_0, %b_val_0_2, %c_val_1_2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_2 = xegpu.dpas %a_val_1_1, %b_val_1_2, %new_c_val_1_2_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_2_temp = xegpu.dpas %a_val_2_0, %b_val_0_2, %c_val_2_2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_2 = xegpu.dpas %a_val_2_1, %b_val_1_2, %new_c_val_2_2_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_2_temp = xegpu.dpas %a_val_3_0, %b_val_0_2, %c_val_3_2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_2 = xegpu.dpas %a_val_3_1, %b_val_1_2, %new_c_val_3_2_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        %new_c_val_0_3_temp = xegpu.dpas %a_val_0_0, %b_val_0_3, %c_val_0_3 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_3 = xegpu.dpas %a_val_0_1, %b_val_1_3, %new_c_val_0_3_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_3_temp = xegpu.dpas %a_val_1_0, %b_val_0_3, %c_val_1_3 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_3 = xegpu.dpas %a_val_1_1, %b_val_1_3, %new_c_val_1_3_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_3_temp = xegpu.dpas %a_val_2_0, %b_val_0_3, %c_val_2_3 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_3 = xegpu.dpas %a_val_2_1, %b_val_1_3, %new_c_val_2_3_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_3_temp = xegpu.dpas %a_val_3_0, %b_val_0_3, %c_val_3_3 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        xegpu.compile_hint
        %new_c_val_3_3 = xegpu.dpas %a_val_3_1, %b_val_1_3, %new_c_val_3_3_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        xegpu.compile_hint
        //  barrier wait
        scf.if %every_nth_iter_cond {
          xegpu.nbarrier_wait %nbarrier : !xegpu.nbarrier
        }

        scf.yield %next_A_tile_0, %next_B_tile_0, %next_B_tile_1,
                  %new_c_val_0_0, %new_c_val_0_1, %new_c_val_0_2, %new_c_val_0_3, %new_c_val_1_0, %new_c_val_1_1, %new_c_val_1_2, %new_c_val_1_3, %new_c_val_2_0, %new_c_val_2_1, %new_c_val_2_2, %new_c_val_2_3, %new_c_val_3_0, %new_c_val_3_1, %new_c_val_3_2, %new_c_val_3_3,
                  %next_A_prefetch_tile, %next_B_prefetch_tile
                  : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
                  !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
                  !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
                  vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,
                  !xegpu.tensor_desc<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      }

      // trunc all DPAS output tiles to f16
      %c_result_0_0_f16 = arith.truncf %k_loop_result#3 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_0_1_f16 = arith.truncf %k_loop_result#4 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_0_2_f16 = arith.truncf %k_loop_result#5 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_0_3_f16 = arith.truncf %k_loop_result#6 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_1_0_f16 = arith.truncf %k_loop_result#7 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_1_1_f16 = arith.truncf %k_loop_result#8 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_1_2_f16 = arith.truncf %k_loop_result#9 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_1_3_f16 = arith.truncf %k_loop_result#10 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_2_0_f16 = arith.truncf %k_loop_result#11 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_2_1_f16 = arith.truncf %k_loop_result#12 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_2_2_f16 = arith.truncf %k_loop_result#13 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_2_3_f16 = arith.truncf %k_loop_result#14 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_3_0_f16 = arith.truncf %k_loop_result#15 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_3_1_f16 = arith.truncf %k_loop_result#16 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_3_2_f16 = arith.truncf %k_loop_result#17 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_3_3_f16 = arith.truncf %k_loop_result#18 : vector<8x16xf32> to vector<8x16xf16>

      // each SG needs to store the result of K loop into a 32x64 tile in C matrix. This is organized in 8x16 DPAS tiles
      // in the layout of 4x4x8x16. The max store size HW supoprt in f16 is 8x32. So we combine two 8x16 DPAS tiles
      // horizontally using vector.shuffle to get the required store size. The store layout then will 4x2x8x32 i.e.
      // we have 8 stores of size 8x32 in the layout 4x2.

      %c_result_8x32_0_0_t1 = vector.shuffle %c_result_0_0_f16, %c_result_0_1_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_0_0_t2 = vector.shape_cast %c_result_8x32_0_0_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_0_0 = vector.shape_cast %c_result_8x32_0_0_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_00 = xegpu.create_nd_tdesc %C[%C_sg_tile_offset_x, %C_sg_tile_offset_y], [%M, %N], [%N, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_0_0, %c_sg_tile_00 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint

      %c_result_8x32_0_1_t1 = vector.shuffle %c_result_0_2_f16, %c_result_0_3_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_0_1_t2 = vector.shape_cast %c_result_8x32_0_1_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_0_1 = vector.shape_cast %c_result_8x32_0_1_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_01 = xegpu.update_nd_offset %c_sg_tile_00, [%c0, %c32]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_0_1, %c_sg_tile_01 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint

      %c_result_8x32_1_0_t1 = vector.shuffle %c_result_1_0_f16, %c_result_1_1_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_1_0_t2 = vector.shape_cast %c_result_8x32_1_0_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_1_0 = vector.shape_cast %c_result_8x32_1_0_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_10 = xegpu.update_nd_offset %c_sg_tile_00, [%c8, %c0]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_1_0, %c_sg_tile_10 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint


      %c_result_8x32_1_1_t1 = vector.shuffle %c_result_1_2_f16, %c_result_1_3_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_1_1_t2 = vector.shape_cast %c_result_8x32_1_1_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_1_1 = vector.shape_cast %c_result_8x32_1_1_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_11 = xegpu.update_nd_offset %c_sg_tile_01, [%c8, %c0]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_1_1, %c_sg_tile_11 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint

      %c_result_8x32_2_0_t1 = vector.shuffle %c_result_2_0_f16, %c_result_2_1_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_2_0_t2 = vector.shape_cast %c_result_8x32_2_0_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_2_0 = vector.shape_cast %c_result_8x32_2_0_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_20 = xegpu.update_nd_offset %c_sg_tile_10, [%c8, %c0]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_2_0, %c_sg_tile_20 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint

      %c_result_8x32_2_1_t1 = vector.shuffle %c_result_2_2_f16, %c_result_2_3_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_2_1_t2 = vector.shape_cast %c_result_8x32_2_1_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_2_1 = vector.shape_cast %c_result_8x32_2_1_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_21 = xegpu.update_nd_offset %c_sg_tile_11, [%c8, %c0]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_2_1, %c_sg_tile_21 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint

      %c_result_8x32_3_0_t1 = vector.shuffle %c_result_3_0_f16, %c_result_3_1_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_3_0_t2 = vector.shape_cast %c_result_8x32_3_0_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_3_0 = vector.shape_cast %c_result_8x32_3_0_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_30 = xegpu.update_nd_offset %c_sg_tile_20, [%c8, %c0]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_3_0, %c_sg_tile_30 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint

      %c_result_8x32_3_1_t1 = vector.shuffle %c_result_3_2_f16, %c_result_3_3_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_3_1_t2 = vector.shape_cast %c_result_8x32_3_1_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_3_1 = vector.shape_cast %c_result_8x32_3_1_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_31 = xegpu.update_nd_offset %c_sg_tile_21, [%c8, %c0]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_3_1, %c_sg_tile_31 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>

      gpu.return
    }

    gpu.func @gemm_kernel_wgX_256_wgY_256_Kblock_32_sgX_32_sgY_64_named_barrier(%A: memref<?x?xf16>, %B: memref<?x?xf16>, %C: memref<?x?xf16>, %M: index, %K: index, %N: index,  %wg_M: index, %wg_K: index, %wg_N: index, %sg_M: index, %sg_K: index, %sg_N: index, %sg_X: index, %sg_Y: index, %prefetch_distance : index, %barrier_distance : index) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // constants
      %c256 = arith.constant 256 : index
      %c512 = arith.constant 512 : index
      %c128 = arith.constant 128 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c48 = arith.constant 48 : index
      %c16 = arith.constant 16 : index
      %c24 = arith.constant 24 : index
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      // get IDs
      %wg_id_x = gpu.block_id x
      %wg_id_y = gpu.block_id y
      // %sg_id = gpu.subgroup_id : index

      // each C wg tile is 256x256 and 32 SGs update it in 8x4 layout
      // C sg tile size is 32x64
      // SG layout for one C tile update
      // |0|1|2|3|
      // |4|5|6|7|
      // .........
      // |28|29|30|31|
      // --> y means cols
      // |
      // V x means rows

      // get unique sg ID in global context
      %global_sg_id_x = gpu.global_id x
      %global_sg_id_y = gpu.global_id y
      %local_sg_id_x = arith.remui %global_sg_id_x, %sg_X : index
      %local_sg_id_y = arith.remui %global_sg_id_y, %sg_Y : index

      // compute SG C tile offsets in x and y dims
      %C_sg_tile_offset_x = arith.muli %global_sg_id_x, %sg_M : index
      %C_sg_tile_offset_y = arith.muli %global_sg_id_y, %sg_N : index

      // each SG needs to do the follwoing compute to update its 32x64 sub tile
      // (32xK)x(Kx64)=(32x64)
      // DPAS size is (8x16)x(16x16)=(8x16)
      // K loop adavances in steps of 32, so inside the compute is (32x32)x(32x64) = (32x64)
      // So we need to (4x2) A tiles of size (8x16) and (2x4) B tiles of size (16x16)
      // tiled compute for a SG is (4x2x8x16)x(2x4x16x16)=(4x4x8x16)
      // this will require 32 DPAS ops (4x2x2) inside the K loop

      // WG tiles offsets for A, B and C
      %A_wg_tile_offset_x = arith.muli %wg_id_x, %wg_M : index
      %A_wg_tile_offset_y = arith.muli %wg_id_y, %wg_K : index

      %B_wg_tile_offset_x = arith.muli %wg_id_x, %wg_K : index
      %B_wg_tile_offset_y = arith.muli %wg_id_y, %wg_N : index


      %C_wg_tile_offset_x = arith.muli %wg_id_x, %wg_M : index
      %C_wg_tile_offset_y = arith.muli %wg_id_y, %wg_N : index

      // Linearized local SG ID
      %local_sg_id_temp = arith.muli %local_sg_id_x, %sg_Y : index
      %local_sg_id = arith.addi %local_sg_id_temp, %local_sg_id_y : index

      // prefetching A and B slice within the 256x256 WG tile
      //
      // prefetch the entire 256x32 slice of A WG tile, this means each subgroups needs to prefetch 8x32 slice
      // each 1x4 row of SGs do a colloborative prefetch of 8x32 slice of the 32x32 tile
      // SG 0 -> slice 0 |
      // SG 1 -> slice 1 |
      // SG 2 -> slice 2  > SG 0,1,2,3 share data prefetch from the top 32x32 tile.
      // SG 3 -> slice 3 |
      // SG 4 -> slice 4
      // ....
      // SG 31 -> slice 31
      %A_sg_prefetch_offset_x_temp = arith.muli %local_sg_id, %c8 : index
      %A_sg_prefetch_offset_x = arith.addi %A_sg_prefetch_offset_x_temp, %A_wg_tile_offset_x : index
      // create A preftech tiles and prefetch
      // stage 1
      %A_sg_prefetch_tile_iter0 = xegpu.create_nd_tdesc %A[%A_sg_prefetch_offset_x, %c0], [%M, %K], [%K, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<8x32xf16>

      // Stage 1 to prefetch_distance
      %A_nth_prefetch_tile = scf.for %i = %c0 to %prefetch_distance step %c1 iter_args(%current_tile = %A_sg_prefetch_tile_iter0) -> !xegpu.tensor_desc<8x32xf16> {
        xegpu.prefetch_nd %current_tile {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>
        %next_tile = xegpu.update_nd_offset %current_tile, [%c0, %sg_K] : !xegpu.tensor_desc<8x32xf16>
        scf.yield %next_tile : !xegpu.tensor_desc<8x32xf16>
      }

      // prefetch the entire 32x256 slice of B WG tile, we still use the prefetch size 8x32.
      // SGs have 8x4 layout. In this case 8 subgroups must do a colloborative  prefetch of 32x64 tile.
      // this because the B tile arrangement within the 32x256 slice is as follows
      // 32x64 | 32x64 | 32x64 | 32x64
      // in terms of 8x32 slices the arrangement is,
      // 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32
      // 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32
      // 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32
      // 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32 || 8x32 | 8x32
      // So SGs 0,1,2,3,....31 prefetch in following fashion
      // | 0  | 16||  1 | 17 || 2  | 18 || 3 | 19 |
      // | 4  | 20||  5 | 21 || 6  | 22 || 7 | 23 |
      // | 8  | 24||  9 | 25 || 10 | 26 || 11| 27 |
      // | 12 | 28|| 13 | 29 || 14 | 30 || 15| 31 |
      // For example, SGs 0,4,8,12,16,20,24,28 share the data in left 32x64 tile of B slice.

      // calculate the x offsets and y offsets within the 32x256 slice
      %B_sg_prefetch_offset_x_temp0 = arith.remui %local_sg_id_x, %sg_Y : index
      %B_sg_prefetch_offset_x = arith.muli %B_sg_prefetch_offset_x_temp0, %sg_X : index
      %B_sg_prefetch_offset_y_temp0 = arith.muli %local_sg_id_y, %sg_N : index
      %B_sg_prefetch_offset_y_temp1 = arith.divui %local_sg_id_x, %sg_Y : index
      %B_sg_prefetch_offset_y_temp2 = arith.muli %B_sg_prefetch_offset_y_temp1, %sg_M : index
      %B_sg_prefetch_offset_y_temp3 = arith.addi %B_sg_prefetch_offset_y_temp0, %B_sg_prefetch_offset_y_temp2 : index
      %B_sg_prefetch_offset_y = arith.addi %B_wg_tile_offset_y, %B_sg_prefetch_offset_y_temp3 : index

      // create B prefetch tiles and prefetch
      %B_sg_prefetch_tile_iter0 = xegpu.create_nd_tdesc %B[%B_sg_prefetch_offset_x, %B_sg_prefetch_offset_y], [%K, %N], [%N, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<8x32xf16>

      // Prefetch B tiles within a scf.for loop
      %B_nth_prefetch_tile = scf.for %i = %c0 to %prefetch_distance step %c1 iter_args(%current_tile = %B_sg_prefetch_tile_iter0) -> !xegpu.tensor_desc<8x32xf16> {
        xegpu.prefetch_nd %current_tile {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>
        %next_tile = xegpu.update_nd_offset %current_tile, [%sg_K, %c0] : !xegpu.tensor_desc<8x32xf16>
        scf.yield %next_tile : !xegpu.tensor_desc<8x32xf16>
      }


      // two 32x16 = 32x32 A tiles from 256x32 WG slice
      %A_sg_init_tile_0 = xegpu.create_nd_tdesc %A[%C_sg_tile_offset_x, %c0], [%M, %K], [%K, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>

      //create B tiles, total of 32x64 = 4 B tiles of size 32x16
      %B_sg_init_tile_0 = xegpu.create_nd_tdesc %B[%c0, %C_sg_tile_offset_y], [%K, %N], [%N, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>

      %B_sg_init_tile_1 = xegpu.update_nd_offset %B_sg_init_tile_0, [%c0, %c32] :  !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>

      // init 16 C tiles of size 8x16 each is initialized to 0.0 assuming a zero C matrix
      %zero_vec = arith.constant dense<0.0> : vector<128xf32>
      %c_init_val_0_0 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_0_1 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_0_2 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_0_3 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_1_0 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_1_1 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_1_2 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_1_3 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_2_0 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_2_1 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_2_2 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_2_3 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_3_0 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_3_1 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_3_2 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_3_3 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>


      // Multi nbarrier implementation,
      // one set nbarrier is used to sync subgroups with same sg_id_x (local_sg_id_x)
      // second set nbarrier us used to sync subgroups with same sg_id_y (local_sg_id_y)
      // In this case wg_size = 8,4 (sg_X = 8; sg_Y = 4)
      // So in Y-direction we need 4 nbarrier (to sync subgroups with same sg_id_y)
      // In X-direction we need 8 nbarrier (to sync subgroups with same sg_id_x)

      // %c_wg_size_x = arith.constant 8 : index
      // %c_wg_size_y = arith.constant 4 : index
      %num_nbarrier = arith.addi %sg_X, %sg_Y : index // 8+4=12
      xegpu.alloc_nbarrier 12 // = 12

      // First set of nbarriers work across coloumns, we have 4 coloums of subgroups,
      // Hnece 4 nbrrier
      // Each nbarrier has 8 producers and consumers
      // nbarrier type is Producer_Consumer (https://gfxspecs.intel.com/Predator/Home/Index/57499)

      // %nbarrier_role = arith.constant 0 : i8
      // %nbarrier_threads_y = arith.constant 8 : i8
      %nbarrier_threads_y = arith.index_cast %sg_X : index to i8
      %nbarrier_id_y = arith.index_cast %local_sg_id_y : index to i8
      %nbarrier_y = xegpu.init_nbarrier %nbarrier_id_y, %nbarrier_threads_y : i8, i8 -> !xegpu.nbarrier

      // Second set of barriers work on across rows of subgroups,
      // we have 8 rows of subgroups. Hnece, 8 nbarrier
      // Each nbarrier has 4 producers and consumers
      // nbarrier type is Producer_Consumer (https://gfxspecs.intel.com/Predator/Home/Index/57499)

      // We already have 4 (=%c_wg_size_y=sg_Y) nbarriers with id 0-3,
      // Now the next set of barrier id would start from 4, hence,
      // %nbarrier_threads_x = arith.constant 4 : i8
      %nbarrier_threads_x = arith.index_cast %sg_Y : index to i8
      %index_nbarrier_id_x = arith.addi %sg_Y, %local_sg_id_x : index
      %nbarrier_id_x = arith.index_cast %index_nbarrier_id_x : index to i8
      %nbarrier_x = xegpu.init_nbarrier %nbarrier_id_x, %nbarrier_threads_x : i8, i8 -> !xegpu.nbarrier

      %barrier_nth_iter = arith.muli %sg_K, %barrier_distance : index

      // K loop advances in 32 steps
      %k_loop_result:21 = scf.for %k = %c0 to %K step %sg_K iter_args (
          %A_tile_0 = %A_sg_init_tile_0,

          %B_tile_0 = %B_sg_init_tile_0,
          %B_tile_1 = %B_sg_init_tile_1,

          %c_val_0_0 = %c_init_val_0_0,
          %c_val_0_1 = %c_init_val_0_1,
          %c_val_0_2 = %c_init_val_0_2,
          %c_val_0_3 = %c_init_val_0_3,
          %c_val_1_0 = %c_init_val_1_0,
          %c_val_1_1 = %c_init_val_1_1,
          %c_val_1_2 = %c_init_val_1_2,
          %c_val_1_3 = %c_init_val_1_3,
          %c_val_2_0 = %c_init_val_2_0,
          %c_val_2_1 = %c_init_val_2_1,
          %c_val_2_2 = %c_init_val_2_2,
          %c_val_2_3 = %c_init_val_2_3,
          %c_val_3_0 = %c_init_val_3_0,
          %c_val_3_1 = %c_init_val_3_1,
          %c_val_3_2 = %c_init_val_3_2,
          %c_val_3_3 = %c_init_val_3_3,

          %A_prefetch_tile = %A_nth_prefetch_tile,
          %B_prefetch_tile = %B_nth_prefetch_tile
          ) ->
          (!xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
          !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
          !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
          vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,
          !xegpu.tensor_desc<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
          )
          {
        // all SGs must arrive here first
        %every_nth_iter = arith.remui %k, %barrier_nth_iter : index
        %every_nth_iter_i32 = arith.index_cast %every_nth_iter : index to i32
        %every_nth_iter_cond = arith.cmpi eq, %every_nth_iter_i32, %c0_i32 : i32
        scf.if %every_nth_iter_cond  {
          xegpu.nbarrier_arrive %nbarrier_y : !xegpu.nbarrier
          xegpu.nbarrier_arrive %nbarrier_x : !xegpu.nbarrier
        }
        // load A tiles
        %a_val = xegpu.load_nd %A_tile_0 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>> -> vector<2x32x16xf16>
        %a_val_0 = vector.extract %a_val [0] : vector<32x16xf16> from vector<2x32x16xf16>
        %a_val_1 = vector.extract %a_val [1] : vector<32x16xf16> from vector<2x32x16xf16>

        // load B tiles
        %b_val_arr_0 = xegpu.load_nd %B_tile_0 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>> -> vector<2x16x16x2xf16>
        %b_val_arr_1 = xegpu.load_nd %B_tile_1 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>> -> vector<2x16x16x2xf16>

        %b_val_0 = vector.extract %b_val_arr_0 [0] : vector<16x16x2xf16> from vector<2x16x16x2xf16>
        %b_val_1 = vector.extract %b_val_arr_0 [1] : vector<16x16x2xf16> from vector<2x16x16x2xf16>
        %b_val_2 = vector.extract %b_val_arr_1 [0] : vector<16x16x2xf16> from vector<2x16x16x2xf16>
        %b_val_3 = vector.extract %b_val_arr_1 [1] : vector<16x16x2xf16> from vector<2x16x16x2xf16>

        xegpu.compile_hint

        // prefetch A and B tiles
        xegpu.prefetch_nd %A_prefetch_tile {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>
        xegpu.prefetch_nd %B_prefetch_tile {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>

        //
        xegpu.compile_hint

        // advance A and B prefetch tiles
        %next_A_prefetch_tile = xegpu.update_nd_offset %A_prefetch_tile, [%c0, %sg_K] : !xegpu.tensor_desc<8x32xf16>
        %next_B_prefetch_tile = xegpu.update_nd_offset %B_prefetch_tile, [%sg_K, %c0] : !xegpu.tensor_desc<8x32xf16>
        // advance A and B tiles
        %next_A_tile_0 = xegpu.update_nd_offset %A_tile_0, [%c0, %sg_K] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>
        // %next_A_tile_1 = xegpu.update_nd_offset %A_tile_1, [%c0, %c32] : !xegpu.tensor_desc<32x16xf16>

        %next_B_tile_0 = xegpu.update_nd_offset %B_tile_0, [%sg_K, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>
        %next_B_tile_1 = xegpu.update_nd_offset %B_tile_1, [%sg_K, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>
        // %next_B_tile_2 = xegpu.update_nd_offset %B_tile_2, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16>
        // %next_B_tile_3 = xegpu.update_nd_offset %B_tile_3, [%c32, %c0] : !xegpu.tensor_desc<32x16xf16>

        xegpu.compile_hint
        %a_val_0_flat = vector.shape_cast %a_val_0 : vector<32x16xf16> to vector<512xf16>
        %a_val_1_flat = vector.shape_cast %a_val_1 : vector<32x16xf16> to vector<512xf16>
        %a_val_0_0_flat = vector.extract_strided_slice %a_val_0_flat { offsets = [0], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_0_0 = vector.shape_cast %a_val_0_0_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_1_0_flat = vector.extract_strided_slice %a_val_0_flat { offsets = [128], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_1_0 = vector.shape_cast %a_val_1_0_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_2_0_flat = vector.extract_strided_slice  %a_val_0_flat { offsets = [256], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_2_0 = vector.shape_cast %a_val_2_0_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_3_0_flat = vector.extract_strided_slice %a_val_0_flat { offsets = [384], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_3_0 = vector.shape_cast %a_val_3_0_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_0_1_flat = vector.extract_strided_slice %a_val_1_flat { offsets = [0], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_0_1 = vector.shape_cast %a_val_0_1_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_1_1_flat = vector.extract_strided_slice %a_val_1_flat {offsets = [128], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_1_1 = vector.shape_cast %a_val_1_1_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_2_1_flat = vector.extract_strided_slice %a_val_1_flat { offsets = [256], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_2_1 = vector.shape_cast %a_val_2_1_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_3_1_flat = vector.extract_strided_slice %a_val_1_flat { offsets = [384], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_3_1 = vector.shape_cast %a_val_3_1_flat : vector<128xf16> to vector<8x16xf16>


        %b_val_0_flat = vector.shape_cast %b_val_0 : vector<16x16x2xf16> to vector<512xf16>
        %b_val_1_flat = vector.shape_cast %b_val_1 : vector<16x16x2xf16> to vector<512xf16>
        %b_val_2_flat = vector.shape_cast %b_val_2 : vector<16x16x2xf16> to vector<512xf16>
        %b_val_3_flat = vector.shape_cast %b_val_3 : vector<16x16x2xf16> to vector<512xf16>
        %b_val_0_0_flat = vector.extract_strided_slice %b_val_0_flat { offsets = [0], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_0_0 = vector.shape_cast %b_val_0_0_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_1_0_flat = vector.extract_strided_slice %b_val_0_flat { offsets = [256], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_1_0 = vector.shape_cast %b_val_1_0_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_0_1_flat = vector.extract_strided_slice %b_val_1_flat { offsets = [0], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_0_1 = vector.shape_cast %b_val_0_1_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_1_1_flat = vector.extract_strided_slice %b_val_1_flat { offsets = [256], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_1_1 = vector.shape_cast %b_val_1_1_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_0_2_flat = vector.extract_strided_slice %b_val_2_flat { offsets = [0], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_0_2 = vector.shape_cast %b_val_0_2_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_1_2_flat = vector.extract_strided_slice %b_val_2_flat { offsets = [256], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_1_2 = vector.shape_cast %b_val_1_2_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_0_3_flat = vector.extract_strided_slice  %b_val_3_flat { offsets = [0], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_0_3 = vector.shape_cast %b_val_0_3_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_1_3_flat = vector.extract_strided_slice %b_val_3_flat {offsets = [256], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_1_3 = vector.shape_cast %b_val_1_3_flat : vector<256xf16> to vector<8x16x2xf16>

        // do DPAS
        xegpu.compile_hint
        %new_c_val_0_0_temp = xegpu.dpas %a_val_0_0, %b_val_0_0, %c_val_0_0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_0 = xegpu.dpas %a_val_0_1, %b_val_1_0, %new_c_val_0_0_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_0_temp = xegpu.dpas %a_val_1_0, %b_val_0_0, %c_val_1_0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_0 = xegpu.dpas %a_val_1_1, %b_val_1_0, %new_c_val_1_0_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_0_temp = xegpu.dpas %a_val_2_0, %b_val_0_0, %c_val_2_0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_0 = xegpu.dpas %a_val_2_1, %b_val_1_0, %new_c_val_2_0_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_0_temp = xegpu.dpas %a_val_3_0, %b_val_0_0, %c_val_3_0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_0 = xegpu.dpas %a_val_3_1, %b_val_1_0, %new_c_val_3_0_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        %new_c_val_0_1_temp = xegpu.dpas %a_val_0_0, %b_val_0_1, %c_val_0_1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_1 = xegpu.dpas %a_val_0_1, %b_val_1_1, %new_c_val_0_1_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_1_temp = xegpu.dpas %a_val_1_0, %b_val_0_1, %c_val_1_1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_1 = xegpu.dpas %a_val_1_1, %b_val_1_1, %new_c_val_1_1_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_1_temp = xegpu.dpas %a_val_2_0, %b_val_0_1, %c_val_2_1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_1 = xegpu.dpas %a_val_2_1, %b_val_1_1, %new_c_val_2_1_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_1_temp = xegpu.dpas %a_val_3_0, %b_val_0_1, %c_val_3_1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_1 = xegpu.dpas %a_val_3_1, %b_val_1_1, %new_c_val_3_1_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        %new_c_val_0_2_temp = xegpu.dpas %a_val_0_0, %b_val_0_2, %c_val_0_2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_2 = xegpu.dpas %a_val_0_1, %b_val_1_2, %new_c_val_0_2_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_2_temp = xegpu.dpas %a_val_1_0, %b_val_0_2, %c_val_1_2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_2 = xegpu.dpas %a_val_1_1, %b_val_1_2, %new_c_val_1_2_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_2_temp = xegpu.dpas %a_val_2_0, %b_val_0_2, %c_val_2_2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_2 = xegpu.dpas %a_val_2_1, %b_val_1_2, %new_c_val_2_2_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_2_temp = xegpu.dpas %a_val_3_0, %b_val_0_2, %c_val_3_2 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_2 = xegpu.dpas %a_val_3_1, %b_val_1_2, %new_c_val_3_2_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        %new_c_val_0_3_temp = xegpu.dpas %a_val_0_0, %b_val_0_3, %c_val_0_3 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_3 = xegpu.dpas %a_val_0_1, %b_val_1_3, %new_c_val_0_3_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_3_temp = xegpu.dpas %a_val_1_0, %b_val_0_3, %c_val_1_3 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_3 = xegpu.dpas %a_val_1_1, %b_val_1_3, %new_c_val_1_3_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_3_temp = xegpu.dpas %a_val_2_0, %b_val_0_3, %c_val_2_3 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_3 = xegpu.dpas %a_val_2_1, %b_val_1_3, %new_c_val_2_3_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_3_temp = xegpu.dpas %a_val_3_0, %b_val_0_3, %c_val_3_3 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        xegpu.compile_hint
        %new_c_val_3_3 = xegpu.dpas %a_val_3_1, %b_val_1_3, %new_c_val_3_3_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        xegpu.compile_hint
        //  barrier wait
        //  barrier wait
        scf.if %every_nth_iter_cond {
          xegpu.nbarrier_wait %nbarrier_y : !xegpu.nbarrier
          xegpu.nbarrier_wait %nbarrier_x : !xegpu.nbarrier
        }

        scf.yield %next_A_tile_0, %next_B_tile_0, %next_B_tile_1,
                  %new_c_val_0_0, %new_c_val_0_1, %new_c_val_0_2, %new_c_val_0_3, %new_c_val_1_0, %new_c_val_1_1, %new_c_val_1_2, %new_c_val_1_3, %new_c_val_2_0, %new_c_val_2_1, %new_c_val_2_2, %new_c_val_2_3, %new_c_val_3_0, %new_c_val_3_1, %new_c_val_3_2, %new_c_val_3_3,
                  %next_A_prefetch_tile, %next_B_prefetch_tile
                  : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
                  !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
                  !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
                  vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,
                  !xegpu.tensor_desc<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      }

      // trunc all DPAS output tiles to f16
      %c_result_0_0_f16 = arith.truncf %k_loop_result#3 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_0_1_f16 = arith.truncf %k_loop_result#4 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_0_2_f16 = arith.truncf %k_loop_result#5 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_0_3_f16 = arith.truncf %k_loop_result#6 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_1_0_f16 = arith.truncf %k_loop_result#7 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_1_1_f16 = arith.truncf %k_loop_result#8 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_1_2_f16 = arith.truncf %k_loop_result#9 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_1_3_f16 = arith.truncf %k_loop_result#10 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_2_0_f16 = arith.truncf %k_loop_result#11 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_2_1_f16 = arith.truncf %k_loop_result#12 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_2_2_f16 = arith.truncf %k_loop_result#13 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_2_3_f16 = arith.truncf %k_loop_result#14 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_3_0_f16 = arith.truncf %k_loop_result#15 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_3_1_f16 = arith.truncf %k_loop_result#16 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_3_2_f16 = arith.truncf %k_loop_result#17 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_3_3_f16 = arith.truncf %k_loop_result#18 : vector<8x16xf32> to vector<8x16xf16>

      // each SG needs to store the result of K loop into a 32x64 tile in C matrix. This is organized in 8x16 DPAS tiles
      // in the layout of 4x4x8x16. The max store size HW supoprt in f16 is 8x32. So we combine two 8x16 DPAS tiles
      // horizontally using vector.shuffle to get the required store size. The store layout then will 4x2x8x32 i.e.
      // we have 8 stores of size 8x32 in the layout 4x2.

      %c_result_8x32_0_0_t1 = vector.shuffle %c_result_0_0_f16, %c_result_0_1_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_0_0_t2 = vector.shape_cast %c_result_8x32_0_0_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_0_0 = vector.shape_cast %c_result_8x32_0_0_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_00 = xegpu.create_nd_tdesc %C[%C_sg_tile_offset_x, %C_sg_tile_offset_y], [%M, %N], [%N, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_0_0, %c_sg_tile_00 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint

      %c_result_8x32_0_1_t1 = vector.shuffle %c_result_0_2_f16, %c_result_0_3_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_0_1_t2 = vector.shape_cast %c_result_8x32_0_1_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_0_1 = vector.shape_cast %c_result_8x32_0_1_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_01 = xegpu.update_nd_offset %c_sg_tile_00, [%c0, %c32]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_0_1, %c_sg_tile_01 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint

      %c_result_8x32_1_0_t1 = vector.shuffle %c_result_1_0_f16, %c_result_1_1_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_1_0_t2 = vector.shape_cast %c_result_8x32_1_0_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_1_0 = vector.shape_cast %c_result_8x32_1_0_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_10 = xegpu.update_nd_offset %c_sg_tile_00, [%c8, %c0]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_1_0, %c_sg_tile_10 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint


      %c_result_8x32_1_1_t1 = vector.shuffle %c_result_1_2_f16, %c_result_1_3_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_1_1_t2 = vector.shape_cast %c_result_8x32_1_1_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_1_1 = vector.shape_cast %c_result_8x32_1_1_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_11 = xegpu.update_nd_offset %c_sg_tile_01, [%c8, %c0]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_1_1, %c_sg_tile_11 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint

      %c_result_8x32_2_0_t1 = vector.shuffle %c_result_2_0_f16, %c_result_2_1_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_2_0_t2 = vector.shape_cast %c_result_8x32_2_0_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_2_0 = vector.shape_cast %c_result_8x32_2_0_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_20 = xegpu.update_nd_offset %c_sg_tile_10, [%c8, %c0]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_2_0, %c_sg_tile_20 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint

      %c_result_8x32_2_1_t1 = vector.shuffle %c_result_2_2_f16, %c_result_2_3_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_2_1_t2 = vector.shape_cast %c_result_8x32_2_1_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_2_1 = vector.shape_cast %c_result_8x32_2_1_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_21 = xegpu.update_nd_offset %c_sg_tile_11, [%c8, %c0]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_2_1, %c_sg_tile_21 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint

      %c_result_8x32_3_0_t1 = vector.shuffle %c_result_3_0_f16, %c_result_3_1_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_3_0_t2 = vector.shape_cast %c_result_8x32_3_0_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_3_0 = vector.shape_cast %c_result_8x32_3_0_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_30 = xegpu.update_nd_offset %c_sg_tile_20, [%c8, %c0]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_3_0, %c_sg_tile_30 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint

      %c_result_8x32_3_1_t1 = vector.shuffle %c_result_3_2_f16, %c_result_3_3_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_3_1_t2 = vector.shape_cast %c_result_8x32_3_1_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_3_1 = vector.shape_cast %c_result_8x32_3_1_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_31 = xegpu.update_nd_offset %c_sg_tile_21, [%c8, %c0]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_3_1, %c_sg_tile_31 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>

      gpu.return
    }

    gpu.func @gemm_kernel_wgX_256_wgY_128_Kblock_32_sgX_32_sgY_32_split_barrier(%A: memref<?x?xf16>, %B: memref<?x?xf16>, %C: memref<?x?xf16>, %M: index, %K: index, %N: index,  %wg_M: index, %wg_K: index, %wg_N: index, %sg_M: index, %sg_K: index, %sg_N: index, %sg_X: index, %sg_Y: index, %prefetch_distance : index, %barrier_distance : index) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // constants
      %c256 = arith.constant 256 : index
      %c512 = arith.constant 512 : index
      %c128 = arith.constant 128 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c48 = arith.constant 48 : index
      %c16 = arith.constant 16 : index
      %c24 = arith.constant 24 : index
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      // get IDs
      %wg_id_x = gpu.block_id x
      %wg_id_y = gpu.block_id y
      // %sg_id = gpu.subgroup_id : index

      // each C wg tile is 256x256 and 32 SGs update it in 8x4 layout
      // C sg tile size is 32x64
      // SG layout for one C tile update
      // |0|1|2|3|
      // |4|5|6|7|
      // .........
      // |28|29|30|31|
      // --> y means cols
      // |
      // V x means rows

      // get unique sg ID in global context
      %global_sg_id_x = gpu.global_id x
      %global_sg_id_y = gpu.global_id y
      %local_sg_id_x = arith.remui %global_sg_id_x, %sg_X : index
      %local_sg_id_y = arith.remui %global_sg_id_y, %sg_Y : index

      // compute SG C tile offsets in x and y dims
      %C_sg_tile_offset_x = arith.muli %global_sg_id_x, %sg_M : index
      %C_sg_tile_offset_y = arith.muli %global_sg_id_y, %sg_N : index

      // each SG needs to do the follwoing compute to update its 32x32 sub tile
      // (32xK)x(Kx32)=(32x32)
      // DPAS size is (8x16)x(16x16)=(8x16)
      // K loop adavances in steps of 32, so inside the compute is (32x32)x(32x32) = (32x32)
      // So we need to (4x2) A tiles of size (8x16) and (2x2) B tiles of size (16x16)
      // tiled compute for a SG is (4x2x8x16)x(2x2x16x16)=(4x2x8x16)
      // this will require 16 DPAS ops (4x2x2) inside the K loop

      // WG tiles offsets for A, B and C
      %A_wg_tile_offset_x = arith.muli %wg_id_x, %wg_M : index
      %A_wg_tile_offset_y = arith.muli %wg_id_y, %wg_K : index

      %B_wg_tile_offset_x = arith.muli %wg_id_x, %wg_K : index
      %B_wg_tile_offset_y = arith.muli %wg_id_y, %wg_N : index


      %C_wg_tile_offset_x = arith.muli %wg_id_x, %wg_M : index
      %C_wg_tile_offset_y = arith.muli %wg_id_y, %wg_N : index

      // Linearized local SG ID
      %local_sg_id_temp = arith.muli %local_sg_id_x, %sg_Y : index
      %local_sg_id = arith.addi %local_sg_id_temp, %local_sg_id_y : index

      // prefetching A and B slice within the 256x256 WG tile
      //
      // prefetch the entire 256x32 slice of A WG tile, this means each subgroups needs to prefetch 8x32 slice
      // each 1x4 row of SGs do a colloborative prefetch of 8x32 slice of the 32x32 tile
      // SG 0 -> slice 0 |
      // SG 1 -> slice 1 |
      // SG 2 -> slice 2  > SG 0,1,2,3 share data prefetch from the top 32x32 tile.
      // SG 3 -> slice 3 |
      // SG 4 -> slice 4
      // ....
      // SG 31 -> slice 31

      // Prefetch the entire A (wg_M x sg_K) and B (sg_K x wg_N) slices of the WG tile needed for the one iteration of the K loop

      %A_sg_prefetch_offset_x_temp = arith.muli %local_sg_id, %c8 : index
      %A_sg_prefetch_offset_x = arith.addi %A_sg_prefetch_offset_x_temp, %A_wg_tile_offset_x : index
      // create A preftech tiles and prefetch
      // Inside a scf.for loop prefetch A tiles, and advance the prefetch tile offset, yielding the next tile to prefetch
      // Create the nd_descriptor for the A tile
      %A_sg_prefetch_tile_iter0 = xegpu.create_nd_tdesc %A[%A_sg_prefetch_offset_x, %c0], [%M, %K], [%K, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<8x32xf16>
      // %A_nth_prefetch_tile = scf.if %prefetch_distance {
      //   scf.for %i = %c0 to %prefetch_distance step %c1 iter_args(%current_tile = %A_sg_prefetch_tile_iter0) -> !xegpu.tensor_desc<8x32xf16> {
      //     xegpu.prefetch_nd %current_tile {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>
      //     %next_tile = xegpu.update_nd_offset %current_tile, [%c0, %sg_K] : !xegpu.tensor_desc<8x32xf16>
      //     scf.yield %next_tile : !xegpu.tensor_desc<8x32xf16>
      //   }
      // } else { // if prefetch_distance is 0, we do not prefetch
      //   %A_sg_prefetch_tile_iter0
      // }

      // Stage 1 to prefetch_distance
      %A_nth_prefetch_tile = scf.for %i = %c0 to %prefetch_distance step %c1 iter_args(%current_tile = %A_sg_prefetch_tile_iter0) -> !xegpu.tensor_desc<8x32xf16> {
        xegpu.prefetch_nd %current_tile {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>
        %next_tile = xegpu.update_nd_offset %current_tile, [%c0, %sg_K] : !xegpu.tensor_desc<8x32xf16>
        scf.yield %next_tile : !xegpu.tensor_desc<8x32xf16>
      }


      // prefetch the entire 32x128 slice of B WG tile, we still use the prefetch size 8x16.
      // SGs have 8x4 layout. In this case 8 subgroups must do a colloborative  prefetch of 32x32 tile.
      // this because the B tile arrangement within the 32x128 slice is as follows
      // 32x32 | 32x32 | 32x32 | 32x32
      // in terms of 8x32 slices the arrangement is,
      // 8x16 | 8x16 || 8x16 | 8x16 || 8x16 | 8x16 || 8x16 | 8x16
      // 8x16 | 8x16 || 8x16 | 8x16 || 8x16 | 8x16 || 8x16 | 8x16
      // 8x16 | 8x16 || 8x16 | 8x16 || 8x16 | 8x16 || 8x16 | 8x16
      // 8x16 | 8x16 || 8x16 | 8x16 || 8x16 | 8x16 || 8x16 | 8x16
      // So SGs 0,1,2,3,....31 prefetch in following fashion
      // | 0  | 16||  1 | 17 || 2  | 18 || 3 | 19 |
      // | 4  | 20||  5 | 21 || 6  | 22 || 7 | 23 |
      // | 8  | 24||  9 | 25 || 10 | 26 || 11| 27 |
      // | 12 | 28|| 13 | 29 || 14 | 30 || 15| 31 |
      // For example, SGs 0,4,8,12,16,20,24,28 share the data in left 32x64 tile of B slice.

      // calculate the x offsets and y offsets within the 32x128 slice
      %B_sg_prefetch_offset_x_temp0 = arith.remui %local_sg_id_x, %sg_Y : index
      %B_sg_prefetch_offset_x = arith.muli %B_sg_prefetch_offset_x_temp0, %sg_X : index
      %B_sg_prefetch_offset_y_temp0 = arith.muli %local_sg_id_y, %sg_N : index
      %B_sg_prefetch_offset_y_temp1 = arith.divui %local_sg_id_x, %sg_Y : index
      %B_sg_prefetch_offset_y_temp2 = arith.muli %B_sg_prefetch_offset_y_temp1, %sg_M : index
      %B_sg_prefetch_offset_y_temp3 = arith.addi %B_sg_prefetch_offset_y_temp0, %B_sg_prefetch_offset_y_temp2 : index
      %B_sg_prefetch_offset_y = arith.addi %B_wg_tile_offset_y, %B_sg_prefetch_offset_y_temp3 : index

      // create B prefetch tiles and prefetch
      %B_sg_prefetch_tile_iter0 = xegpu.create_nd_tdesc %B[%B_sg_prefetch_offset_x, %B_sg_prefetch_offset_y], [%K, %N], [%N, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<8x16xf16>

      // Prefetch B tiles within a scf.for loop
      %B_nth_prefetch_tile = scf.for %i = %c0 to %prefetch_distance step %c1 iter_args(%current_tile = %B_sg_prefetch_tile_iter0) -> !xegpu.tensor_desc<8x16xf16> {
        xegpu.prefetch_nd %current_tile {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x16xf16>
        %next_tile = xegpu.update_nd_offset %current_tile, [%sg_K, %c0] : !xegpu.tensor_desc<8x16xf16>
        scf.yield %next_tile : !xegpu.tensor_desc<8x16xf16>
      }

      // two 32x16 = 32x32 A tiles from 256x32 WG slice
      %A_sg_init_tile_0 = xegpu.create_nd_tdesc %A[%C_sg_tile_offset_x, %c0], [%M, %K], [%K, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>

      // two 32x16 = 32x32 B tiles from 256x32 WG slice
      %B_sg_init_tile_0 = xegpu.create_nd_tdesc %B[%c0, %C_sg_tile_offset_y], [%K, %N], [%N, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>


      // init 16 C tiles of size 8x16 each is initialized to 0.0 assuming a zero C matrix
      %zero_vec = arith.constant dense<0.0> : vector<128xf32>
      %c_init_val_0_0 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_0_1 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>

      %c_init_val_1_0 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_1_1 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>

      %c_init_val_2_0 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_2_1 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>

      %c_init_val_3_0 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_3_1 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>



      xegpu.alloc_nbarrier 16
      %nbarrier_id = arith.constant 1 : i8
      %num_threads = arith.constant 32 : i8
      %nbarrier = xegpu.init_nbarrier %nbarrier_id, %num_threads : i8, i8 -> !xegpu.nbarrier
      %barrier_nth_iter = arith.muli %sg_K, %barrier_distance : index
      // K loop advances in 32 steps
      %k_loop_result:12 = scf.for %k = %c0 to %K step %sg_K iter_args (
          %A_tile_0 = %A_sg_init_tile_0,
          %B_tile_0 = %B_sg_init_tile_0,

          %c_val_0_0 = %c_init_val_0_0,
          %c_val_0_1 = %c_init_val_0_1,
          %c_val_1_0 = %c_init_val_1_0,
          %c_val_1_1 = %c_init_val_1_1,
          %c_val_2_0 = %c_init_val_2_0,
          %c_val_2_1 = %c_init_val_2_1,
          %c_val_3_0 = %c_init_val_3_0,
          %c_val_3_1 = %c_init_val_3_1,

          %A_prefetch_tile = %A_nth_prefetch_tile,
          %B_prefetch_tile = %B_nth_prefetch_tile
          ) ->
          (!xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
          !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
          vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,
          !xegpu.tensor_desc<8x32xf16>, !xegpu.tensor_desc<8x16xf16>
          )
          {
        // all SGs must arrive here first
        %every_nth_iter = arith.remui %k, %barrier_nth_iter : index
        %every_nth_iter_i32 = arith.index_cast %every_nth_iter : index to i32
        %every_nth_iter_cond = arith.cmpi eq, %every_nth_iter_i32, %c0_i32 : i32
        scf.if %every_nth_iter_cond  {
          xegpu.nbarrier_arrive %nbarrier : !xegpu.nbarrier
        }
        // load A tiles
        %a_val = xegpu.load_nd %A_tile_0 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>> -> vector<2x32x16xf16>
        %a_val_0 = vector.extract %a_val [0] : vector<32x16xf16> from vector<2x32x16xf16>
        %a_val_1 = vector.extract %a_val [1] : vector<32x16xf16> from vector<2x32x16xf16>

        // load B tiles
        %b_val_arr_0 = xegpu.load_nd %B_tile_0 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>> -> vector<2x16x16x2xf16>

        %b_val_0 = vector.extract %b_val_arr_0 [0] : vector<16x16x2xf16> from vector<2x16x16x2xf16>
        %b_val_1 = vector.extract %b_val_arr_0 [1] : vector<16x16x2xf16> from vector<2x16x16x2xf16>

        xegpu.compile_hint

        // prefetch A and B tiles
        xegpu.prefetch_nd %A_prefetch_tile {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>
        xegpu.prefetch_nd %B_prefetch_tile {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x16xf16>

        //
        xegpu.compile_hint

        // advance A and B prefetch tiles
        %next_A_prefetch_tile = xegpu.update_nd_offset %A_prefetch_tile, [%c0, %sg_K] : !xegpu.tensor_desc<8x32xf16>
        %next_B_prefetch_tile = xegpu.update_nd_offset %B_prefetch_tile, [%sg_K, %c0] : !xegpu.tensor_desc<8x16xf16>
        // advance A and B tiles
        %next_A_tile_0 = xegpu.update_nd_offset %A_tile_0, [%c0, %sg_K] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>

        %next_B_tile_0 = xegpu.update_nd_offset %B_tile_0, [%sg_K, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>

        xegpu.compile_hint
        %a_val_0_flat = vector.shape_cast %a_val_0 : vector<32x16xf16> to vector<512xf16>
        %a_val_1_flat = vector.shape_cast %a_val_1 : vector<32x16xf16> to vector<512xf16>
        %a_val_0_0_flat = vector.extract_strided_slice %a_val_0_flat { offsets = [0], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_0_0 = vector.shape_cast %a_val_0_0_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_1_0_flat = vector.extract_strided_slice %a_val_0_flat { offsets = [128], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_1_0 = vector.shape_cast %a_val_1_0_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_2_0_flat = vector.extract_strided_slice  %a_val_0_flat { offsets = [256], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_2_0 = vector.shape_cast %a_val_2_0_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_3_0_flat = vector.extract_strided_slice %a_val_0_flat { offsets = [384], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_3_0 = vector.shape_cast %a_val_3_0_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_0_1_flat = vector.extract_strided_slice %a_val_1_flat { offsets = [0], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_0_1 = vector.shape_cast %a_val_0_1_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_1_1_flat = vector.extract_strided_slice %a_val_1_flat {offsets = [128], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_1_1 = vector.shape_cast %a_val_1_1_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_2_1_flat = vector.extract_strided_slice %a_val_1_flat { offsets = [256], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_2_1 = vector.shape_cast %a_val_2_1_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_3_1_flat = vector.extract_strided_slice %a_val_1_flat { offsets = [384], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_3_1 = vector.shape_cast %a_val_3_1_flat : vector<128xf16> to vector<8x16xf16>


        %b_val_0_flat = vector.shape_cast %b_val_0 : vector<16x16x2xf16> to vector<512xf16>
        %b_val_1_flat = vector.shape_cast %b_val_1 : vector<16x16x2xf16> to vector<512xf16>

        %b_val_0_0_flat = vector.extract_strided_slice %b_val_0_flat { offsets = [0], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_0_0 = vector.shape_cast %b_val_0_0_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_1_0_flat = vector.extract_strided_slice %b_val_0_flat { offsets = [256], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_1_0 = vector.shape_cast %b_val_1_0_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_0_1_flat = vector.extract_strided_slice %b_val_1_flat { offsets = [0], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_0_1 = vector.shape_cast %b_val_0_1_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_1_1_flat = vector.extract_strided_slice %b_val_1_flat { offsets = [256], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_1_1 = vector.shape_cast %b_val_1_1_flat : vector<256xf16> to vector<8x16x2xf16>


        // do DPAS
        xegpu.compile_hint
        %new_c_val_0_0_temp = xegpu.dpas %a_val_0_0, %b_val_0_0, %c_val_0_0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_0 = xegpu.dpas %a_val_0_1, %b_val_1_0, %new_c_val_0_0_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_0_temp = xegpu.dpas %a_val_1_0, %b_val_0_0, %c_val_1_0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_0 = xegpu.dpas %a_val_1_1, %b_val_1_0, %new_c_val_1_0_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_0_temp = xegpu.dpas %a_val_2_0, %b_val_0_0, %c_val_2_0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_0 = xegpu.dpas %a_val_2_1, %b_val_1_0, %new_c_val_2_0_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_0_temp = xegpu.dpas %a_val_3_0, %b_val_0_0, %c_val_3_0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_0 = xegpu.dpas %a_val_3_1, %b_val_1_0, %new_c_val_3_0_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        %new_c_val_0_1_temp = xegpu.dpas %a_val_0_0, %b_val_0_1, %c_val_0_1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_1 = xegpu.dpas %a_val_0_1, %b_val_1_1, %new_c_val_0_1_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_1_temp = xegpu.dpas %a_val_1_0, %b_val_0_1, %c_val_1_1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_1 = xegpu.dpas %a_val_1_1, %b_val_1_1, %new_c_val_1_1_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_1_temp = xegpu.dpas %a_val_2_0, %b_val_0_1, %c_val_2_1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_1 = xegpu.dpas %a_val_2_1, %b_val_1_1, %new_c_val_2_1_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_1_temp = xegpu.dpas %a_val_3_0, %b_val_0_1, %c_val_3_1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_1 = xegpu.dpas %a_val_3_1, %b_val_1_1, %new_c_val_3_1_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        xegpu.compile_hint
        //  barrier wait
        scf.if %every_nth_iter_cond {
          xegpu.nbarrier_wait %nbarrier : !xegpu.nbarrier
        }

        scf.yield %next_A_tile_0, %next_B_tile_0,
                  %new_c_val_0_0, %new_c_val_0_1, %new_c_val_1_0, %new_c_val_1_1, %new_c_val_2_0, %new_c_val_2_1, %new_c_val_3_0, %new_c_val_3_1,
                  %next_A_prefetch_tile, %next_B_prefetch_tile
                  : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
                  !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
                  vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,
                  !xegpu.tensor_desc<8x32xf16>, !xegpu.tensor_desc<8x16xf16>
      }

      // trunc all DPAS output tiles to f16
      %c_result_0_0_f16 = arith.truncf %k_loop_result#2 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_0_1_f16 = arith.truncf %k_loop_result#3 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_1_0_f16 = arith.truncf %k_loop_result#4 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_1_1_f16 = arith.truncf %k_loop_result#5 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_2_0_f16 = arith.truncf %k_loop_result#6 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_2_1_f16 = arith.truncf %k_loop_result#7 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_3_0_f16 = arith.truncf %k_loop_result#8 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_3_1_f16 = arith.truncf %k_loop_result#9 : vector<8x16xf32> to vector<8x16xf16>

      // each SG needs to store the result of K loop into a 32x32 tile in C matrix. This is organized in 8x16 DPAS tiles
      // in the layout of 4x2x8x16. The max store size HW supoprt in f16 is 8x32. So we combine two 8x16 DPAS tiles
      // horizontally using vector.shuffle to get the required store size. The store layout then will 4x8x32 i.e.
      // we have 4 stores of size 8x32 in the layout 4x1.

      %c_result_8x32_0_0_t1 = vector.shuffle %c_result_0_0_f16, %c_result_0_1_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_0_0_t2 = vector.shape_cast %c_result_8x32_0_0_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_0_0 = vector.shape_cast %c_result_8x32_0_0_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_00 = xegpu.create_nd_tdesc %C[%C_sg_tile_offset_x, %C_sg_tile_offset_y], [%M, %N], [%N, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_0_0, %c_sg_tile_00 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint


      %c_result_8x32_1_0_t1 = vector.shuffle %c_result_1_0_f16, %c_result_1_1_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_1_0_t2 = vector.shape_cast %c_result_8x32_1_0_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_1_0 = vector.shape_cast %c_result_8x32_1_0_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_10 = xegpu.update_nd_offset %c_sg_tile_00, [%c8, %c0]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_1_0, %c_sg_tile_10 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint


      %c_result_8x32_2_0_t1 = vector.shuffle %c_result_2_0_f16, %c_result_2_1_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_2_0_t2 = vector.shape_cast %c_result_8x32_2_0_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_2_0 = vector.shape_cast %c_result_8x32_2_0_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_20 = xegpu.update_nd_offset %c_sg_tile_10, [%c8, %c0]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_2_0, %c_sg_tile_20 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint


      %c_result_8x32_3_0_t1 = vector.shuffle %c_result_3_0_f16, %c_result_3_1_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_3_0_t2 = vector.shape_cast %c_result_8x32_3_0_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_3_0 = vector.shape_cast %c_result_8x32_3_0_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_30 = xegpu.update_nd_offset %c_sg_tile_20, [%c8, %c0]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_3_0, %c_sg_tile_30 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint

      gpu.return
    }

    gpu.func @gemm_kernel_wgX_256_wgY_128_Kblock_32_sgX_32_sgY_32_named_barrier(%A: memref<?x?xf16>, %B: memref<?x?xf16>, %C: memref<?x?xf16>, %M: index, %K: index, %N: index,  %wg_M: index, %wg_K: index, %wg_N: index, %sg_M: index, %sg_K: index, %sg_N: index, %sg_X: index, %sg_Y: index, %prefetch_distance : index, %barrier_distance : index) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // constants
      %c256 = arith.constant 256 : index
      %c512 = arith.constant 512 : index
      %c128 = arith.constant 128 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c48 = arith.constant 48 : index
      %c16 = arith.constant 16 : index
      %c24 = arith.constant 24 : index
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      // get IDs
      %wg_id_x = gpu.block_id x
      %wg_id_y = gpu.block_id y
      // %sg_id = gpu.subgroup_id : index

      // each C wg tile is 256x256 and 32 SGs update it in 8x4 layout
      // C sg tile size is 32x64
      // SG layout for one C tile update
      // |0|1|2|3|
      // |4|5|6|7|
      // .........
      // |28|29|30|31|
      // --> y means cols
      // |
      // V x means rows

      // get unique sg ID in global context
      %global_sg_id_x = gpu.global_id x
      %global_sg_id_y = gpu.global_id y
      %local_sg_id_x = arith.remui %global_sg_id_x, %sg_X : index
      %local_sg_id_y = arith.remui %global_sg_id_y, %sg_Y : index

      // compute SG C tile offsets in x and y dims
      %C_sg_tile_offset_x = arith.muli %global_sg_id_x, %sg_M : index
      %C_sg_tile_offset_y = arith.muli %global_sg_id_y, %sg_N : index

      // each SG needs to do the follwoing compute to update its 32x32 sub tile
      // (32xK)x(Kx32)=(32x32)
      // DPAS size is (8x16)x(16x16)=(8x16)
      // K loop adavances in steps of 32, so inside the compute is (32x32)x(32x32) = (32x32)
      // So we need to (4x2) A tiles of size (8x16) and (2x2) B tiles of size (16x16)
      // tiled compute for a SG is (4x2x8x16)x(2x2x16x16)=(4x2x8x16)
      // this will require 16 DPAS ops (4x2x2) inside the K loop

      // WG tiles offsets for A, B and C
      %A_wg_tile_offset_x = arith.muli %wg_id_x, %wg_M : index
      %A_wg_tile_offset_y = arith.muli %wg_id_y, %wg_K : index

      %B_wg_tile_offset_x = arith.muli %wg_id_x, %wg_K : index
      %B_wg_tile_offset_y = arith.muli %wg_id_y, %wg_N : index


      %C_wg_tile_offset_x = arith.muli %wg_id_x, %wg_M : index
      %C_wg_tile_offset_y = arith.muli %wg_id_y, %wg_N : index

      // Linearized local SG ID
      %local_sg_id_temp = arith.muli %local_sg_id_x, %sg_Y : index
      %local_sg_id = arith.addi %local_sg_id_temp, %local_sg_id_y : index

      // prefetching A and B slice within the 256x256 WG tile
      //
      // prefetch the entire 256x32 slice of A WG tile, this means each subgroups needs to prefetch 8x32 slice
      // each 1x4 row of SGs do a colloborative prefetch of 8x32 slice of the 32x32 tile
      // SG 0 -> slice 0 |
      // SG 1 -> slice 1 |
      // SG 2 -> slice 2  > SG 0,1,2,3 share data prefetch from the top 32x32 tile.
      // SG 3 -> slice 3 |
      // SG 4 -> slice 4
      // ....
      // SG 31 -> slice 31

      // Prefetch the entire A (wg_M x sg_K) and B (sg_K x wg_N) slices of the WG tile needed for the one iteration of the K loop

      %A_sg_prefetch_offset_x_temp = arith.muli %local_sg_id, %c8 : index
      %A_sg_prefetch_offset_x = arith.addi %A_sg_prefetch_offset_x_temp, %A_wg_tile_offset_x : index
      // create A preftech tiles and prefetch
      // Inside a scf.for loop prefetch A tiles, and advance the prefetch tile offset, yielding the next tile to prefetch
      // Create the nd_descriptor for the A tile
      %A_sg_prefetch_tile_iter0 = xegpu.create_nd_tdesc %A[%A_sg_prefetch_offset_x, %c0], [%M, %K], [%K, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<8x32xf16>
      // %A_nth_prefetch_tile = scf.if %prefetch_distance {
      //   scf.for %i = %c0 to %prefetch_distance step %c1 iter_args(%current_tile = %A_sg_prefetch_tile_iter0) -> !xegpu.tensor_desc<8x32xf16> {
      //     xegpu.prefetch_nd %current_tile {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>
      //     %next_tile = xegpu.update_nd_offset %current_tile, [%c0, %sg_K] : !xegpu.tensor_desc<8x32xf16>
      //     scf.yield %next_tile : !xegpu.tensor_desc<8x32xf16>
      //   }
      // } else { // if prefetch_distance is 0, we do not prefetch
      //   %A_sg_prefetch_tile_iter0
      // }

      // Stage 1 to prefetch_distance
      %A_nth_prefetch_tile = scf.for %i = %c0 to %prefetch_distance step %c1 iter_args(%current_tile = %A_sg_prefetch_tile_iter0) -> !xegpu.tensor_desc<8x32xf16> {
        xegpu.prefetch_nd %current_tile {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>
        %next_tile = xegpu.update_nd_offset %current_tile, [%c0, %sg_K] : !xegpu.tensor_desc<8x32xf16>
        scf.yield %next_tile : !xegpu.tensor_desc<8x32xf16>
      }


      // prefetch the entire 32x128 slice of B WG tile, we still use the prefetch size 8x16.
      // SGs have 8x4 layout. In this case 8 subgroups must do a colloborative  prefetch of 32x32 tile.
      // this because the B tile arrangement within the 32x128 slice is as follows
      // 32x32 | 32x32 | 32x32 | 32x32
      // in terms of 8x32 slices the arrangement is,
      // 8x16 | 8x16 || 8x16 | 8x16 || 8x16 | 8x16 || 8x16 | 8x16
      // 8x16 | 8x16 || 8x16 | 8x16 || 8x16 | 8x16 || 8x16 | 8x16
      // 8x16 | 8x16 || 8x16 | 8x16 || 8x16 | 8x16 || 8x16 | 8x16
      // 8x16 | 8x16 || 8x16 | 8x16 || 8x16 | 8x16 || 8x16 | 8x16
      // So SGs 0,1,2,3,....31 prefetch in following fashion
      // | 0  | 16||  1 | 17 || 2  | 18 || 3 | 19 |
      // | 4  | 20||  5 | 21 || 6  | 22 || 7 | 23 |
      // | 8  | 24||  9 | 25 || 10 | 26 || 11| 27 |
      // | 12 | 28|| 13 | 29 || 14 | 30 || 15| 31 |
      // For example, SGs 0,4,8,12,16,20,24,28 share the data in left 32x64 tile of B slice.

      // calculate the x offsets and y offsets within the 32x128 slice
      %B_sg_prefetch_offset_x_temp0 = arith.remui %local_sg_id_x, %sg_Y : index
      %B_sg_prefetch_offset_x = arith.muli %B_sg_prefetch_offset_x_temp0, %sg_X : index
      %B_sg_prefetch_offset_y_temp0 = arith.muli %local_sg_id_y, %sg_N : index
      %B_sg_prefetch_offset_y_temp1 = arith.divui %local_sg_id_x, %sg_Y : index
      %B_sg_prefetch_offset_y_temp2 = arith.muli %B_sg_prefetch_offset_y_temp1, %sg_M : index
      %B_sg_prefetch_offset_y_temp3 = arith.addi %B_sg_prefetch_offset_y_temp0, %B_sg_prefetch_offset_y_temp2 : index
      %B_sg_prefetch_offset_y = arith.addi %B_wg_tile_offset_y, %B_sg_prefetch_offset_y_temp3 : index

      // create B prefetch tiles and prefetch
      %B_sg_prefetch_tile_iter0 = xegpu.create_nd_tdesc %B[%B_sg_prefetch_offset_x, %B_sg_prefetch_offset_y], [%K, %N], [%N, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<8x16xf16>

      // Prefetch B tiles within a scf.for loop
      %B_nth_prefetch_tile = scf.for %i = %c0 to %prefetch_distance step %c1 iter_args(%current_tile = %B_sg_prefetch_tile_iter0) -> !xegpu.tensor_desc<8x16xf16> {
        xegpu.prefetch_nd %current_tile {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x16xf16>
        %next_tile = xegpu.update_nd_offset %current_tile, [%sg_K, %c0] : !xegpu.tensor_desc<8x16xf16>
        scf.yield %next_tile : !xegpu.tensor_desc<8x16xf16>
      }

      // two 32x16 = 32x32 A tiles from 256x32 WG slice
      %A_sg_init_tile_0 = xegpu.create_nd_tdesc %A[%C_sg_tile_offset_x, %c0], [%M, %K], [%K, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>

      // two 32x16 = 32x32 B tiles from 256x32 WG slice
      %B_sg_init_tile_0 = xegpu.create_nd_tdesc %B[%c0, %C_sg_tile_offset_y], [%K, %N], [%N, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>


      // init 16 C tiles of size 8x16 each is initialized to 0.0 assuming a zero C matrix
      %zero_vec = arith.constant dense<0.0> : vector<128xf32>
      %c_init_val_0_0 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_0_1 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>

      %c_init_val_1_0 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_1_1 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>

      %c_init_val_2_0 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_2_1 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>

      %c_init_val_3_0 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>
      %c_init_val_3_1 = vector.shape_cast %zero_vec : vector<128xf32> to vector<8x16xf32>

      // Multi nbarrier implementation,
      // one set nbarrier is used to sync subgroups with same sg_id_x (local_sg_id_x)
      // second set nbarrier us used to sync subgroups with same sg_id_y (local_sg_id_y)
      // In this case wg_size = 8,4 (sg_X = 8; sg_Y = 4)
      // So in Y-direction we need 4 nbarrier (to sync subgroups with same sg_id_y)
      // In X-direction we need 8 nbarrier (to sync subgroups with same sg_id_x)

      // %c_wg_size_x = arith.constant 8 : index
      // %c_wg_size_y = arith.constant 4 : index
      %num_nbarrier = arith.addi %sg_X, %sg_Y : index // 8+4=12
      xegpu.alloc_nbarrier 12 // = 12

      // xegpu.alloc_nbarrier 12

      // First set of nbarriers work across coloumns, we have 4 coloums of subgroups,
      // Hnece 4 nbrrier
      // Each nbarrier has 8 producers and consumers
      // nbarrier type is Producer_Consumer (https://gfxspecs.intel.com/Predator/Home/Index/57499)

      // %nbarrier_role = arith.constant 0 : i8
      // %nbarrier_threads_y = arith.constant 8 : i8
      %nbarrier_threads_y = arith.index_cast %sg_X : index to i8
      %nbarrier_id_y = arith.index_cast %local_sg_id_y : index to i8
      %nbarrier_y = xegpu.init_nbarrier %nbarrier_id_y, %nbarrier_threads_y : i8, i8 -> !xegpu.nbarrier

      // Second set of barriers work on across rows of subgroups,
      // we have 8 rows of subgroups. Hnece, 8 nbarrier
      // Each nbarrier has 4 producers and consumers
      // nbarrier type is Producer_Consumer (https://gfxspecs.intel.com/Predator/Home/Index/57499)

      // We already have 4 (=%c_wg_size_y=sg_Y) nbarriers with id 0-3,
      // Now the next set of barrier id would start from 4, hence,
      // %nbarrier_threads_x = arith.constant 4 : i8
      %nbarrier_threads_x = arith.index_cast %sg_Y : index to i8
      %index_nbarrier_id_x = arith.addi %sg_Y, %local_sg_id_x : index
      %nbarrier_id_x = arith.index_cast %index_nbarrier_id_x : index to i8
      %nbarrier_x = xegpu.init_nbarrier %nbarrier_id_x, %nbarrier_threads_x : i8, i8 -> !xegpu.nbarrier

      %barrier_nth_iter = arith.muli %sg_K, %barrier_distance : index
      // K loop advances in 32 steps
      %k_loop_result:12 = scf.for %k = %c0 to %K step %sg_K iter_args (
          %A_tile_0 = %A_sg_init_tile_0,
          %B_tile_0 = %B_sg_init_tile_0,

          %c_val_0_0 = %c_init_val_0_0,
          %c_val_0_1 = %c_init_val_0_1,
          %c_val_1_0 = %c_init_val_1_0,
          %c_val_1_1 = %c_init_val_1_1,
          %c_val_2_0 = %c_init_val_2_0,
          %c_val_2_1 = %c_init_val_2_1,
          %c_val_3_0 = %c_init_val_3_0,
          %c_val_3_1 = %c_init_val_3_1,

          %A_prefetch_tile = %A_nth_prefetch_tile,
          %B_prefetch_tile = %B_nth_prefetch_tile
          ) ->
          (!xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
          !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
          vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,
          !xegpu.tensor_desc<8x32xf16>, !xegpu.tensor_desc<8x16xf16>
          )
          {
        // all SGs must arrive here first
        %every_nth_iter = arith.remui %k, %barrier_nth_iter : index
        %every_nth_iter_i32 = arith.index_cast %every_nth_iter : index to i32
        %every_nth_iter_cond = arith.cmpi eq, %every_nth_iter_i32, %c0_i32 : i32
        scf.if %every_nth_iter_cond  {
          xegpu.nbarrier_arrive %nbarrier_y : !xegpu.nbarrier
          xegpu.nbarrier_arrive %nbarrier_x : !xegpu.nbarrier
        }
        // load A tiles
        %a_val = xegpu.load_nd %A_tile_0 {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>> -> vector<2x32x16xf16>
        %a_val_0 = vector.extract %a_val [0] : vector<32x16xf16> from vector<2x32x16xf16>
        %a_val_1 = vector.extract %a_val [1] : vector<32x16xf16> from vector<2x32x16xf16>

        // load B tiles
        %b_val_arr_0 = xegpu.load_nd %B_tile_0 {packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>> -> vector<2x16x16x2xf16>

        %b_val_0 = vector.extract %b_val_arr_0 [0] : vector<16x16x2xf16> from vector<2x16x16x2xf16>
        %b_val_1 = vector.extract %b_val_arr_0 [1] : vector<16x16x2xf16> from vector<2x16x16x2xf16>

        xegpu.compile_hint

        // prefetch A and B tiles
        xegpu.prefetch_nd %A_prefetch_tile {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x32xf16>
        xegpu.prefetch_nd %B_prefetch_tile {l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>} : !xegpu.tensor_desc<8x16xf16>

        //
        xegpu.compile_hint

        // advance A and B prefetch tiles
        %next_A_prefetch_tile = xegpu.update_nd_offset %A_prefetch_tile, [%c0, %sg_K] : !xegpu.tensor_desc<8x32xf16>
        %next_B_prefetch_tile = xegpu.update_nd_offset %B_prefetch_tile, [%sg_K, %c0] : !xegpu.tensor_desc<8x16xf16>
        // advance A and B tiles
        %next_A_tile_0 = xegpu.update_nd_offset %A_tile_0, [%c0, %sg_K] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>

        %next_B_tile_0 = xegpu.update_nd_offset %B_tile_0, [%sg_K, %c0] : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>

        xegpu.compile_hint
        %a_val_0_flat = vector.shape_cast %a_val_0 : vector<32x16xf16> to vector<512xf16>
        %a_val_1_flat = vector.shape_cast %a_val_1 : vector<32x16xf16> to vector<512xf16>
        %a_val_0_0_flat = vector.extract_strided_slice %a_val_0_flat { offsets = [0], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_0_0 = vector.shape_cast %a_val_0_0_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_1_0_flat = vector.extract_strided_slice %a_val_0_flat { offsets = [128], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_1_0 = vector.shape_cast %a_val_1_0_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_2_0_flat = vector.extract_strided_slice  %a_val_0_flat { offsets = [256], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_2_0 = vector.shape_cast %a_val_2_0_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_3_0_flat = vector.extract_strided_slice %a_val_0_flat { offsets = [384], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_3_0 = vector.shape_cast %a_val_3_0_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_0_1_flat = vector.extract_strided_slice %a_val_1_flat { offsets = [0], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_0_1 = vector.shape_cast %a_val_0_1_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_1_1_flat = vector.extract_strided_slice %a_val_1_flat {offsets = [128], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_1_1 = vector.shape_cast %a_val_1_1_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_2_1_flat = vector.extract_strided_slice %a_val_1_flat { offsets = [256], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_2_1 = vector.shape_cast %a_val_2_1_flat : vector<128xf16> to vector<8x16xf16>
        %a_val_3_1_flat = vector.extract_strided_slice %a_val_1_flat { offsets = [384], sizes = [128], strides = [1]} :
          vector<512xf16> to vector<128xf16>
        %a_val_3_1 = vector.shape_cast %a_val_3_1_flat : vector<128xf16> to vector<8x16xf16>


        %b_val_0_flat = vector.shape_cast %b_val_0 : vector<16x16x2xf16> to vector<512xf16>
        %b_val_1_flat = vector.shape_cast %b_val_1 : vector<16x16x2xf16> to vector<512xf16>

        %b_val_0_0_flat = vector.extract_strided_slice %b_val_0_flat { offsets = [0], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_0_0 = vector.shape_cast %b_val_0_0_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_1_0_flat = vector.extract_strided_slice %b_val_0_flat { offsets = [256], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_1_0 = vector.shape_cast %b_val_1_0_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_0_1_flat = vector.extract_strided_slice %b_val_1_flat { offsets = [0], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_0_1 = vector.shape_cast %b_val_0_1_flat : vector<256xf16> to vector<8x16x2xf16>
        %b_val_1_1_flat = vector.extract_strided_slice %b_val_1_flat { offsets = [256], sizes = [256], strides = [1]} :
          vector<512xf16> to vector<256xf16>
        %b_val_1_1 = vector.shape_cast %b_val_1_1_flat : vector<256xf16> to vector<8x16x2xf16>


        // do DPAS
        xegpu.compile_hint
        %new_c_val_0_0_temp = xegpu.dpas %a_val_0_0, %b_val_0_0, %c_val_0_0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_0 = xegpu.dpas %a_val_0_1, %b_val_1_0, %new_c_val_0_0_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_0_temp = xegpu.dpas %a_val_1_0, %b_val_0_0, %c_val_1_0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_0 = xegpu.dpas %a_val_1_1, %b_val_1_0, %new_c_val_1_0_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_0_temp = xegpu.dpas %a_val_2_0, %b_val_0_0, %c_val_2_0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_0 = xegpu.dpas %a_val_2_1, %b_val_1_0, %new_c_val_2_0_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_0_temp = xegpu.dpas %a_val_3_0, %b_val_0_0, %c_val_3_0 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_0 = xegpu.dpas %a_val_3_1, %b_val_1_0, %new_c_val_3_0_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        %new_c_val_0_1_temp = xegpu.dpas %a_val_0_0, %b_val_0_1, %c_val_0_1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_0_1 = xegpu.dpas %a_val_0_1, %b_val_1_1, %new_c_val_0_1_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_1_temp = xegpu.dpas %a_val_1_0, %b_val_0_1, %c_val_1_1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_1_1 = xegpu.dpas %a_val_1_1, %b_val_1_1, %new_c_val_1_1_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_1_temp = xegpu.dpas %a_val_2_0, %b_val_0_1, %c_val_2_1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_2_1 = xegpu.dpas %a_val_2_1, %b_val_1_1, %new_c_val_2_1_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_1_temp = xegpu.dpas %a_val_3_0, %b_val_0_1, %c_val_3_1 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %new_c_val_3_1 = xegpu.dpas %a_val_3_1, %b_val_1_1, %new_c_val_3_1_temp : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        xegpu.compile_hint
        //  barrier wait
        scf.if %every_nth_iter_cond {
          xegpu.nbarrier_wait %nbarrier_y : !xegpu.nbarrier
          xegpu.nbarrier_wait %nbarrier_x : !xegpu.nbarrier
        }

        scf.yield %next_A_tile_0, %next_B_tile_0,
                  %new_c_val_0_0, %new_c_val_0_1, %new_c_val_1_0, %new_c_val_1_1, %new_c_val_2_0, %new_c_val_2_1, %new_c_val_3_0, %new_c_val_3_1,
                  %next_A_prefetch_tile, %next_B_prefetch_tile
                  : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
                  !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2>>,
                  vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,vector<8x16xf32>,
                  !xegpu.tensor_desc<8x32xf16>, !xegpu.tensor_desc<8x16xf16>
      }

      // trunc all DPAS output tiles to f16
      %c_result_0_0_f16 = arith.truncf %k_loop_result#2 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_0_1_f16 = arith.truncf %k_loop_result#3 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_1_0_f16 = arith.truncf %k_loop_result#4 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_1_1_f16 = arith.truncf %k_loop_result#5 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_2_0_f16 = arith.truncf %k_loop_result#6 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_2_1_f16 = arith.truncf %k_loop_result#7 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_3_0_f16 = arith.truncf %k_loop_result#8 : vector<8x16xf32> to vector<8x16xf16>
      %c_result_3_1_f16 = arith.truncf %k_loop_result#9 : vector<8x16xf32> to vector<8x16xf16>

      // each SG needs to store the result of K loop into a 32x64 tile in C matrix. This is organized in 8x16 DPAS tiles
      // in the layout of 4x2x8x16. The max store size HW supoprt in f16 is 8x32. So we combine two 8x16 DPAS tiles
      // horizontally using vector.shuffle to get the required store size. The store layout then will 4x8x32 i.e.
      // we have 4 stores of size 8x32 in the layout 4x1.

      %c_result_8x32_0_0_t1 = vector.shuffle %c_result_0_0_f16, %c_result_0_1_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_0_0_t2 = vector.shape_cast %c_result_8x32_0_0_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_0_0 = vector.shape_cast %c_result_8x32_0_0_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_00 = xegpu.create_nd_tdesc %C[%C_sg_tile_offset_x, %C_sg_tile_offset_y], [%M, %N], [%N, %c1] : memref<?x?xf16> -> !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_0_0, %c_sg_tile_00 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>
      xegpu.compile_hint

      %c_result_8x32_1_0_t1 = vector.shuffle %c_result_1_0_f16, %c_result_1_1_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_1_0_t2 = vector.shape_cast %c_result_8x32_1_0_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_1_0 = vector.shape_cast %c_result_8x32_1_0_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_10 = xegpu.update_nd_offset %c_sg_tile_00, [%c8, %c0]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_1_0, %c_sg_tile_10 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>

      %c_result_8x32_2_0_t1 = vector.shuffle %c_result_2_0_f16, %c_result_2_1_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_2_0_t2 = vector.shape_cast %c_result_8x32_2_0_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_2_0 = vector.shape_cast %c_result_8x32_2_0_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_20 = xegpu.update_nd_offset %c_sg_tile_10, [%c8, %c0]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_2_0, %c_sg_tile_20 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>

      %c_result_8x32_3_0_t1 = vector.shuffle %c_result_3_0_f16, %c_result_3_1_f16 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x16xf16>, vector<8x16xf16>
      %c_result_8x32_3_0_t2 = vector.shape_cast %c_result_8x32_3_0_t1 : vector<16x16xf16> to vector<256xf16>
      %c_result_8x32_3_0 = vector.shape_cast %c_result_8x32_3_0_t2 : vector<256xf16> to vector<8x32xf16>
      %c_sg_tile_30 = xegpu.update_nd_offset %c_sg_tile_20, [%c8, %c0]  : !xegpu.tensor_desc<8x32xf16>
      xegpu.store_nd %c_result_8x32_3_0, %c_sg_tile_30 {l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>} : vector<8x32xf16>, !xegpu.tensor_desc<8x32xf16>

      gpu.return
    }


  }

  // compute CPU reference (takes minutes)
  // compute CPU reference (takes minutes)
  func.func @cpu_reference(%A : memref<?x?xf16>, %B : memref<?x?xf16>, %C : memref<?x?xf32>, %M: index, %K: index, %N: index) {
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    scf.for %i = %c0 to %M step %c1 {
      scf.for %j = %c0 to %N step %c1 {
        %c_curr = memref.load %C[%i, %j] : memref<?x?xf32>
        %c_val = scf.for %k_tile = %c0 to %K step %c16 iter_args(%c_partial = %c_curr) -> f32 {
          %c_val_dpas = scf.for %k = %c0 to %c16 step %c1 iter_args(%c_dpas_partial = %c_partial) -> f32 {
            %k_dpas = arith.addi %k_tile, %k : index
            %a_val = memref.load %A[%i, %k_dpas] : memref<?x?xf16>
            %b_val = memref.load %B[%k_dpas, %j] : memref<?x?xf16>
            %a_cast = arith.extf %a_val : f16 to f32
            %b_cast = arith.extf %b_val : f16 to f32
            %t = arith.mulf %a_cast, %b_cast : f32
            // %t_cast = arith.extf %t : f16 to f16
            %c_sum = arith.addf %t, %c_dpas_partial : f32
            scf.yield %c_sum : f32
          }
          scf.yield %c_val_dpas : f32
        }
        %c_val_f16 = arith.truncf %c_val : f32 to f16
        %c_val_ = arith.extf %c_val_f16 : f16 to f32
        memref.store %c_val_ , %C[%i, %j] : memref<?x?xf32>
      }
    }
    return
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_f16 = arith.constant 1.0 : f16
    %c2_f16 = arith.constant 2.0 : f16
    %c4096 = arith.constant 4096 : index
    %cf_0 = arith.constant 0.0 : f16
    %cf_1 = arith.constant 1.0 : f16
    %c_gen_int = arith.constant 0 : i1
    %cf_lower = arith.constant -0.5 : f32
    %cf_upper = arith.constant 0.5 : f32

    // -------------------------------------- Start of GEMM parameters -------------------------------------- //

    // Matrices with dynamic size, neded for the CPU reference and gpu test
    // For any GPU GEMM that uses wg_tile (256,128) and sg_tile (32,32,32), they only need to modify the matrix size
    // and prefetch_distance. The rest of the code is the same.

    // Original Matrices with static size

    %M = arith.constant 4096 : index
    %K = arith.constant 2048 : index
    %N = arith.constant 8192 : index

    // Workgroup tile size for the GPU
    %wg_M = arith.constant 256 : index
    %wg_K = arith.constant 2048 : index
    %wg_N = arith.constant 256 : index

    // Subgroup tile size for the GPU
    %sg_M = arith.constant 32 : index
    %sg_K = arith.constant 32 : index
    %sg_N = arith.constant 64 : index


    // Workgroup set up in the gpu
    %wg_X = arith.constant 16 : index
    %wg_Y = arith.constant 32 : index
    %wg_Z = arith.constant 1 : index
    // Subgroup set up in the gpu
    %sg_X = arith.constant 8 : index
    %sg_Y = arith.constant 4 : index
    %sg_Z = arith.constant 1 : index

    // Prefetch distance for the GPU
    %prefetch_distance = arith.constant 3 : index

    // barrier distance for the GPU, after how many iterations of k loop the barrier is called
    %barrier_distance = arith.constant 8 : index


    // Allocate A, B, C matrices
    %A = memref.alloc(%M, %K) : memref<?x?xf16>
    %B = memref.alloc(%K, %N) : memref<?x?xf16>
    %C = memref.alloc(%M, %N) : memref<?x?xf16>
    %C_ref = memref.alloc(%M, %N) : memref<?x?xf32>

    // -------------------------------------- End of GEMM parameters -------------------------------------- //

    %A_random = memref.cast %A : memref<?x?xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%A_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf16>, f32, f32, i1) -> ()

    // Option 2:  convert the memref to 1D and fill with random values in (-0.5, 0.5)
    %B_random = memref.cast %B : memref<?x?xf16>  to memref<*xf16>
    call @fillResource1DRandomF16(%B_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf16>, f32, f32, i1) -> ()

    // intialize matrix C and C_ref ; C[i, j] = 0
    %c0_f16 = arith.constant 0.0 : f16
    %c0_f32 = arith.constant 0.0 : f32
    scf.for %i = %c0 to %M step %c1 {
      scf.for %j = %c0 to %N step %c1 {
        memref.store %c0_f16, %C[%i, %j] : memref<?x?xf16>
        memref.store %c0_f32, %C_ref[%i, %j] : memref<?x?xf32>
      }
    }
    // print input fror debug
    // %A_row_0 = memref.subview %A[1, 0][1, 4096][1, 1] : memref<?x?xf16> to memref<1x4096xf16, strided<[4096, 1], offset: 4096>>
    // %A_row_0_cast = memref.cast %A_row_0 : memref<1x4096xf16, strided<[4096, 1], offset: 4096>> to memref<*xf16>
    // call @printMemrefF16(%A_row_0_cast) : (memref<*xf16>) -> ()

    // run GPU
    %2 = call @test(%A, %B, %C, %M, %K, %N, %wg_M, %wg_K, %wg_N, %sg_M, %sg_K, %sg_N, %wg_X, %wg_Y, %wg_Z, %sg_X, %sg_Y, %sg_Z, %prefetch_distance, %barrier_distance) : (memref<?x?xf16>, memref<?x?xf16>, memref<?x?xf16>, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) -> memref<?x?xf16>

    call @cpu_reference(%A, %B, %C_ref, %M, %K, %N) : (memref<?x?xf16>, memref<?x?xf16>, memref<?x?xf32>, index, index, index) -> ()

    // // %cast = memref.cast %A : memref<?x?xf16> to memref<*xf16>
    // // call @printMemrefF16(%cast) : (memref<*xf16>) -> ()
    %cast_C = memref.cast %2 : memref<?x?xf16> to memref<*xf16>
    %cast_C_ref = memref.cast %C_ref : memref<?x?xf32> to memref<*xf32>
    // // call @printMemrefF16(%cast_C) : (memref<*xf16>) -> ()
    // // call @printMemrefF32(%cast_C_ref) : (memref<*xf32>) -> ()


    // %nth_row = arith.constant 0 : index

    // %C_row_0 = memref.subview %C_ref[%nth_row, %c0][1, %N][1, 1] : memref<?x?xf32> to memref<1x?xf32, strided<[?, 1], offset:?>>
    // %C_row_0_cast = memref.cast %C_row_0 : memref<1x?xf32, strided<[?, 1], offset: ?>> to memref<*xf32>
    // call @printMemrefF32(%C_row_0_cast) : (memref<*xf32>) -> ()

    // %C_row_0_gpu  = memref.subview %2[%nth_row, %c0][1, %N][1, 1] : memref<?x?xf16> to memref<1x?xf16, strided<[?, 1], offset:?>>
    // %C_row_0_cast_gpu = memref.cast %C_row_0_gpu : memref<1x?xf16, strided<[?, 1], offset: ?>> to memref<*xf16>
    // call @printMemrefF16(%C_row_0_cast_gpu) : (memref<*xf16>) -> ()

    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF16(%cast_C, %cast_C_ref) : (memref<*xf16>, memref<*xf32>) -> ()
    // call @printAllcloseF16(%C_row_0_cast_gpu, %C_row_0_cast) : (memref<*xf16>, memref<*xf32>) -> ()

    memref.dealloc %A : memref<?x?xf16>
    memref.dealloc %B : memref<?x?xf16>
    memref.dealloc %C : memref<?x?xf16>
    memref.dealloc %C_ref : memref<?x?xf32>
    return
  }
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}

}






