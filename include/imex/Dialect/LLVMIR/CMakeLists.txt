add_mlir_dialect(XeVMOps xevm)
add_mlir_doc(XeVMOps XeVMDialect Dialects/ -gen-dialect-doc -dialect=xevm)
set(LLVM_TARGET_DEFINITIONS XeVMOps.td)
mlir_tablegen(XeVMConversions.inc -gen-llvmir-conversions)
mlir_tablegen(XeVMOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(XeVMOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(XeVMOpsAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=xevm)
mlir_tablegen(XeVMOpsAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=xevm)
add_public_tablegen_target(MLIRXeVMConversionsIncGen)
