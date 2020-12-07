
≥
yvoid im2col4d_kernel<float, int>(im2col4d_params, cudnnConvolutionStruct, cudnnTensor4dStruct, float const*, float*, int)*28æ€@æ€Hæ€Xbears_model/conv2d_220/Conv2Dh
è
≤void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, int, int, float, int, int, int)*28ﬁò@ﬁòHﬁòXb>gradient_tape/ears_model/conv2d_205/Conv2D/Conv2DBackpropInputh
è
≤void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, int, int, float, int, int, int)*28ﬁï@ﬁïHﬁïXb>gradient_tape/ears_model/conv2d_221/Conv2D/Conv2DBackpropInputh
^
$volta_scudnn_128x64_relu_small_nn_v1*28ˇÔ@ˇÔHˇÔXbears_model/conv2d_201/Conv2Dh
^
$volta_scudnn_128x64_relu_small_nn_v1*28ˇÔ@ˇÔHˇÔXbears_model/conv2d_203/Conv2Dh
^
$volta_scudnn_128x64_relu_small_nn_v1*28ˇÎ@ˇÎHˇÎXbears_model/conv2d_217/Conv2Dh
^
$volta_scudnn_128x64_relu_small_nn_v1*28øÎ@øÎHøÎXbears_model/conv2d_219/Conv2Dh
q
volta_sgemm_128x64_nt*28ﬂ€@ﬂ€Hﬂ€Xb>gradient_tape/ears_model/conv2d_217/Conv2D/Conv2DBackpropInputh
q
volta_sgemm_128x64_nt*28ˇœ@ˇœHˇœXb>gradient_tape/ears_model/conv2d_203/Conv2D/Conv2DBackpropInputh
q
volta_sgemm_128x64_nt*28ˇœ@ˇœHˇœXb>gradient_tape/ears_model/conv2d_219/Conv2D/Conv2DBackpropInputh
q
volta_sgemm_128x64_nt*28ﬂŒ@ﬂŒHﬂŒXb>gradient_tape/ears_model/conv2d_201/Conv2D/Conv2DBackpropInputh
a
'volta_scudnn_128x64_relu_interior_nn_v1*28ü≤@ü≤Hü≤Xbears_model/conv2d_202/Conv2Dh
ê
3volta_scudnn_128x128_stridedB_splitK_interior_nn_v1*28øæ@øæHøæXb?gradient_tape/ears_model/conv2d_221/Conv2D/Conv2DBackpropFilterh
˜
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28†¥@†¥H†¥b%Adam/Adam/update_52/ResourceApplyAdamh
˜
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28ˇØ@ˇØHˇØb%Adam/Adam/update_44/ResourceApplyAdamh
‹
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)*28øß@øßHøßXb?gradient_tape/ears_model/conv2d_190/Conv2D/Conv2DBackpropFilterh
ó
;volta_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1*28Ä†@Ä†HÄ†Xb>gradient_tape/ears_model/conv2d_195/Conv2D/Conv2DBackpropInputh
ó
;volta_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1*28ﬂü@ﬂüHﬂüXb>gradient_tape/ears_model/conv2d_191/Conv2D/Conv2DBackpropInputh
è
2volta_scudnn_128x64_stridedB_splitK_interior_nn_v1*28ˇú@ˇúHˇúXb?gradient_tape/ears_model/conv2d_220/Conv2D/Conv2DBackpropFilterh
ó
;volta_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1*28†ö@†öH†öXb>gradient_tape/ears_model/conv2d_193/Conv2D/Conv2DBackpropInputh
å
/volta_scudnn_128x64_stridedB_splitK_small_nn_v1*28¿ï@¿ïH¿ïXb?gradient_tape/ears_model/conv2d_201/Conv2D/Conv2DBackpropFilterh
™
Ãvoid cudnn::detail::wgrad_alg1_engine<float, 128, 6, 8, 3, 3, 5, true, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, int, float, float, int, int, int*, int*, int, int)*28ﬂì@ﬂìHﬂìXb?gradient_tape/ears_model/conv2d_217/Conv2D/Conv2DBackpropFilterh
å
/volta_scudnn_128x64_stridedB_splitK_small_nn_v1*28ﬂì@ﬂìHﬂìXb?gradient_tape/ears_model/conv2d_203/Conv2D/Conv2DBackpropFilterh
ñ
€void cudnn::detail::implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, true, false, true>(int, int, int, float const*, int, float*, float*, kernel_conv_params, int, float, float, int, float*, float*, int, int)*28üà@üàHüàXbears_model/conv2d_191/Conv2Dh
á
≠void pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor)*28‡Ç@‡ÇH‡Çb=gradient_tape/ears_model/max_pooling2d_68/MaxPool/MaxPoolGradh
™
Ãvoid cudnn::detail::wgrad_alg1_engine<float, 128, 6, 8, 3, 3, 5, true, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, int, float, float, int, int, int*, int*, int, int)*28øÅ@øÅHøÅXb?gradient_tape/ears_model/conv2d_219/Conv2D/Conv2DBackpropFilterh
˜
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28ÄÄ@ÄÄHÄÄb%Adam/Adam/update_54/ResourceApplyAdamh
“
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28ˇˇ@ˇˇHˇˇb@gradient_tape/ears_model/leaky_re_lu_150/LeakyRelu/LeakyReluGradh
˜
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28ˇˇ@ˇˇHˇˇb%Adam/Adam/update_46/ResourceApplyAdamh
˚
üvoid fft2d_r2c_32x32<float, false, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)*28ü˝@†}HˇXb?gradient_tape/ears_model/conv2d_191/Conv2D/Conv2DBackpropFilterh
a
'volta_scudnn_128x64_relu_interior_nn_v1*28ˇ¸@ˇ¸Hˇ¸Xbears_model/conv2d_214/Conv2Dh
a
'volta_scudnn_128x64_relu_interior_nn_v1*28ü˚@ü˚Hü˚Xbears_model/conv2d_200/Conv2Dh
ç
“void cudnn::detail::explicit_convolve_sgemm<float, int, 128, 5, 5, 3, 3, 3, 0, true>(int, int, int, float const*, int, float const*, int, float*, kernel_conv_params, int, int, float, float, int, float*, float*)*28‡@‡H‡Xbears_model/conv2d_220/Conv2Dh
r
volta_sgemm_128x64_nt*28Ä@ÄHÄXb?gradient_tape/ears_model/conv2d_193/Conv2D/Conv2DBackpropFilterh
a
'volta_scudnn_128x64_relu_interior_nn_v1*28ˇÔ@ˇÔHˇÔXbears_model/conv2d_216/Conv2Dh
è
2volta_scudnn_128x64_stridedB_splitK_interior_nn_v1*28ˇÔ@ˇÔHˇÔXb?gradient_tape/ears_model/conv2d_205/Conv2D/Conv2DBackpropFilterh
r
volta_sgemm_128x64_nt*28øÔ@øÔHøÔXb?gradient_tape/ears_model/conv2d_195/Conv2D/Conv2DBackpropFilterh
á
≠void pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor)*28ˇÓ@ˇÓHˇÓb=gradient_tape/ears_model/max_pooling2d_72/MaxPool/MaxPoolGradh
r
volta_sgemm_128x64_nt*28†È@†ÈH†ÈXb?gradient_tape/ears_model/conv2d_197/Conv2D/Conv2DBackpropFilterh
r
volta_sgemm_128x64_nt*28ˇ‰@ˇ‰Hˇ‰Xb?gradient_tape/ears_model/conv2d_199/Conv2D/Conv2DBackpropFilterh
‡
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28ˇﬂ@ˇﬂHˇﬂXb>gradient_tape/ears_model/conv2d_217/Conv2D/Conv2DBackpropInputh
å
≤void pooling_bw_kernel_max_nchw_fully_packed_large<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, int)*28ˇﬂ@ˇﬂHˇﬂb=gradient_tape/ears_model/max_pooling2d_65/MaxPool/MaxPoolGradh
q
volta_sgemm_128x64_nt*28ˇﬂ@ˇﬂHˇﬂXb>gradient_tape/ears_model/conv2d_199/Conv2D/Conv2DBackpropInputh
≥
yvoid im2col4d_kernel<float, int>(im2col4d_params, cudnnConvolutionStruct, cudnnTensor4dStruct, float const*, float*, int)*28‡›@‡›H‡›Xbears_model/conv2d_213/Conv2Dh
‡
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28ø›@ø›Hø›Xb>gradient_tape/ears_model/conv2d_219/Conv2D/Conv2DBackpropInputh
q
volta_sgemm_128x64_nt*28Äÿ@ÄÿHÄÿXb>gradient_tape/ears_model/conv2d_197/Conv2D/Conv2DBackpropInputh
˚
ùvoid gemv2N_kernel<int, int, float, float, float, 128, 8, 4, 4, 1, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)*28Ä◊@Ä◊HÄ◊Xb?gradient_tape/ears_model/conv2d_190/Conv2D/Conv2DBackpropFilterh
q
volta_sgemm_128x64_nt*28†÷@†÷H†÷Xb>gradient_tape/ears_model/conv2d_213/Conv2D/Conv2DBackpropInputh
è
2volta_scudnn_128x64_stridedB_splitK_interior_nn_v1*28Ä‘@Ä‘HÄ‘Xb?gradient_tape/ears_model/conv2d_204/Conv2D/Conv2DBackpropFilterh
ó
‹void cudnn::detail::implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, true, false, true>(int, int, int, float const*, int, float*, float*, kernel_conv_params, int, float, float, int, float*, float*, int, int)*28¿“@¿“H¿“Xbears_model/conv2d_204/Conv2Dh
≥
yvoid im2col4d_kernel<float, int>(im2col4d_params, cudnnConvolutionStruct, cudnnTensor4dStruct, float const*, float*, int)*28¿“@¿“H¿“Xbears_model/conv2d_215/Conv2Dh
ó
‹void cudnn::detail::implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, true, false, true>(int, int, int, float const*, int, float*, float*, kernel_conv_params, int, float, float, int, float*, float*, int, int)*28ˇœ@ˇœHˇœXbears_model/conv2d_218/Conv2Dh
è
2volta_scudnn_128x64_stridedB_splitK_interior_nn_v1*28ˇœ@ˇœHˇœXb?gradient_tape/ears_model/conv2d_202/Conv2D/Conv2DBackpropFilterh
q
volta_sgemm_128x64_nt*28ˇœ@ˇœHˇœXb>gradient_tape/ears_model/conv2d_215/Conv2D/Conv2DBackpropInputh
‡
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28üœ@üœHüœXb>gradient_tape/ears_model/conv2d_201/Conv2D/Conv2DBackpropInputh
è
2volta_scudnn_128x64_stridedB_splitK_interior_nn_v1*28Äœ@ÄœHÄœXb?gradient_tape/ears_model/conv2d_200/Conv2D/Conv2DBackpropFilterh
ï
∑void cudnn::detail::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, true, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, int, float, int, int, int, int)*28ÄŒ@ÄŒHÄŒXb?gradient_tape/ears_model/conv2d_194/Conv2D/Conv2DBackpropFilterh
‡
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28ﬂ≈@ﬂ≈Hﬂ≈Xb>gradient_tape/ears_model/conv2d_203/Conv2D/Conv2DBackpropInputh
r
volta_sgemm_128x64_nt*28¿¿@¿¿H¿¿Xb?gradient_tape/ears_model/conv2d_213/Conv2D/Conv2DBackpropFilterh
r
volta_sgemm_128x64_nt*28ü¿@ü¿Hü¿Xb?gradient_tape/ears_model/conv2d_215/Conv2D/Conv2DBackpropFilterh
^
$volta_scudnn_128x32_relu_small_nn_v1*28†∑@†∑H†∑Xbears_model/conv2d_207/Conv2Dh
q
volta_cgemm_32x32_tn*28‡∂@‡∂H‡∂Xb?gradient_tape/ears_model/conv2d_191/Conv2D/Conv2DBackpropFilterh
^
$volta_scudnn_128x64_relu_small_nn_v1*28ˇ±@ˇ±Hˇ±Xbears_model/conv2d_193/Conv2Dh
ó
‹void cudnn::detail::implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, true, false, true>(int, int, int, float const*, int, float*, float*, kernel_conv_params, int, float, float, int, float*, float*, int, int)*28ﬂ±@ﬂ±Hﬂ±Xbears_model/conv2d_205/Conv2Dh
^
$volta_scudnn_128x64_relu_small_nn_v1*28Ä∞@Ä∞HÄ∞Xbears_model/conv2d_195/Conv2Dh
Ø
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28†§@†§H†§b7gradient_tape/ears_model/conv2d_190/BiasAdd/BiasAddGradh
á
≠void pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor)*28†¢@†¢H†¢b=gradient_tape/ears_model/max_pooling2d_67/MaxPool/MaxPoolGradh
O
volta_sgemm_128x64_nn*28Ä†@Ä†HÄ†Xbears_model/conv2d_199/Conv2Dh
a
'volta_scudnn_128x64_relu_interior_nn_v1*28ˇü@ˇüHˇüXbears_model/conv2d_212/Conv2Dh
O
volta_sgemm_128x64_nn*28ˇü@ˇüHˇüXbears_model/conv2d_197/Conv2Dh
¥
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28‡ü@‡üH‡übogradient_tape/ears_model/max_pooling2d_65/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
a
'volta_scudnn_128x64_relu_interior_nn_v1*28Äù@ÄùHÄùXbears_model/conv2d_196/Conv2Dh
a
'volta_scudnn_128x64_relu_interior_nn_v1*28ﬂú@ﬂúHﬂúXbears_model/conv2d_194/Conv2Dh
a
'volta_scudnn_128x64_relu_interior_nn_v1*28‡ñ@‡ñH‡ñXbears_model/conv2d_210/Conv2Dh
á
+volta_scudnn_128x64_stridedB_interior_nn_v1*28üñ@üñHüñXb>gradient_tape/ears_model/conv2d_204/Conv2D/Conv2DBackpropInputh
≤
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Äì@ÄìHÄìbmgradient_tape/ears_model/max_pooling2d_65/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
á
+volta_scudnn_128x64_stridedB_interior_nn_v1*28ˇè@ˇèHˇèXb>gradient_tape/ears_model/conv2d_220/Conv2D/Conv2DBackpropInputh
æ
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28†è@†èH†èXbears_model/conv2d_197/Conv2Dh
˜
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28‡é@‡éH‡éb%Adam/Adam/update_60/ResourceApplyAdamh
Ø
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28†å@†åH†åb7gradient_tape/ears_model/conv2d_217/BiasAdd/BiasAddGradh
¿
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28†å@†åH†åXbears_model/conv2d_201/Conv2Dh
ó
‹void cudnn::detail::implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, true, false, true>(int, int, int, float const*, int, float*, float*, kernel_conv_params, int, float, float, int, float*, float*, int, int)*28Äå@ÄåHÄåXbears_model/conv2d_221/Conv2Dh
q
volta_sgemm_128x64_nt*28‡â@‡âH‡âXb>gradient_tape/ears_model/conv2d_209/Conv2D/Conv2DBackpropInputh
æ
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28Äâ@ÄâHÄâXbears_model/conv2d_199/Conv2Dh
¿
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28‡Ö@‡ÖH‡ÖXbears_model/conv2d_217/Conv2Dh
á
≠void pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor)*28ÄÖ@ÄÖHÄÖb=gradient_tape/ears_model/max_pooling2d_71/MaxPool/MaxPoolGradh
¥
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28†Ñ@†ÑH†Ñbogradient_tape/ears_model/conv2d_190/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
Ø
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28¿É@¿ÉH¿Éb7gradient_tape/ears_model/conv2d_219/BiasAdd/BiasAddGradh
¿
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28¿É@¿ÉH¿ÉXbears_model/conv2d_203/Conv2Dh
€
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28¿Ç@¿ÇH¿ÇXb>gradient_tape/ears_model/conv2d_217/Conv2D/Conv2DBackpropInputh
q
volta_sgemm_128x64_nt*28üÇ@üÇHüÇXb>gradient_tape/ears_model/conv2d_211/Conv2D/Conv2DBackpropInputh
î
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28ﬂÄ@ﬂÄHﬂÄbOears_model/conv2d_190/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
€
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28†Ä@†ÄH†ÄXb>gradient_tape/ears_model/conv2d_219/Conv2D/Conv2DBackpropInputh
⁄
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28ÄÄ@ÄÄHÄÄXb?gradient_tape/ears_model/conv2d_190/Conv2D/Conv2DBackpropFilterh
¿
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28ÄÄ@ÄÄHÄÄXbears_model/conv2d_219/Conv2Dh
‚
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28ÄÄ@ÄÄHÄÄXb>gradient_tape/ears_model/conv2d_203/Conv2D/Conv2DBackpropInputh
ã
”void cudnn::detail::explicit_convolve_sgemm<float, int, 1024, 5, 5, 3, 3, 3, 0, true>(int, int, int, float const*, int, float const*, int, float*, kernel_conv_params, int, int, float, float, int, float*, float*)*28ˇ@ˇHˇXbears_model/conv2d_215/Conv2Dh
¶
Àvoid fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)*28ˇ@ˇHˇXb?gradient_tape/ears_model/conv2d_191/Conv2D/Conv2DBackpropFilterh
Ñ
≠void pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor)*28‡@‡H‡b=gradient_tape/ears_model/max_pooling2d_66/MaxPool/MaxPoolGradh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28‡@‡H‡b%Adam/Adam/update_62/ResourceApplyAdamh
í
∑void cudnn::detail::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, true, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, int, float, int, int, int, int)*28ﬂ@ﬂHﬂXb?gradient_tape/ears_model/conv2d_218/Conv2D/Conv2DBackpropFilterh
ÿ
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28‡}@‡}H‡}Xb>gradient_tape/ears_model/conv2d_203/Conv2D/Conv2DBackpropInputh
[
$volta_scudnn_128x32_relu_small_nn_v1*28¿}@¿}H¿}Xbears_model/conv2d_190/Conv2Dh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28†}@†}H†}Xb>gradient_tape/ears_model/conv2d_201/Conv2D/Conv2DBackpropInputh
î
‹void cudnn::detail::implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, true, false, true>(int, int, int, float const*, int, float*, float*, kernel_conv_params, int, float, float, int, float*, float*, int, int)*28Ä}@Ä}HÄ}Xbears_model/conv2d_198/Conv2Dh
ÿ
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Ä}@Ä}HÄ}Xb>gradient_tape/ears_model/conv2d_201/Conv2D/Conv2DBackpropInputh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28ˇ{@ˇ{Hˇ{Xb>gradient_tape/ears_model/conv2d_219/Conv2D/Conv2DBackpropInputh
í
∑void cudnn::detail::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, true, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, int, float, int, int, int, int)*28øz@øzHøzXb?gradient_tape/ears_model/conv2d_211/Conv2D/Conv2DBackpropFilterh
·
Üvoid cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)*28†z@†zH†zXb?gradient_tape/ears_model/conv2d_199/Conv2D/Conv2DBackpropFilterh
S
volta_sgemm_32x32_sliced1x4_nn*28†y@†yH†yXbears_model/dense_12/MatMulh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Äx@ÄxHÄxXb>gradient_tape/ears_model/conv2d_217/Conv2D/Conv2DBackpropInputh
ã
”void cudnn::detail::explicit_convolve_sgemm<float, int, 1024, 5, 5, 3, 3, 3, 0, true>(int, int, int, float const*, int, float const*, int, float*, kernel_conv_params, int, int, float, float, int, float*, float*)*28†w@†wH†wXbears_model/conv2d_213/Conv2Dh
í
∑void cudnn::detail::wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, true, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, int, float, int, int, int, int)*28†w@†wH†wXb?gradient_tape/ears_model/conv2d_209/Conv2D/Conv2DBackpropFilterh
Â
ëvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)*28¿u@¿uH¿ub:categorical_crossentropy/softmax_cross_entropy_with_logitsh
·
Üvoid cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)*28ﬂt@ﬂtHﬂtXb?gradient_tape/ears_model/conv2d_213/Conv2D/Conv2DBackpropFilterh
·
Üvoid cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)*28üt@ütHütXb?gradient_tape/ears_model/conv2d_197/Conv2D/Conv2DBackpropFilterh
í
∑void cudnn::detail::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, true, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, int, float, int, int, int, int)*28ˇs@ˇsHˇsXb?gradient_tape/ears_model/conv2d_206/Conv2D/Conv2DBackpropFilterh
Ö
,volta_scudnn_128x128_stridedB_interior_nn_v1*28¿s@¿sH¿sXb>gradient_tape/ears_model/conv2d_212/Conv2D/Conv2DBackpropInputh
Ö
,volta_scudnn_128x128_stridedB_interior_nn_v1*28†s@†sH†sXb>gradient_tape/ears_model/conv2d_214/Conv2D/Conv2DBackpropInputh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28Äs@ÄsHÄsb$ears_model/leaky_re_lu_150/LeakyReluh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28†r@†rH†rXb?gradient_tape/ears_model/conv2d_219/Conv2D/Conv2DBackpropFilterh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28‡q@‡qH‡qXb?gradient_tape/ears_model/conv2d_217/Conv2D/Conv2DBackpropFilterh
¥
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)*28‡p@Ä0H‡@Xb?gradient_tape/ears_model/conv2d_192/Conv2D/Conv2DBackpropFilterh
·
Üvoid cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)*28Äp@ÄpHÄpXb?gradient_tape/ears_model/conv2d_215/Conv2D/Conv2DBackpropFilterh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Äp@ÄpHÄpXb?gradient_tape/ears_model/conv2d_201/Conv2D/Conv2DBackpropFilterh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28‡o@‡oH‡oXb?gradient_tape/ears_model/conv2d_203/Conv2D/Conv2DBackpropFilterh
Ñ
+volta_scudnn_128x64_stridedB_interior_nn_v1*28øl@ølHølXb>gradient_tape/ears_model/conv2d_216/Conv2D/Conv2DBackpropInputh
í
∑void cudnn::detail::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, true, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, int, float, int, int, int, int)*28Äl@ÄlHÄlXb?gradient_tape/ears_model/conv2d_198/Conv2D/Conv2DBackpropFilterh
Ñ
+volta_scudnn_128x64_stridedB_interior_nn_v1*28¿j@¿jH¿jXb>gradient_tape/ears_model/conv2d_202/Conv2D/Conv2DBackpropInputh
ÿ
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28†j@†jH†jXb>gradient_tape/ears_model/conv2d_213/Conv2D/Conv2DBackpropInputh
ÿ
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28†i@†iH†iXb>gradient_tape/ears_model/conv2d_215/Conv2D/Conv2DBackpropInputh
¨
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28Äi@ÄiHÄib7gradient_tape/ears_model/conv2d_191/BiasAdd/BiasAddGradh
Ñ
+volta_scudnn_128x64_stridedB_interior_nn_v1*28ﬂe@ﬂeHﬂeXb>gradient_tape/ears_model/conv2d_218/Conv2D/Conv2DBackpropInputh
Ñ
+volta_scudnn_128x64_stridedB_interior_nn_v1*28¿e@¿eH¿eXb>gradient_tape/ears_model/conv2d_200/Conv2D/Conv2DBackpropInputh
Ö
,volta_scudnn_128x128_stridedB_interior_nn_v1*28ˇb@ˇbHˇbXb>gradient_tape/ears_model/conv2d_198/Conv2D/Conv2DBackpropInputh
Ö
,volta_scudnn_128x128_stridedB_interior_nn_v1*28üb@übHübXb>gradient_tape/ears_model/conv2d_196/Conv2D/Conv2DBackpropInputh
n
volta_sgemm_128x64_nt*28†`@†`H†`Xb>gradient_tape/ears_model/conv2d_207/Conv2D/Conv2DBackpropInputh
¢
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*28Ä`@Ä`HÄ`Xb?gradient_tape/ears_model/conv2d_191/Conv2D/Conv2DBackpropFilterh
¨
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28†_@†_H†_b7gradient_tape/ears_model/conv2d_215/BiasAdd/BiasAddGradh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28‡^@‡^H‡^b%Adam/Adam/update_36/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä^@Ä^HÄ^b%Adam/Adam/update_28/ResourceApplyAdamh
ÿ
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28†]@†]H†]Xb>gradient_tape/ears_model/conv2d_211/Conv2D/Conv2DBackpropInputh
·
Üvoid cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)*28‡Z@‡ZH‡ZXb?gradient_tape/ears_model/conv2d_195/Conv2D/Conv2DBackpropFilterh
¨
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28¿Z@¿ZH¿Zb7gradient_tape/ears_model/conv2d_213/BiasAdd/BiasAddGradh
·
Üvoid cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)*28ÄX@ÄXHÄXXb?gradient_tape/ears_model/conv2d_190/Conv2D/Conv2DBackpropFilterh
Ö
,volta_scudnn_128x128_stridedB_interior_nn_v1*28ÄX@ÄXHÄXXb>gradient_tape/ears_model/conv2d_210/Conv2D/Conv2DBackpropInputh
·
Üvoid cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)*28‡U@‡UH‡UXb?gradient_tape/ears_model/conv2d_193/Conv2D/Conv2DBackpropFilterh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28ÄU@ÄUHÄUb%Adam/Adam/update_38/ResourceApplyAdamh
í
∑void cudnn::detail::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, true, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, int, float, int, int, int, int)*28ﬂT@ﬂTHﬂTXb?gradient_tape/ears_model/conv2d_216/Conv2D/Conv2DBackpropFilterh
ÿ
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28ÄT@ÄTHÄTXb>gradient_tape/ears_model/conv2d_209/Conv2D/Conv2DBackpropInputh
ÿ
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28‡R@‡RH‡RXb>gradient_tape/ears_model/conv2d_199/Conv2D/Conv2DBackpropInputh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28‡R@‡RH‡Rb%Adam/Adam/update_30/ResourceApplyAdamh
^
'volta_scudnn_128x32_relu_interior_nn_v1*28‡P@‡PH‡PXbears_model/conv2d_192/Conv2Dh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28ÄP@ÄPHÄPb@gradient_tape/ears_model/leaky_re_lu_151/LeakyRelu/LeakyReluGradh
í
∑void cudnn::detail::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, true, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, int, float, int, int, int, int)*28ÄP@ÄPHÄPXb?gradient_tape/ears_model/conv2d_207/Conv2D/Conv2DBackpropFilterh
ÿ
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28ÄP@ÄPHÄPXb>gradient_tape/ears_model/conv2d_197/Conv2D/Conv2DBackpropInputh
L
volta_sgemm_128x64_nn*28ÄP@ÄPHÄPXbears_model/conv2d_209/Conv2Dh
—
îvoid cudnn::detail::pooling_fw_4d_kernel<float, float, cudnn::detail::maxpooling_func<float, (cudnnNanPropagation_t)0>, 0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28†O@†OH†Ob#ears_model/max_pooling2d_65/MaxPoolh
ª
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28†O@†OH†OXbears_model/conv2d_209/Conv2Dh
Ñ
+volta_scudnn_128x64_stridedB_interior_nn_v1*28üO@üOHüOXb>gradient_tape/ears_model/conv2d_194/Conv2D/Conv2DBackpropInputh
í
∑void cudnn::detail::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, true, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, int, float, int, int, int, int)*28ÄO@ÄOHÄOXb?gradient_tape/ears_model/conv2d_214/Conv2D/Conv2DBackpropFilterh
í
∑void cudnn::detail::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, true, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, int, float, int, int, int, int)*28‡N@‡NH‡NXb?gradient_tape/ears_model/conv2d_196/Conv2D/Conv2DBackpropFilterh
L
volta_sgemm_128x64_nn*28‡N@‡NH‡NXbears_model/conv2d_211/Conv2Dh
Ñ
+volta_scudnn_128x64_stridedB_interior_nn_v1*28†N@†NH†NXb>gradient_tape/ears_model/conv2d_208/Conv2D/Conv2DBackpropInputh
Ù
£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28ÄN@ÄNHÄNb7gradient_tape/ears_model/conv2d_221/BiasAdd/BiasAddGradh
ª
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28ﬂM@ﬂMHﬂMXbears_model/conv2d_211/Conv2Dh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28‡L@‡LH‡Lb%Adam/Adam/update_64/ResourceApplyAdamh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28†L@†LH†Lb@gradient_tape/ears_model/leaky_re_lu_155/LeakyRelu/LeakyReluGradh
X
volta_sgemm_32x128_nt*28†L@†LH†Lb*gradient_tape/ears_model/dense_12/MatMul_1h
Ù
£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28ˇK@ˇKHˇKb7gradient_tape/ears_model/conv2d_205/BiasAdd/BiasAddGradh
Ñ
+volta_scudnn_128x64_stridedB_interior_nn_v1*28†K@†KH†KXb>gradient_tape/ears_model/conv2d_192/Conv2D/Conv2DBackpropInputh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28‡J@‡JH‡Jb@gradient_tape/ears_model/leaky_re_lu_153/LeakyRelu/LeakyReluGradh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28‡I@‡IH‡IXb?gradient_tape/ears_model/conv2d_205/Conv2D/Conv2DBackpropFilterh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28ÄH@ÄHHÄHbears_model/conv2d_190/BiasAddh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28‡D@‡DH‡DXbears_model/conv2d_221/Conv2Dh
[
$volta_scudnn_128x32_relu_small_nn_v1*28†D@†DH†DXbears_model/conv2d_206/Conv2Dh
Ÿ
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)*28†B@†BH†BXb?gradient_tape/ears_model/conv2d_193/Conv2D/Conv2DBackpropFilterh
Ù
£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28ÄA@ÄAHÄAb7gradient_tape/ears_model/conv2d_195/BiasAdd/BiasAddGradh
n
volta_cgemm_32x32_tn*28†@@†@H†@Xb?gradient_tape/ears_model/conv2d_192/Conv2D/Conv2DBackpropFilterh
›
âvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)*28Ä@@Ä@HÄ@b:categorical_crossentropy/softmax_cross_entropy_with_logitsh
¨
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28Ä@@Ä@HÄ@b7gradient_tape/ears_model/conv2d_211/BiasAdd/BiasAddGradh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä@@Ä@HÄ@b%Adam/Adam/update_56/ResourceApplyAdamh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä@@Ä@HÄ@Xb?gradient_tape/ears_model/conv2d_221/Conv2D/Conv2DBackpropFilterh
Ù
£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28ˇ?@ˇ?Hˇ?b7gradient_tape/ears_model/conv2d_193/BiasAdd/BiasAddGradh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä?@Ä?HÄ?b%Adam/Adam/update_58/ResourceApplyAdamh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28ˇ>@ˇ>Hˇ>Xb>gradient_tape/ears_model/conv2d_221/Conv2D/Conv2DBackpropInputh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28‡>@‡>H‡>Xbears_model/conv2d_205/Conv2Dh
Ÿ
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)*28¿>@¿>H¿>Xb?gradient_tape/ears_model/conv2d_197/Conv2D/Conv2DBackpropFilterh
Ÿ
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)*28Ä>@Ä>HÄ>Xb?gradient_tape/ears_model/conv2d_195/Conv2D/Conv2DBackpropFilterh
Ÿ
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)*28Ä>@Ä>HÄ>Xb?gradient_tape/ears_model/conv2d_213/Conv2D/Conv2DBackpropFilterh
¨
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28ﬂ=@ﬂ=Hﬂ=b7gradient_tape/ears_model/conv2d_204/BiasAdd/BiasAddGradh
X
volta_sgemm_128x32_tn*28†<@†<H†<Xb(gradient_tape/ears_model/dense_12/MatMulh
ÿ
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28¿:@¿:H¿:Xb>gradient_tape/ears_model/conv2d_207/Conv2D/Conv2DBackpropInputh
í
∑void cudnn::detail::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, true, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, int, float, int, int, int, int)*28Ä:@Ä:HÄ:Xb?gradient_tape/ears_model/conv2d_212/Conv2D/Conv2DBackpropFilterh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28ˇ8@ˇ8Hˇ8Xbears_model/conv2d_220/Conv2Dh
„
àvoid fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)*28†8@†8H†8Xb?gradient_tape/ears_model/conv2d_192/Conv2D/Conv2DBackpropFilterh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28†8@†8H†8Xbears_model/conv2d_215/Conv2Dh
‰
´void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*28ﬂ7@‡Hˇ!bears_model/concatenate_7/concath
Ÿ
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)*28¿6@¿6H¿6Xb?gradient_tape/ears_model/conv2d_199/Conv2D/Conv2DBackpropFilterh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä6@Ä6HÄ6Xb>gradient_tape/ears_model/conv2d_197/Conv2D/Conv2DBackpropInputh
¨
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28¿5@¿5H¿5b7gradient_tape/ears_model/conv2d_220/BiasAdd/BiasAddGradh
í
∑void cudnn::detail::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, true, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, int, float, int, int, int, int)*28Ä5@Ä5HÄ5Xb?gradient_tape/ears_model/conv2d_210/Conv2D/Conv2DBackpropFilterh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28†4@†4H†4Xb>gradient_tape/ears_model/conv2d_205/Conv2D/Conv2DBackpropInputh
Ÿ
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)*28¿3@¿3H¿3Xb?gradient_tape/ears_model/conv2d_215/Conv2D/Conv2DBackpropFilterh
¨
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28†3@†3H†3b7gradient_tape/ears_model/conv2d_209/BiasAdd/BiasAddGradh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä3@Ä3HÄ3Xbears_model/conv2d_199/Conv2Dh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä3@Ä3HÄ3Xbears_model/conv2d_213/Conv2Dh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28¿2@¿2H¿2Xb>gradient_tape/ears_model/conv2d_213/Conv2D/Conv2DBackpropInputh
Ö
´void cudnn::detail::pooling_bw_kernel_avg<float, float, cudnn::detail::averpooling_func<float>, 2, false>(cudnnTensorStruct, float const*, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28Ä2@Ä2HÄ2b@gradient_tape/ears_model/average_pooling2d_7/AvgPool/AvgPoolGradh
Ñ
≠void pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor)*28Ä2@Ä2HÄ2b=gradient_tape/ears_model/max_pooling2d_69/MaxPool/MaxPoolGradh
Ñ
≠void pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor)*28ˇ1@ˇ1Hˇ1b=gradient_tape/ears_model/max_pooling2d_70/MaxPool/MaxPoolGradh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28ﬂ1@ﬂ1Hﬂ1Xbears_model/conv2d_197/Conv2Dh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28¿1@¿1H¿1Xbears_model/conv2d_204/Conv2Dh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28†1@†1H†1Xb>gradient_tape/ears_model/conv2d_204/Conv2D/Conv2DBackpropInputh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä1@Ä1HÄ1Xb>gradient_tape/ears_model/conv2d_199/Conv2D/Conv2DBackpropInputh
Ù
£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28ø0@ø0Hø0b7gradient_tape/ears_model/conv2d_199/BiasAdd/BiasAddGradh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28ü0@ü0Hü0bogradient_tape/ears_model/conv2d_194/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28Ä0@Ä0HÄ0b@gradient_tape/ears_model/leaky_re_lu_159/LeakyRelu/LeakyReluGradh
˜
±void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 2> const, Eigen::DSizes<int, 2> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 2> const, Eigen::DSizes<int, 2> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä0@Ä0HÄ0b,gradient_tape/ears_model/concatenate_7/Sliceh
ª
˚void cudnn::detail::pooling_fw_4d_kernel<float, float, cudnn::detail::averpooling_func<float>, 2, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28Ä0@Ä0HÄ0b&ears_model/average_pooling2d_7/AvgPoolh
í
∑void cudnn::detail::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, true, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, int, float, int, int, int, int)*28Ä0@Ä0HÄ0Xb?gradient_tape/ears_model/conv2d_208/Conv2D/Conv2DBackpropFilterh
›
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Ä0@Ä0HÄ0Xb>gradient_tape/ears_model/conv2d_197/Conv2D/Conv2DBackpropInputh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28Ä0@Ä0HÄ0bears_model/conv2d_193/BiasAddh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28Ä0@Ä0HÄ0bears_model/conv2d_195/BiasAddh
Ù
£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Ä0@Ä0HÄ0b7gradient_tape/ears_model/conv2d_197/BiasAdd/BiasAddGradh
Ù
£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Ä0@Ä0HÄ0b7gradient_tape/ears_model/conv2d_201/BiasAdd/BiasAddGradh
Ù
£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Ä0@Ä0HÄ0b7gradient_tape/ears_model/conv2d_203/BiasAdd/BiasAddGradh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä0@Ä0HÄ0Xb?gradient_tape/ears_model/conv2d_197/Conv2D/Conv2DBackpropFilterh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä0@Ä0HÄ0Xb?gradient_tape/ears_model/conv2d_199/Conv2D/Conv2DBackpropFilterh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä0@Ä0HÄ0Xb?gradient_tape/ears_model/conv2d_215/Conv2D/Conv2DBackpropFilterh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä0@Ä0HÄ0Xb>gradient_tape/ears_model/conv2d_215/Conv2D/Conv2DBackpropInputh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä0@Ä0HÄ0Xb>gradient_tape/ears_model/conv2d_220/Conv2D/Conv2DBackpropInputh
ë
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä0@Ä0HÄ0bOears_model/conv2d_195/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
ë
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä0@Ä0HÄ0bOears_model/conv2d_197/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
ë
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä0@Ä0HÄ0bOears_model/conv2d_199/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä0@Ä0HÄ0bogradient_tape/ears_model/conv2d_193/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
≤
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä0@Ä0HÄ0bpgradient_tape/ears_model/conv2d_194/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
Ø
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä0@Ä0HÄ0bmgradient_tape/ears_model/max_pooling2d_66/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
≤
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 8, 256, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä0@Ä0HÄ0bpgradient_tape/ears_model/average_pooling2d_7/AvgPool/AvgPoolGrad-1-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28ˇ/@ˇ/Hˇ/b%Adam/Adam/update_22/ResourceApplyAdamh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28ˇ/@ˇ/Hˇ/bogradient_tape/ears_model/conv2d_195/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28‡/@‡/H‡/bogradient_tape/ears_model/max_pooling2d_67/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28¿/@¿/H¿/bears_model/conv2d_191/BiasAddh
Û
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28¿/@¿/H¿/b$Adam/Adam/update_6/ResourceApplyAdamh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28¿/@¿/H¿/Xb?gradient_tape/ears_model/conv2d_213/Conv2D/Conv2DBackpropFilterh
î
‹void cudnn::detail::implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, true, false, true>(int, int, int, float const*, int, float*, float*, kernel_conv_params, int, float, float, int, float*, float*, int, int)*28†/@†/H†/Xbears_model/conv2d_208/Conv2Dh
Ö
´void cudnn::detail::pooling_bw_kernel_avg<float, float, cudnn::detail::averpooling_func<float>, 2, false>(cudnnTensorStruct, float const*, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28†/@†/H†/b@gradient_tape/ears_model/average_pooling2d_8/AvgPool/AvgPoolGradh
ë
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28†/@†/H†/bOears_model/conv2d_191/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
Ø
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28†/@†/H†/bmgradient_tape/ears_model/max_pooling2d_67/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28ü/@ü/Hü/b%Adam/Adam/update_14/ResourceApplyAdamh
ë
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä/@Ä/HÄ/bOears_model/conv2d_193/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 4, 512, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28ˇ.@ˇ.Hˇ.bogradient_tape/ears_model/conv2d_203/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
∞
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28‡.@‡.H‡.bngradient_tape/ears_model/conv2d_191/Conv2D/Conv2DBackpropInput-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
—
îvoid cudnn::detail::pooling_fw_4d_kernel<float, float, cudnn::detail::maxpooling_func<float, (cudnnNanPropagation_t)0>, 0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28¿.@¿.H¿.b#ears_model/max_pooling2d_67/MaxPoolh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿.@¿.H¿.bogradient_tape/ears_model/max_pooling2d_68/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28†.@†.H†.b$ears_model/leaky_re_lu_155/LeakyReluh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28†.@†.H†.Xb?gradient_tape/ears_model/conv2d_204/Conv2D/Conv2DBackpropFilterh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28†.@†.H†.Xb?gradient_tape/ears_model/conv2d_220/Conv2D/Conv2DBackpropFilterh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28†.@†.H†.bogradient_tape/ears_model/max_pooling2d_66/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28‡-@‡-H‡-b$ears_model/leaky_re_lu_153/LeakyReluh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28ﬂ-@ﬂ-Hﬂ-b@gradient_tape/ears_model/leaky_re_lu_157/LeakyRelu/LeakyReluGradh
≤
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28†-@†-H†-bpgradient_tape/ears_model/conv2d_198/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä-@Ä-HÄ-b%Adam/Adam/update_48/ResourceApplyAdamh
›
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28¿,@¿,H¿,Xb>gradient_tape/ears_model/conv2d_217/Conv2D/Conv2DBackpropInputh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 4, 512, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä,@Ä,HÄ,bogradient_tape/ears_model/conv2d_202/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28Ä+@Ä+HÄ+bears_model/conv2d_205/BiasAddh
›
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28‡*@‡*H‡*Xb>gradient_tape/ears_model/conv2d_207/Conv2D/Conv2DBackpropInputh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿*@¿*H¿*bogradient_tape/ears_model/conv2d_199/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 4, 512, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿*@¿*H¿*bogradient_tape/ears_model/conv2d_201/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
›
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28Ä*@Ä*HÄ*Xb>gradient_tape/ears_model/conv2d_197/Conv2D/Conv2DBackpropInputh
∂
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28‡)@‡)H‡)Xbears_model/conv2d_209/Conv2Dh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28‡)@‡)H‡)b%Adam/Adam/update_45/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28‡)@‡)H‡)b%Adam/Adam/update_50/ResourceApplyAdamh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28‡)@‡)H‡)Xb>gradient_tape/ears_model/conv2d_202/Conv2D/Conv2DBackpropInputh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28¿)@¿)H¿)b%Adam/Adam/update_55/ResourceApplyAdamh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28¿)@¿)H¿)Xb?gradient_tape/ears_model/conv2d_202/Conv2D/Conv2DBackpropFilterh
ë
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 512, 4, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28†)@†)H†)bOears_model/conv2d_203/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
ª
˚void cudnn::detail::pooling_fw_4d_kernel<float, float, cudnn::detail::averpooling_func<float>, 2, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28Ä)@Ä)HÄ)b&ears_model/average_pooling2d_8/AvgPoolh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28‡(@‡(H‡(b@gradient_tape/ears_model/leaky_re_lu_163/LeakyRelu/LeakyReluGradh
›
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28‡(@‡(H‡(Xb>gradient_tape/ears_model/conv2d_199/Conv2D/Conv2DBackpropInputh
ª
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28¿(@¿(H¿(Xbears_model/conv2d_197/Conv2Dh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28¿(@¿(H¿(b%Adam/Adam/update_12/ResourceApplyAdamh
›
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28†(@†(H†(Xb>gradient_tape/ears_model/conv2d_211/Conv2D/Conv2DBackpropInputh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28†(@†(H†(b%Adam/Adam/update_32/ResourceApplyAdamh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28†'@†'H†'bogradient_tape/ears_model/conv2d_197/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
Û
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä'@Ä'HÄ'b$Adam/Adam/update_4/ResourceApplyAdamh
—
îvoid cudnn::detail::pooling_fw_4d_kernel<float, float, cudnn::detail::maxpooling_func<float, (cudnnNanPropagation_t)0>, 0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28‡&@‡&H‡&b#ears_model/max_pooling2d_66/MaxPoolh
¨
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28‡&@‡&H‡&b7gradient_tape/ears_model/conv2d_192/BiasAdd/BiasAddGradh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 4, 512, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä&@Ä&HÄ&bogradient_tape/ears_model/conv2d_204/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
›
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28‡%@‡%H‡%Xb>gradient_tape/ears_model/conv2d_219/Conv2D/Conv2DBackpropInputh
¨
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28‡%@‡%H‡%b7gradient_tape/ears_model/conv2d_194/BiasAdd/BiasAddGradh
ö
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 256, 8, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä%@Ä%HÄ%bXears_model/average_pooling2d_7/AvgPool-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
ë
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 512, 4, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä%@Ä%HÄ%bOears_model/conv2d_201/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
›
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28‡$@‡$H‡$Xb>gradient_tape/ears_model/conv2d_203/Conv2D/Conv2DBackpropInputh
¨
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28‡$@‡$H‡$b7gradient_tape/ears_model/conv2d_206/BiasAdd/BiasAddGradh
≤
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 128, 9, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28‡$@‡$H‡$bpgradient_tape/ears_model/conv2d_205/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
Û
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28¿$@¿$H¿$b$Adam/Adam/update_2/ResourceApplyAdamh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28†$@†$H†$Xbears_model/conv2d_202/Conv2Dh
◊
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Ä$@Ä$HÄ$Xb?gradient_tape/ears_model/conv2d_215/Conv2D/Conv2DBackpropFilterh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä$@Ä$HÄ$b%Adam/Adam/update_47/ResourceApplyAdamh
›
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28‡#@‡#H‡#Xb>gradient_tape/ears_model/conv2d_209/Conv2D/Conv2DBackpropInputh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28‡#@‡#H‡#bears_model/conv2d_199/BiasAddh
¨
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28ﬂ#@ﬂ#Hﬂ#b7gradient_tape/ears_model/conv2d_202/BiasAdd/BiasAddGradh
›
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28¿#@¿#H¿#Xb>gradient_tape/ears_model/conv2d_215/Conv2D/Conv2DBackpropInputh
¨
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28¿#@¿#H¿#b7gradient_tape/ears_model/conv2d_216/BiasAdd/BiasAddGradh
›
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28†#@†#H†#Xb>gradient_tape/ears_model/conv2d_213/Conv2D/Conv2DBackpropInputh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 4, 512, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28†#@†#H†#bogradient_tape/ears_model/conv2d_220/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
Ω
Èvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_quotient_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_quotient_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28ü#@ü#Hü#b:categorical_crossentropy/softmax_cross_entropy_with_logitsh
ª
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Ä#@Ä#HÄ#Xbears_model/conv2d_199/Conv2Dh
›
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Ä#@Ä#HÄ#Xb>gradient_tape/ears_model/conv2d_215/Conv2D/Conv2DBackpropInputh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä#@Ä#HÄ#b%Adam/Adam/update_20/ResourceApplyAdamh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä#@Ä#HÄ#bogradient_tape/ears_model/conv2d_198/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
¨
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28¿"@¿"H¿"b7gradient_tape/ears_model/conv2d_200/BiasAdd/BiasAddGradh
§
övoid tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿"@¿"H¿"bpgradient_tape/ears_model/conv2d_203/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28ø"@ø"Hø"b%Adam/Adam/update_42/ResourceApplyAdamh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28†"@†"H†"b$ears_model/leaky_re_lu_151/LeakyReluh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 4, 512, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28†"@†"H†"bogradient_tape/ears_model/conv2d_218/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
◊
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Ä"@Ä"HÄ"Xb?gradient_tape/ears_model/conv2d_195/Conv2D/Conv2DBackpropFilterh
˘
±void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 2> const, Eigen::DSizes<int, 2> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 2> const, Eigen::DSizes<int, 2> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28ˇ!@ˇ!Hˇ!b.gradient_tape/ears_model/concatenate_7/Slice_1h
Ø
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿!@¿!H¿!bmgradient_tape/ears_model/max_pooling2d_68/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
≤
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 5, 256, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿!@¿!H¿!bpgradient_tape/ears_model/average_pooling2d_8/AvgPool/AvgPoolGrad-1-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 4, 512, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿!@¿!H¿!bogradient_tape/ears_model/conv2d_217/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
∂
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28ø!@ø!Hø!Xbears_model/conv2d_211/Conv2Dh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28ø!@ø!Hø!Xbears_model/conv2d_198/Conv2Dh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28†!@†!H†!Xb>gradient_tape/ears_model/conv2d_195/Conv2D/Conv2DBackpropInputh
◊
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Ä!@Ä!HÄ!Xb?gradient_tape/ears_model/conv2d_199/Conv2D/Conv2DBackpropFilterh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä!@Ä!HÄ!b%Adam/Adam/update_40/ResourceApplyAdamh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 8, 256, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä!@Ä!HÄ!bogradient_tape/ears_model/conv2d_213/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
›
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28‡ @‡ H‡ Xb>gradient_tape/ears_model/conv2d_199/Conv2D/Conv2DBackpropInputh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28‡ @‡ H‡ b%Adam/Adam/update_61/ResourceApplyAdamh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28‡ @‡ H‡ bogradient_tape/ears_model/conv2d_206/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
≤
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 512, 4, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28‡ @‡ H‡ bpgradient_tape/ears_model/conv2d_202/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
©
Ovoid scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*28¿ @¿ H¿ Xb?gradient_tape/ears_model/conv2d_201/Conv2D/Conv2DBackpropFilterh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 8, 256, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿ @¿ H¿ bogradient_tape/ears_model/conv2d_215/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 4, 512, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿ @¿ H¿ bogradient_tape/ears_model/conv2d_219/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28† @† H† b$ears_model/leaky_re_lu_159/LeakyReluh
◊
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28† @† H† Xb?gradient_tape/ears_model/conv2d_193/Conv2D/Conv2DBackpropFilterh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28† @† H† b%Adam/Adam/update_18/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28† @† H† b%Adam/Adam/update_63/ResourceApplyAdamh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28† @† H† Xbears_model/conv2d_211/Conv2Dh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28† @† H† Xb>gradient_tape/ears_model/conv2d_216/Conv2D/Conv2DBackpropInputh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28Ä @Ä HÄ b@gradient_tape/ears_model/leaky_re_lu_161/LeakyRelu/LeakyReluGradh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28Ä @Ä HÄ b@gradient_tape/ears_model/leaky_re_lu_165/LeakyRelu/LeakyReluGradh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28Ä @Ä HÄ b@gradient_tape/ears_model/leaky_re_lu_166/LeakyRelu/LeakyReluGradh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28Ä @Ä HÄ b@gradient_tape/ears_model/leaky_re_lu_168/LeakyRelu/LeakyReluGradh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28Ä @Ä HÄ b@gradient_tape/ears_model/leaky_re_lu_170/LeakyRelu/LeakyReluGradh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28Ä @Ä HÄ b@gradient_tape/ears_model/leaky_re_lu_171/LeakyRelu/LeakyReluGradh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28Ä @Ä HÄ b@gradient_tape/ears_model/leaky_re_lu_173/LeakyRelu/LeakyReluGradh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28Ä @Ä HÄ b@gradient_tape/ears_model/leaky_re_lu_174/LeakyRelu/LeakyReluGradh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28Ä @Ä HÄ b@gradient_tape/ears_model/leaky_re_lu_176/LeakyRelu/LeakyReluGradh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28Ä @Ä HÄ b@gradient_tape/ears_model/leaky_re_lu_178/LeakyRelu/LeakyReluGradh
˜	
£	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28Ä @Ä HÄ b:categorical_crossentropy/softmax_cross_entropy_with_logitsh
è
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<long long, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<long long, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28Ä @Ä HÄ bArgMax_1h
—
îvoid cudnn::detail::pooling_fw_4d_kernel<float, float, cudnn::detail::maxpooling_func<float, (cudnnNanPropagation_t)0>, 0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28Ä @Ä HÄ b#ears_model/max_pooling2d_68/MaxPoolh
ª
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Ä @Ä HÄ Xbears_model/conv2d_211/Conv2Dh
›
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Ä @Ä HÄ Xb>gradient_tape/ears_model/conv2d_211/Conv2D/Conv2DBackpropInputh
›
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28Ä @Ä HÄ Xb>gradient_tape/ears_model/conv2d_201/Conv2D/Conv2DBackpropInputh
◊
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Ä @Ä HÄ Xb?gradient_tape/ears_model/conv2d_197/Conv2D/Conv2DBackpropFilterh
◊
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Ä @Ä HÄ Xb?gradient_tape/ears_model/conv2d_213/Conv2D/Conv2DBackpropFilterh
©
Ovoid scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*28Ä @Ä HÄ Xb?gradient_tape/ears_model/conv2d_203/Conv2D/Conv2DBackpropFilterh
¨
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28Ä @Ä HÄ b7gradient_tape/ears_model/conv2d_196/BiasAdd/BiasAddGradh
¨
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28Ä @Ä HÄ b7gradient_tape/ears_model/conv2d_218/BiasAdd/BiasAddGradh
†
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int)*28Ä @Ä HÄ b5gradient_tape/ears_model/dense_12/BiasAdd/BiasAddGradh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28Ä @Ä HÄ bears_model/conv2d_197/BiasAddh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28Ä @Ä HÄ bears_model/conv2d_203/BiasAddh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28Ä @Ä HÄ bears_model/conv2d_209/BiasAddh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28Ä @Ä HÄ bears_model/conv2d_216/BiasAddh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28Ä @Ä HÄ bears_model/conv2d_221/BiasAddh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä @Ä HÄ b%Adam/Adam/update_11/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä @Ä HÄ b%Adam/Adam/update_17/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä @Ä HÄ b%Adam/Adam/update_21/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä @Ä HÄ b%Adam/Adam/update_23/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä @Ä HÄ b%Adam/Adam/update_24/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä @Ä HÄ b%Adam/Adam/update_26/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä @Ä HÄ b%Adam/Adam/update_31/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä @Ä HÄ b%Adam/Adam/update_34/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä @Ä HÄ b%Adam/Adam/update_43/ResourceApplyAdamh
Û
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä @Ä HÄ b$Adam/Adam/update_8/ResourceApplyAdamh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ Xbears_model/conv2d_193/Conv2Dh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ Xbears_model/conv2d_195/Conv2Dh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ Xbears_model/conv2d_200/Conv2Dh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ Xbears_model/conv2d_207/Conv2Dh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ Xbears_model/conv2d_209/Conv2Dh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ Xbears_model/conv2d_218/Conv2Dh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ Xb>gradient_tape/ears_model/conv2d_191/Conv2D/Conv2DBackpropInputh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ Xb?gradient_tape/ears_model/conv2d_193/Conv2D/Conv2DBackpropFilterh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ Xb>gradient_tape/ears_model/conv2d_200/Conv2D/Conv2DBackpropInputh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ Xb>gradient_tape/ears_model/conv2d_207/Conv2D/Conv2DBackpropInputh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ Xb>gradient_tape/ears_model/conv2d_209/Conv2D/Conv2DBackpropInputh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ Xb>gradient_tape/ears_model/conv2d_210/Conv2D/Conv2DBackpropInputh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ Xb>gradient_tape/ears_model/conv2d_211/Conv2D/Conv2DBackpropInputh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ Xb>gradient_tape/ears_model/conv2d_214/Conv2D/Conv2DBackpropInputh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ Xb?gradient_tape/ears_model/conv2d_218/Conv2D/Conv2DBackpropFilterh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ Xb>gradient_tape/ears_model/conv2d_218/Conv2D/Conv2DBackpropInputh
£
övoid tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä @Ä HÄ bogradient_tape/ears_model/conv2d_212/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
ë
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 128, 5, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä @Ä HÄ bOears_model/conv2d_220/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
≤
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 128, 5, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä @Ä HÄ bpgradient_tape/ears_model/conv2d_221/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 5, 128, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä @Ä HÄ bogradient_tape/ears_model/conv2d_220/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
ö
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 256, 5, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä @Ä HÄ bXears_model/average_pooling2d_8/AvgPool-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
ë
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä @Ä HÄ bOears_model/conv2d_194/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä @Ä HÄ bogradient_tape/ears_model/conv2d_211/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä @Ä HÄ bogradient_tape/ears_model/max_pooling2d_69/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 8, 256, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä @Ä HÄ bogradient_tape/ears_model/conv2d_214/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
Ø
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 8, 256, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä @Ä HÄ bmgradient_tape/ears_model/max_pooling2d_72/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
ë
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 512, 4, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä @Ä HÄ bOears_model/conv2d_219/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
≤
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 512, 4, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä @Ä HÄ bpgradient_tape/ears_model/conv2d_218/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
≤
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 512, 4, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä @Ä HÄ bpgradient_tape/ears_model/conv2d_220/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28ˇ@ˇHˇb$ears_model/leaky_re_lu_157/LeakyReluh
›
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28ˇ@ˇHˇXb>gradient_tape/ears_model/conv2d_213/Conv2D/Conv2DBackpropInputh
©
Ovoid scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*28ˇ@ˇHˇXb?gradient_tape/ears_model/conv2d_221/Conv2D/Conv2DBackpropFilterh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28ˇ@ˇHˇb%Adam/Adam/update_53/ResourceApplyAdamh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28ˇ@ˇHˇXbears_model/conv2d_214/Conv2Dh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28ˇ@ˇHˇXbears_model/conv2d_216/Conv2Dh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 9, 128, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28ˇ@ˇHˇbogradient_tape/ears_model/conv2d_204/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
ë
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 512, 4, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28ˇ@ˇHˇbOears_model/conv2d_217/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
≤
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 512, 4, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28ˇ@ˇHˇbpgradient_tape/ears_model/conv2d_204/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28‡@‡H‡Xbears_model/conv2d_191/Conv2Dh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28‡@‡H‡Xb>gradient_tape/ears_model/conv2d_193/Conv2D/Conv2DBackpropInputh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28‡@‡H‡Xb?gradient_tape/ears_model/conv2d_212/Conv2D/Conv2DBackpropFilterh
ë
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 256, 8, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28‡@‡H‡bOears_model/conv2d_215/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 256, 8, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28‡@‡H‡bogradient_tape/ears_model/max_pooling2d_72/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28‡@‡H‡bogradient_tape/ears_model/conv2d_209/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28¿@¿H¿b@gradient_tape/ears_model/leaky_re_lu_172/LeakyRelu/LeakyReluGradh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28¿@¿H¿b%Adam/Adam/update_16/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28¿@¿H¿b%Adam/Adam/update_25/ResourceApplyAdamh
∂
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28†@†H†Xbears_model/conv2d_197/Conv2Dh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28†@†H†Xb?gradient_tape/ears_model/conv2d_194/Conv2D/Conv2DBackpropFilterh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28†@†H†Xb>gradient_tape/ears_model/conv2d_194/Conv2D/Conv2DBackpropInputh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28†@†H†Xb>gradient_tape/ears_model/conv2d_196/Conv2D/Conv2DBackpropInputh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28†@†H†Xb?gradient_tape/ears_model/conv2d_216/Conv2D/Conv2DBackpropFilterh
≤
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28†@†H†bpgradient_tape/ears_model/conv2d_199/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28Ä@ÄHÄbears_model/conv2d_211/BiasAddh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä@ÄHÄXb>gradient_tape/ears_model/conv2d_212/Conv2D/Conv2DBackpropInputh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä@ÄHÄXb?gradient_tape/ears_model/conv2d_214/Conv2D/Conv2DBackpropFilterh
Û
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28‡@‡H‡b$Adam/Adam/update_1/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28‡@‡H‡b%Adam/Adam/update_39/ResourceApplyAdamh
Û
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28‡@‡H‡b$Adam/Adam/update_5/ResourceApplyAdamh
—
îvoid cudnn::detail::pooling_fw_4d_kernel<float, float, cudnn::detail::maxpooling_func<float, (cudnnNanPropagation_t)0>, 0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28¿@¿H¿b#ears_model/max_pooling2d_72/MaxPoolh
¨
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28¿@¿H¿b7gradient_tape/ears_model/conv2d_208/BiasAdd/BiasAddGradh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28¿@¿H¿b%Adam/Adam/update_37/ResourceApplyAdamh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28¿@¿H¿Xbears_model/conv2d_212/Conv2Dh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28¿@¿H¿Xb?gradient_tape/ears_model/conv2d_196/Conv2D/Conv2DBackpropFilterh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28¿@¿H¿Xb>gradient_tape/ears_model/conv2d_198/Conv2D/Conv2DBackpropInputh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28¿@¿H¿Xb?gradient_tape/ears_model/conv2d_200/Conv2D/Conv2DBackpropFilterh
≤
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 256, 8, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿@¿H¿bpgradient_tape/ears_model/conv2d_214/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
ë
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿@¿H¿bOears_model/conv2d_207/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
ë
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿@¿H¿bOears_model/conv2d_210/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28†@†H†b$ears_model/leaky_re_lu_154/LeakyReluh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28†@†H†b$ears_model/leaky_re_lu_174/LeakyReluh
—
îvoid cudnn::detail::pooling_fw_4d_kernel<float, float, cudnn::detail::maxpooling_func<float, (cudnnNanPropagation_t)0>, 0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28†@†H†b#ears_model/max_pooling2d_71/MaxPoolh
¨
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28†@†H†b7gradient_tape/ears_model/conv2d_207/BiasAdd/BiasAddGradh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28†@†H†b%Adam/Adam/update_15/ResourceApplyAdamh
£
övoid tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28†@†H†bogradient_tape/ears_model/conv2d_214/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
ë
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 256, 8, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28†@†H†bOears_model/conv2d_213/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28Ä@ÄHÄbears_model/conv2d_192/BiasAddh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä@ÄHÄb%Adam/Adam/update_41/ResourceApplyAdamh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä@ÄHÄXbears_model/conv2d_192/Conv2Dh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä@ÄHÄXb?gradient_tape/ears_model/conv2d_191/Conv2D/Conv2DBackpropFilterh
£
övoid tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä@ÄHÄbogradient_tape/ears_model/conv2d_200/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
ë
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä@ÄHÄbOears_model/conv2d_198/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä@ÄHÄbogradient_tape/ears_model/conv2d_209/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
Ø
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä@ÄHÄbmgradient_tape/ears_model/max_pooling2d_69/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
§
övoid tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28ˇ@ˇHˇbpgradient_tape/ears_model/conv2d_213/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28‡@‡H‡b@gradient_tape/ears_model/leaky_re_lu_175/LeakyRelu/LeakyReluGradh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28‡@‡H‡b@gradient_tape/ears_model/leaky_re_lu_177/LeakyRelu/LeakyReluGradh
—
îvoid cudnn::detail::pooling_fw_4d_kernel<float, float, cudnn::detail::maxpooling_func<float, (cudnnNanPropagation_t)0>, 0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28‡@‡H‡b#ears_model/max_pooling2d_70/MaxPoolh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28‡@‡H‡b%Adam/Adam/update_10/ResourceApplyAdamh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28‡@‡H‡Xb?gradient_tape/ears_model/conv2d_195/Conv2D/Conv2DBackpropFilterh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28‡@‡H‡Xb?gradient_tape/ears_model/conv2d_209/Conv2D/Conv2DBackpropFilterh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28‡@‡H‡Xb?gradient_tape/ears_model/conv2d_211/Conv2D/Conv2DBackpropFilterh
≤
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28‡@‡H‡bpgradient_tape/ears_model/conv2d_209/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28¿@¿H¿b%Adam/Adam/update_19/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28¿@¿H¿b%Adam/Adam/update_29/ResourceApplyAdamh
Û
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28¿@¿H¿b$Adam/Adam/update_9/ResourceApplyAdamh
§
övoid tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿@¿H¿bpgradient_tape/ears_model/conv2d_217/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
∞
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿@¿H¿bngradient_tape/ears_model/conv2d_207/Conv2D/Conv2DBackpropInput-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
˜
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_pow_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_pow_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28†@†H†b
Adam/Pow_1h
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28†@†H†b@gradient_tape/ears_model/leaky_re_lu_156/LeakyRelu/LeakyReluGradh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28†@†H†b@gradient_tape/ears_model/leaky_re_lu_164/LeakyRelu/LeakyReluGradh
—
îvoid cudnn::detail::pooling_fw_4d_kernel<float, float, cudnn::detail::maxpooling_func<float, (cudnnNanPropagation_t)0>, 0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28†@†H†b#ears_model/max_pooling2d_69/MaxPoolh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28†@†H†b%Adam/Adam/update_57/ResourceApplyAdamh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28†@†H†Xbears_model/conv2d_194/Conv2Dh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28†@†H†Xbears_model/conv2d_210/Conv2Dh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28†@†H†Xb?gradient_tape/ears_model/conv2d_198/Conv2D/Conv2DBackpropFilterh
§
övoid tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28†@†H†bpgradient_tape/ears_model/conv2d_201/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
§
övoid tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28†@†H†bpgradient_tape/ears_model/conv2d_219/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
ë
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28†@†H†bOears_model/conv2d_192/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28ü@üHüb%Adam/Adam/update_49/ResourceApplyAdamh
©
Ocudnn::gemm::computeWgradSplitKOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28Ä@ÄHÄXb?gradient_tape/ears_model/conv2d_201/Conv2D/Conv2DBackpropFilterh
à
ﬂvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbears_model/Casth
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄb$ears_model/leaky_re_lu_152/LeakyReluh
◊
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)*28Ä@ÄHÄXb>gradient_tape/ears_model/conv2d_191/Conv2D/Conv2DBackpropInputh
Ò
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä@ÄHÄb"Adam/Adam/update/ResourceApplyAdamh
Û
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä@ÄHÄb$Adam/Adam/update_3/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä@ÄHÄb%Adam/Adam/update_33/ResourceApplyAdamh
˜
¶void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Ä@ÄHÄb7gradient_tape/ears_model/conv2d_193/BiasAdd/BiasAddGradh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä@ÄHÄXbears_model/conv2d_208/Conv2Dh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28‡@‡H‡b$ears_model/leaky_re_lu_179/LeakyReluh
ë
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28‡@‡H‡bOears_model/conv2d_206/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28‡@‡H‡bogradient_tape/ears_model/conv2d_193/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28‡@‡H‡bogradient_tape/ears_model/conv2d_194/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
≤
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28‡@‡H‡bpgradient_tape/ears_model/conv2d_197/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28ﬂ@ﬂHﬂXb?gradient_tape/ears_model/conv2d_207/Conv2D/Conv2DBackpropFilterh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28¿@¿H¿b$ears_model/leaky_re_lu_156/LeakyReluh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28¿@¿H¿b$ears_model/leaky_re_lu_167/LeakyReluh
˜
¶void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28¿@¿H¿b7gradient_tape/ears_model/conv2d_199/BiasAdd/BiasAddGradh
≤
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿@¿H¿bpgradient_tape/ears_model/conv2d_193/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿@¿H¿bogradient_tape/ears_model/conv2d_198/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
ÿ
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28†@†H†b;gradient_tape/categorical_crossentropy/weighted_loss/Tile_1h
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28†@†H†b$ears_model/leaky_re_lu_165/LeakyReluh
©
Ovoid scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*28Ä@ÄHÄXb?gradient_tape/ears_model/conv2d_204/Conv2D/Conv2DBackpropFilterh
˜
¶void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Ä@ÄHÄb7gradient_tape/ears_model/conv2d_201/BiasAdd/BiasAddGradh
É
övoid tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä@ÄHÄbOears_model/conv2d_214/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28‡@‡H‡b%Adam/Adam/update_65/ResourceApplyAdamh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28¿@¿H¿Xbears_model/conv2d_219/Conv2Dh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28¿@¿H¿b@gradient_tape/ears_model/leaky_re_lu_162/LeakyRelu/LeakyReluGradh
¬
ãvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long const, long long const>, Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long const, long long const>, Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28¿@¿H¿bAdam/Adam/AssignAddVariableOph
É
övoid tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿@¿H¿bOears_model/conv2d_216/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
É
övoid tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿@¿H¿bOears_model/conv2d_218/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
§
övoid tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿@¿H¿bpgradient_tape/ears_model/conv2d_215/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
ã
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28†@†H†b,categorical_crossentropy/weighted_loss/valueh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28†@†H†b$ears_model/leaky_re_lu_172/LeakyReluh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28†@†H†b@gradient_tape/ears_model/leaky_re_lu_154/LeakyRelu/LeakyReluGradh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28†@†H†b@gradient_tape/ears_model/leaky_re_lu_158/LeakyRelu/LeakyReluGradh
¨
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28†@†H†b7gradient_tape/ears_model/conv2d_214/BiasAdd/BiasAddGradh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28†@†H†bears_model/conv2d_218/BiasAddh
≤
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28ü@üHübpgradient_tape/ears_model/conv2d_210/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄb@gradient_tape/ears_model/leaky_re_lu_152/LeakyRelu/LeakyReluGradh
ë
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä@ÄHÄbOears_model/conv2d_211/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
∞
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä@ÄHÄbngradient_tape/ears_model/conv2d_192/Conv2D/Conv2DBackpropInput-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä@ÄHÄbogradient_tape/ears_model/conv2d_210/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
¨
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28ˇ@ˇHˇb7gradient_tape/ears_model/conv2d_212/BiasAdd/BiasAddGradh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28‡@‡H‡Xbears_model/conv2d_201/Conv2Dh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28‡@‡H‡b$ears_model/leaky_re_lu_170/LeakyReluh
ç
Ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<long long, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<long long, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28‡@‡H‡bArgMaxh
©
Ovoid scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*28‡@‡H‡Xb?gradient_tape/ears_model/conv2d_200/Conv2D/Conv2DBackpropFilterh
ü
Fcudnn::gemm::computeBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28¿@¿H¿Xb>gradient_tape/ears_model/conv2d_216/Conv2D/Conv2DBackpropInputh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28ø@øHøbears_model/conv2d_198/BiasAddh
ù
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28†@†H†Xb>gradient_tape/ears_model/conv2d_194/Conv2D/Conv2DBackpropInputh
•
Kcudnn::gemm::computeWgradBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28†@†H†Xb?gradient_tape/ears_model/conv2d_202/Conv2D/Conv2DBackpropFilterh
Ò
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28†@†H†bCasth
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28†@†H†b$ears_model/leaky_re_lu_160/LeakyReluh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28†@†H†b$ears_model/leaky_re_lu_161/LeakyReluh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28†@†H†b$ears_model/leaky_re_lu_162/LeakyReluh
›
âvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::MaxReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::MaxReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)*28†@†H†b:categorical_crossentropy/softmax_cross_entropy_with_logitsh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28†@†H†Xb>gradient_tape/ears_model/conv2d_192/Conv2D/Conv2DBackpropInputh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28†@†H†Xb?gradient_tape/ears_model/conv2d_206/Conv2D/Conv2DBackpropFilterh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28†@†H†Xb?gradient_tape/ears_model/conv2d_208/Conv2D/Conv2DBackpropFilterh
Ñ
ﬂvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ü@üHübAdam/Cast_1h
ù
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28Ä@ÄHÄXb>gradient_tape/ears_model/conv2d_208/Conv2D/Conv2DBackpropInputh
í
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAssignAddVariableOp_3h
©
Ovoid scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*28Ä@ÄHÄXb?gradient_tape/ears_model/conv2d_205/Conv2D/Conv2DBackpropFilterh
É
övoid tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä@ÄHÄbOears_model/conv2d_212/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä@ÄHÄbogradient_tape/ears_model/conv2d_196/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
ù
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28¿@¿H¿Xb>gradient_tape/ears_model/conv2d_210/Conv2D/Conv2DBackpropInputh
∆
ëvoid tensorflow::(anonymous namespace)::GenerateNormalizedProb<float, float, 4>(float const*, float const*, float const*, float*, int, int, bool)*28†@†H†bears_model/dense_12/Softmaxh
˜
¶void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28†@†H†b7gradient_tape/ears_model/conv2d_205/BiasAdd/BiasAddGradh
ü
Fcudnn::gemm::computeBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28Ä@ÄHÄXb>gradient_tape/ears_model/conv2d_202/Conv2D/Conv2DBackpropInputh
©
Ocudnn::gemm::computeWgradSplitKOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28Ä@ÄHÄXb?gradient_tape/ears_model/conv2d_202/Conv2D/Conv2DBackpropFilterh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄb$ears_model/leaky_re_lu_169/LeakyReluh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28Ä@ÄHÄbears_model/conv2d_194/BiasAddh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28ˇ@ˇHˇXbears_model/conv2d_203/Conv2Dh
ù
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28ˇ@ˇHˇXb>gradient_tape/ears_model/conv2d_200/Conv2D/Conv2DBackpropInputh
®
Ovoid scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*28‡@‡H‡Xb>gradient_tape/ears_model/conv2d_205/Conv2D/Conv2DBackpropInputh
à
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*28‡@‡H‡bears_model/dense_12/BiasAddh
ù
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28¿@¿H¿Xb>gradient_tape/ears_model/conv2d_216/Conv2D/Conv2DBackpropInputh
¢
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*28¿@¿H¿Xb?gradient_tape/ears_model/conv2d_192/Conv2D/Conv2DBackpropFilterh
£
övoid tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿@¿H¿bogradient_tape/ears_model/conv2d_219/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿@¿H¿bogradient_tape/ears_model/conv2d_208/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿@¿H¿bogradient_tape/ears_model/conv2d_211/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
ü
Fcudnn::gemm::computeBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28†@†H†Xb>gradient_tape/ears_model/conv2d_220/Conv2D/Conv2DBackpropInputh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28†@†H†b$ears_model/leaky_re_lu_158/LeakyReluh
ù
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28Ä@ÄHÄXb>gradient_tape/ears_model/conv2d_212/Conv2D/Conv2DBackpropInputh
•
Kcudnn::gemm::computeWgradBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28Ä@ÄHÄXb?gradient_tape/ears_model/conv2d_203/Conv2D/Conv2DBackpropFilterh
∂
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Ä@ÄHÄXbears_model/conv2d_199/Conv2Dh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä@ÄHÄb%Adam/Adam/update_13/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä@ÄHÄb%Adam/Adam/update_35/ResourceApplyAdamh
É
övoid tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä@ÄHÄbOears_model/conv2d_200/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28ˇ@ˇHˇXbears_model/conv2d_207/Conv2Dh
•
—void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long)*28‡@‡H‡b:categorical_crossentropy/softmax_cross_entropy_with_logitsh
¨
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28‡@‡H‡b7gradient_tape/ears_model/conv2d_198/BiasAdd/BiasAddGradh
¨
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28‡@‡H‡b7gradient_tape/ears_model/conv2d_210/BiasAdd/BiasAddGradh
Û
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28‡@‡H‡b$Adam/Adam/update_7/ResourceApplyAdamh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28‡@‡H‡Xbears_model/conv2d_196/Conv2Dh
£
övoid tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28‡@‡H‡bogradient_tape/ears_model/conv2d_215/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
≤
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28‡@‡H‡bpgradient_tape/ears_model/conv2d_211/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
É
övoid tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28ﬂ@ﬂHﬂbOears_model/conv2d_202/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿@¿H¿bogradient_tape/ears_model/max_pooling2d_71/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
Ø
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿@¿H¿bmgradient_tape/ears_model/max_pooling2d_71/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28†@†H†b%Adam/Adam/update_27/ResourceApplyAdamh
£
övoid tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28†@†H†bogradient_tape/ears_model/conv2d_217/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄb@gradient_tape/ears_model/leaky_re_lu_179/LeakyRelu/LeakyReluGradh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä@ÄHÄXb?gradient_tape/ears_model/conv2d_210/Conv2D/Conv2DBackpropFilterh
≤
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä@ÄHÄbpgradient_tape/ears_model/conv2d_195/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
˜
¶void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28‡@‡H‡b7gradient_tape/ears_model/conv2d_195/BiasAdd/BiasAddGradh
£
övoid tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28‡@‡H‡bogradient_tape/ears_model/conv2d_202/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
£
övoid tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28‡@‡H‡bogradient_tape/ears_model/conv2d_216/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
¯
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28¿@¿H¿bMulh
›
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28¿@¿H¿Xb>gradient_tape/ears_model/conv2d_209/Conv2D/Conv2DBackpropInputh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28¿@¿H¿b%Adam/Adam/update_59/ResourceApplyAdamh
˜
¶void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28¿@¿H¿b7gradient_tape/ears_model/conv2d_203/BiasAdd/BiasAddGradh
˙
≈void tensorflow::functor::RowReduceKernel<cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long>, float*, cub::Sum>(cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long>, float*, int, int, cub::Sum, std::iterator_traits<cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long> >::value_type)*28¿@¿H¿bears_model/dense_12/Softmaxh
¡
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int)*28†@†H†bLgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mulh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28†@†H†bears_model/conv2d_207/BiasAddh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28†@†H†Xbears_model/conv2d_206/Conv2Dh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 5, 128, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28†@†H†bogradient_tape/ears_model/conv2d_221/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28†@†H†bogradient_tape/ears_model/conv2d_199/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
˜
¶void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Ä@ÄHÄb7gradient_tape/ears_model/conv2d_197/BiasAdd/BiasAddGradh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä@ÄHÄbogradient_tape/ears_model/conv2d_195/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
ë
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28ˇ@ˇHˇbOears_model/conv2d_209/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
◊
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)*28‡@‡H‡Xb>gradient_tape/ears_model/conv2d_195/Conv2D/Conv2DBackpropInputh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28‡@‡H‡b%Adam/Adam/update_51/ResourceApplyAdamh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 9, 128, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28‡@‡H‡bogradient_tape/ears_model/conv2d_205/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
©
Ocudnn::gemm::computeWgradSplitKOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28¿@¿H¿Xb?gradient_tape/ears_model/conv2d_203/Conv2D/Conv2DBackpropFilterh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28¿@¿H¿b$ears_model/leaky_re_lu_171/LeakyReluh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28¿@¿H¿b$ears_model/leaky_re_lu_173/LeakyReluh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28¿@¿H¿bears_model/conv2d_217/BiasAddh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28¿@¿H¿Xb?gradient_tape/ears_model/conv2d_190/Conv2D/Conv2DBackpropFilterh
ë
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 128, 9, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿@¿H¿bOears_model/conv2d_204/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28ø@øHøbears_model/conv2d_201/BiasAddh
Ω
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28ø@øHøXbears_model/conv2d_190/Conv2Dh
ª
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28†@†H†Xbears_model/conv2d_209/Conv2Dh
æ
àvoid splitKreduce_kernel<float, float, float>(cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*)*28†@†H†Xbears_model/dense_12/MatMulh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28‡@‡H‡b$ears_model/leaky_re_lu_164/LeakyReluh
ñ
ﬂvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28¿@¿H¿bcategorical_crossentropy/Casth
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28¿@¿H¿bears_model/conv2d_220/BiasAddh
£
övoid tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿@¿H¿bogradient_tape/ears_model/conv2d_213/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
ë
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿@¿H¿bOears_model/conv2d_208/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿@¿H¿bogradient_tape/ears_model/max_pooling2d_70/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28†@†H†Xbears_model/conv2d_192/Conv2Dh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28†@†H†Xbears_model/conv2d_202/Conv2Dh
ù
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28†@†H†Xb>gradient_tape/ears_model/conv2d_214/Conv2D/Conv2DBackpropInputh
•
Kcudnn::gemm::computeWgradBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28†@†H†Xb?gradient_tape/ears_model/conv2d_204/Conv2D/Conv2DBackpropFilterh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28†@†H†b$ears_model/leaky_re_lu_175/LeakyReluh
·
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*28†@†H†bSum_2h
Ü
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*28†@†H†b*categorical_crossentropy/weighted_loss/Sumh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28†@†H†bogradient_tape/ears_model/conv2d_210/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28ü@üHübears_model/conv2d_208/BiasAddh
ù
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28Ä@ÄHÄXb>gradient_tape/ears_model/conv2d_202/Conv2D/Conv2DBackpropInputh
ù
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28Ä@ÄHÄXb>gradient_tape/ears_model/conv2d_204/Conv2D/Conv2DBackpropInputh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄb$ears_model/leaky_re_lu_176/LeakyReluh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄb@gradient_tape/ears_model/leaky_re_lu_167/LeakyRelu/LeakyReluGradh
£
övoid tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä@ÄHÄbogradient_tape/ears_model/conv2d_201/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
ù
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28ˇ@ˇHˇXb>gradient_tape/ears_model/conv2d_192/Conv2D/Conv2DBackpropInputh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28‡@‡H‡Xbears_model/conv2d_190/Conv2Dh
ù
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28‡@‡H‡Xb>gradient_tape/ears_model/conv2d_198/Conv2D/Conv2DBackpropInputh
©
Ocudnn::gemm::computeWgradSplitKOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28‡@‡H‡Xb?gradient_tape/ears_model/conv2d_200/Conv2D/Conv2DBackpropFilterh
©
Ocudnn::gemm::computeWgradSplitKOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28‡@‡H‡Xb?gradient_tape/ears_model/conv2d_221/Conv2D/Conv2DBackpropFilterh
•
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28‡@‡H‡b8categorical_crossentropy/weighted_loss/num_elements/Casth
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28‡@‡H‡b@gradient_tape/ears_model/leaky_re_lu_160/LeakyRelu/LeakyReluGradh
œ
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28‡@‡H‡b@gradient_tape/ears_model/leaky_re_lu_169/LeakyRelu/LeakyReluGradh
ë
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28‡@‡H‡bOears_model/conv2d_196/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
ù
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28¿@¿H¿Xb>gradient_tape/ears_model/conv2d_196/Conv2D/Conv2DBackpropInputh
•
Kcudnn::gemm::computeWgradBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28¿@¿H¿Xb?gradient_tape/ears_model/conv2d_220/Conv2D/Conv2DBackpropFilterh
•
Kcudnn::gemm::computeWgradBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28¿@¿H¿Xb?gradient_tape/ears_model/conv2d_221/Conv2D/Conv2DBackpropFilterh
©
Ocudnn::gemm::computeWgradSplitKOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28¿@¿H¿Xb?gradient_tape/ears_model/conv2d_205/Conv2D/Conv2DBackpropFilterh
ı
’void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28¿@¿H¿bCast_2h
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28¿@¿H¿bears_model/conv2d_213/BiasAddh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿@¿H¿bogradient_tape/ears_model/conv2d_197/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
ü
Fcudnn::gemm::computeBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28†@†H†Xb>gradient_tape/ears_model/conv2d_214/Conv2D/Conv2DBackpropInputh
ü
Fcudnn::gemm::computeBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28†@†H†Xb>gradient_tape/ears_model/conv2d_218/Conv2D/Conv2DBackpropInputh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28†@†H†Xbears_model/conv2d_217/Conv2Dh
Û
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28†@†H†bCast_3h
£
övoid tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28†@†H†bogradient_tape/ears_model/conv2d_203/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
ù
˚void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long, long long>, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long, long long>, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAdam/addh
›
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Ä@ÄHÄXb>gradient_tape/ears_model/conv2d_207/Conv2D/Conv2DBackpropInputh
©
Ovoid scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*28Ä@ÄHÄXb?gradient_tape/ears_model/conv2d_220/Conv2D/Conv2DBackpropFilterh
Ø
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä@ÄHÄbmgradient_tape/ears_model/max_pooling2d_70/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28‡@‡H‡Xbears_model/conv2d_216/Conv2Dh
ê
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28‡@‡H‡bAssignAddVariableOph
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28‡@‡H‡bears_model/conv2d_202/BiasAddh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28‡@‡H‡bears_model/conv2d_206/BiasAddh
∫
`compute_gemm_pointers(float2**, float2 const*, int, float2 const*, int, float2 const*, int, int)*28¿@¿H¿Xb?gradient_tape/ears_model/conv2d_192/Conv2D/Conv2DBackpropFilterh
ü
Fcudnn::gemm::computeBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28¿@¿H¿Xb>gradient_tape/ears_model/conv2d_194/Conv2D/Conv2DBackpropInputh
ü
Fcudnn::gemm::computeBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28¿@¿H¿Xb>gradient_tape/ears_model/conv2d_204/Conv2D/Conv2DBackpropInputh
•
Kcudnn::gemm::computeWgradBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28¿@¿H¿Xb?gradient_tape/ears_model/conv2d_205/Conv2D/Conv2DBackpropFilterh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28¿@¿H¿bears_model/conv2d_212/BiasAddh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28†@†H†b$ears_model/leaky_re_lu_163/LeakyReluh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28†@†H†bears_model/conv2d_200/BiasAddh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28Ä@ÄHÄXbears_model/conv2d_193/Conv2Dh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28Ä@ÄHÄbears_model/conv2d_219/BiasAddh
í
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28‡@‡H‡bAssignAddVariableOp_2h
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28‡@‡H‡bears_model/conv2d_215/BiasAddh
£
övoid tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28‡@‡H‡bogradient_tape/ears_model/conv2d_218/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh

—void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::equal_to<long long>, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::equal_to<long long>, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28¿@¿H¿bEqualh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28†@†H†Xbears_model/conv2d_212/Conv2Dh
È
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28†@†H†b
div_no_nanh
§
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28†@†H†bEgradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nanh
Î
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ü@üHübdiv_no_nan_1h
ü
Fcudnn::gemm::computeBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28Ä@ÄHÄXb>gradient_tape/ears_model/conv2d_192/Conv2D/Conv2DBackpropInputh
◊
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)*28Ä@ÄHÄXb>gradient_tape/ears_model/conv2d_193/Conv2D/Conv2DBackpropInputh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28Ä@ÄHÄbears_model/conv2d_204/BiasAddh
Ô
õvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long)*28‡@‡H‡b:categorical_crossentropy/softmax_cross_entropy_with_logitsh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28‡@‡H‡bears_model/conv2d_214/BiasAddh
∫
`compute_gemm_pointers(float2**, float2 const*, int, float2 const*, int, float2 const*, int, int)*28¿@¿H¿Xb?gradient_tape/ears_model/conv2d_191/Conv2D/Conv2DBackpropFilterh
•
Kcudnn::gemm::computeWgradBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28¿@¿H¿Xb?gradient_tape/ears_model/conv2d_200/Conv2D/Conv2DBackpropFilterh
Â
¡void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_boolean_and_op, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_boolean_and_op, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28¿@¿H¿b
LogicalAndh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28¿@¿H¿bears_model/conv2d_196/BiasAddh
ÿ
£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Max>(float const*, float*, int, int, cub::Max, std::iterator_traits<float const*>::value_type)*28¿@¿H¿bears_model/dense_12/Softmaxh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28†@†H†Xb>gradient_tape/ears_model/conv2d_208/Conv2D/Conv2DBackpropInputh
ü
Fcudnn::gemm::computeBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28Ä@ÄHÄXb>gradient_tape/ears_model/conv2d_196/Conv2D/Conv2DBackpropInputh
ü
Fcudnn::gemm::computeBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28Ä@ÄHÄXb>gradient_tape/ears_model/conv2d_198/Conv2D/Conv2DBackpropInputh
ü
Fcudnn::gemm::computeBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28Ä@ÄHÄXb>gradient_tape/ears_model/conv2d_200/Conv2D/Conv2DBackpropInputh
ü
Fcudnn::gemm::computeBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28Ä@ÄHÄXb>gradient_tape/ears_model/conv2d_208/Conv2D/Conv2DBackpropInputh
ü
Fcudnn::gemm::computeBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28Ä@ÄHÄXb>gradient_tape/ears_model/conv2d_210/Conv2D/Conv2DBackpropInputh
ü
Fcudnn::gemm::computeBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28Ä@ÄHÄXb>gradient_tape/ears_model/conv2d_212/Conv2D/Conv2DBackpropInputh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28Ä@ÄHÄXbears_model/conv2d_194/Conv2Dh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28Ä@ÄHÄXbears_model/conv2d_195/Conv2Dh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28Ä@ÄHÄXbears_model/conv2d_196/Conv2Dh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28Ä@ÄHÄXbears_model/conv2d_200/Conv2Dh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28Ä@ÄHÄXbears_model/conv2d_206/Conv2Dh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28Ä@ÄHÄXbears_model/conv2d_210/Conv2Dh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28Ä@ÄHÄXbears_model/conv2d_214/Conv2Dh
ù
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28Ä@ÄHÄXb>gradient_tape/ears_model/conv2d_218/Conv2D/Conv2DBackpropInputh
ù
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28Ä@ÄHÄXb>gradient_tape/ears_model/conv2d_220/Conv2D/Conv2DBackpropInputh
©
Ocudnn::gemm::computeWgradSplitKOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28Ä@ÄHÄXb?gradient_tape/ears_model/conv2d_204/Conv2D/Conv2DBackpropFilterh
©
Ocudnn::gemm::computeWgradSplitKOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28Ä@ÄHÄXb?gradient_tape/ears_model/conv2d_220/Conv2D/Conv2DBackpropFilterh
ı
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_pow_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_pow_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAdam/Powh
í
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAssignAddVariableOp_1h
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄb$ears_model/leaky_re_lu_166/LeakyReluh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄb$ears_model/leaky_re_lu_168/LeakyReluh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄb$ears_model/leaky_re_lu_177/LeakyReluh
≥
ıvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<float const, float const> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄb$ears_model/leaky_re_lu_178/LeakyReluh
∫
ãvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long const, long long const>, Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long const, long long const>, Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAssignAddVariableOp_4h
©
Ovoid scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*28Ä@ÄHÄXb?gradient_tape/ears_model/conv2d_202/Conv2D/Conv2DBackpropFilterh
®
Ovoid scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*28Ä@ÄHÄXb>gradient_tape/ears_model/conv2d_221/Conv2D/Conv2DBackpropInputh
è
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28Ä@ÄHÄbears_model/conv2d_210/BiasAddh
˜
¶void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Ä@ÄHÄb7gradient_tape/ears_model/conv2d_221/BiasAdd/BiasAddGradh
‡
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä@ÄHÄXb?gradient_tape/ears_model/conv2d_192/Conv2D/Conv2DBackpropFilterh
•
Kcudnn::gemm::computeWgradBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28ˇ@ˇHˇXb?gradient_tape/ears_model/conv2d_201/Conv2D/Conv2DBackpropFilterh