
�
�void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, true>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, int, int, float, int, int, int)*28ݘ@ݘHݘXb>gradient_tape/ears_model/conv2d_237/Conv2D/Conv2DBackpropInputh
^
$volta_scudnn_128x64_relu_small_nn_v1*28��
@��
H��
Xbears_model/conv2d_235/Conv2Dh
^
$volta_scudnn_128x64_relu_small_nn_v1*28ݯ
@ݯ
Hݯ
Xbears_model/conv2d_233/Conv2Dh
a
'volta_scudnn_128x64_relu_interior_nn_v1*28އ	@އ	Hއ	Xbears_model/conv2d_234/Conv2Dh
q
volta_sgemm_128x64_nt*28ݏ@ݏHݏXb>gradient_tape/ears_model/conv2d_235/Conv2D/Conv2DBackpropInputh
q
volta_sgemm_128x64_nt*28��@��H��Xb>gradient_tape/ears_model/conv2d_233/Conv2D/Conv2DBackpropInputh
�
;volta_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1*28��@��H��Xb>gradient_tape/ears_model/conv2d_223/Conv2D/Conv2DBackpropInputh
�
;volta_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1*28��@��H��Xb>gradient_tape/ears_model/conv2d_227/Conv2D/Conv2DBackpropInputh
�
;volta_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1*28��@��H��Xb>gradient_tape/ears_model/conv2d_225/Conv2D/Conv2DBackpropInputh
�
/volta_scudnn_128x64_stridedB_splitK_small_nn_v1*28��@��H��Xb?gradient_tape/ears_model/conv2d_233/Conv2D/Conv2DBackpropFilterh
�
/volta_scudnn_128x64_stridedB_splitK_small_nn_v1*28��@��H��Xb?gradient_tape/ears_model/conv2d_235/Conv2D/Conv2DBackpropFilterh
�
�void cudnn::detail::implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, true, false, true>(int, int, int, float const*, int, float*, float*, kernel_conv_params, int, float, float, int, float*, float*, int, int)*28��@��H��Xbears_model/conv2d_223/Conv2Dh
a
'volta_scudnn_128x64_relu_interior_nn_v1*28��@��H��Xbears_model/conv2d_232/Conv2Dh
r
volta_sgemm_128x64_nt*28��@��H��Xb?gradient_tape/ears_model/conv2d_225/Conv2D/Conv2DBackpropFilterh
r
volta_sgemm_128x64_nt*28��@��H��Xb?gradient_tape/ears_model/conv2d_227/Conv2D/Conv2DBackpropFilterh
r
volta_sgemm_128x64_nt*28��@��H��Xb?gradient_tape/ears_model/conv2d_229/Conv2D/Conv2DBackpropFilterh
r
volta_sgemm_128x64_nt*28��@��H��Xb?gradient_tape/ears_model/conv2d_231/Conv2D/Conv2DBackpropFilterh
�
2volta_scudnn_128x64_stridedB_splitK_interior_nn_v1*28��@��H��Xb?gradient_tape/ears_model/conv2d_237/Conv2D/Conv2DBackpropFilterh
q
volta_sgemm_128x64_nt*28��@��H��Xb>gradient_tape/ears_model/conv2d_231/Conv2D/Conv2DBackpropInputh
�
�void pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor)*28��@��H��b=gradient_tape/ears_model/max_pooling2d_76/MaxPool/MaxPoolGradh
q
volta_sgemm_128x64_nt*28��@��H��Xb>gradient_tape/ears_model/conv2d_229/Conv2D/Conv2DBackpropInputh
�
�void cudnn::detail::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, true, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, int, float, int, int, int, int)*28ߠ@ߠHߠXb?gradient_tape/ears_model/conv2d_226/Conv2D/Conv2DBackpropFilterh
�
2volta_scudnn_128x64_stridedB_splitK_interior_nn_v1*28��@��H��Xb?gradient_tape/ears_model/conv2d_236/Conv2D/Conv2DBackpropFilterh
�
�void cudnn::detail::implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, true, false, true>(int, int, int, float const*, int, float*, float*, kernel_conv_params, int, float, float, int, float*, float*, int, int)*28ߟ@ߟHߟXbears_model/conv2d_236/Conv2Dh
�
2volta_scudnn_128x64_stridedB_splitK_interior_nn_v1*28ߓ@ߓHߓXb?gradient_tape/ears_model/conv2d_234/Conv2D/Conv2DBackpropFilterh
�
2volta_scudnn_128x64_stridedB_splitK_interior_nn_v1*28��@��H��Xb?gradient_tape/ears_model/conv2d_232/Conv2D/Conv2DBackpropFilterh
�
�void cudnn::detail::implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, true, false, true>(int, int, int, float const*, int, float*, float*, kernel_conv_params, int, float, float, int, float*, float*, int, int)*28��@��H��Xbears_model/conv2d_237/Conv2Dh
^
$volta_scudnn_128x64_relu_small_nn_v1*28��@��H��Xbears_model/conv2d_225/Conv2Dh
�
�void fft2d_r2c_32x32<float, false, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)*28��@��H��Xb?gradient_tape/ears_model/conv2d_223/Conv2D/Conv2DBackpropFilterh
^
$volta_scudnn_128x64_relu_small_nn_v1*28��@��H��Xbears_model/conv2d_227/Conv2Dh
q
volta_cgemm_32x32_tn*28߱@߱H߱Xb?gradient_tape/ears_model/conv2d_223/Conv2D/Conv2DBackpropFilterh
O
volta_sgemm_128x64_nn*28��@��H��Xbears_model/conv2d_229/Conv2Dh
O
volta_sgemm_128x64_nn*28��@��H��Xbears_model/conv2d_231/Conv2Dh
�
+volta_scudnn_128x64_stridedB_interior_nn_v1*28ߏ@ߏHߏXb>gradient_tape/ears_model/conv2d_236/Conv2D/Conv2DBackpropInputh
a
'volta_scudnn_128x64_relu_interior_nn_v1*28��@��H��Xbears_model/conv2d_226/Conv2Dh
a
'volta_scudnn_128x64_relu_interior_nn_v1*28��@��H��Xbears_model/conv2d_228/Conv2Dh
�
�void pooling_bw_kernel_max_nchw_fully_packed_large<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, int)*28��@��H��b=gradient_tape/ears_model/max_pooling2d_73/MaxPool/MaxPoolGradh
�
�void pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor)*28��@��H��b=gradient_tape/ears_model/max_pooling2d_75/MaxPool/MaxPoolGradh
�
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28��@��H��Xb>gradient_tape/ears_model/conv2d_233/Conv2D/Conv2DBackpropInputh
�
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28��@��H��Xb>gradient_tape/ears_model/conv2d_235/Conv2D/Conv2DBackpropInputh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)*28��@��H��b:categorical_crossentropy/softmax_cross_entropy_with_logitsh
�
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28��@��H��Xb?gradient_tape/ears_model/conv2d_222/Conv2D/Conv2DBackpropFilterh
�
�void cudnn::detail::implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, true, false, true>(int, int, int, float const*, int, float*, float*, kernel_conv_params, int, float, float, int, float*, float*, int, int)*28߷@߷H߷Xbears_model/conv2d_230/Conv2Dh
�
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)*28߷@߷H߷Xb?gradient_tape/ears_model/conv2d_222/Conv2D/Conv2DBackpropFilterh
^
$volta_scudnn_128x32_relu_small_nn_v1*28��@��H��Xbears_model/conv2d_222/Conv2Dh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28��@��H��b%Adam/Adam/update_26/ResourceApplyAdamh
�
�void gemv2N_kernel<int, int, float, float, float, 128, 8, 4, 4, 1, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)*28��@��H��Xb?gradient_tape/ears_model/conv2d_222/Conv2D/Conv2DBackpropFilterh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28��@��H��b%Adam/Adam/update_22/ResourceApplyAdamh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�@�H�Xbears_model/conv2d_233/Conv2Dh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28��@��H��Xbears_model/conv2d_235/Conv2Dh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28��@��H��bogradient_tape/ears_model/max_pooling2d_73/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28��@��H��Xb>gradient_tape/ears_model/conv2d_233/Conv2D/Conv2DBackpropInputh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28ߣ@ߣHߣXb>gradient_tape/ears_model/conv2d_235/Conv2D/Conv2DBackpropInputh
�
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28��@��H��b7gradient_tape/ears_model/conv2d_222/BiasAdd/BiasAddGradh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28��@��H��Xb?gradient_tape/ears_model/conv2d_233/Conv2D/Conv2DBackpropFilterh
�
�
�
+volta_scudnn_128x64_stridedB_interior_nn_v1*28��@��H��Xb>gradient_tape/ears_model/conv2d_234/Conv2D/Conv2DBackpropInputh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28��@��H��Xb?gradient_tape/ears_model/conv2d_235/Conv2D/Conv2DBackpropFilterh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28��@��H��bOears_model/conv2d_222/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�void cudnn::detail::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, true, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, int, float, int, int, int, int)*28ߔ@ߔHߔXb?gradient_tape/ears_model/conv2d_230/Conv2D/Conv2DBackpropFilterh
�
,volta_scudnn_128x128_stridedB_interior_nn_v1*28��@��H��Xb>gradient_tape/ears_model/conv2d_228/Conv2D/Conv2DBackpropInputh
�
,volta_scudnn_128x128_stridedB_interior_nn_v1*28��@��H��Xb>gradient_tape/ears_model/conv2d_230/Conv2D/Conv2DBackpropInputh
�
+volta_scudnn_128x64_stridedB_interior_nn_v1*28��@��H��Xb>gradient_tape/ears_model/conv2d_232/Conv2D/Conv2DBackpropInputh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28��@��H��bogradient_tape/ears_model/conv2d_222/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28��@��H��bmgradient_tape/ears_model/max_pooling2d_73/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)*28��@��H��Xb?gradient_tape/ears_model/conv2d_223/Conv2D/Conv2DBackpropFilterh
�
�void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)*28��@��H��Xb?gradient_tape/ears_model/conv2d_229/Conv2D/Conv2DBackpropFilterh
�
�void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)*28��@��H��Xb?gradient_tape/ears_model/conv2d_231/Conv2D/Conv2DBackpropFilterh
�
�void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28��@��H��Xb>gradient_tape/ears_model/conv2d_233/Conv2D/Conv2DBackpropInputh
�
�void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28��@��H��Xb>gradient_tape/ears_model/conv2d_235/Conv2D/Conv2DBackpropInputh
�
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*28��@��H��Xb?gradient_tape/ears_model/conv2d_223/Conv2D/Conv2DBackpropFilterh
�
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)*28��@�[H��Xb?gradient_tape/ears_model/conv2d_224/Conv2D/Conv2DBackpropFilterh
�
�void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)*28��@��H��Xb?gradient_tape/ears_model/conv2d_225/Conv2D/Conv2DBackpropFilterh
�
�void pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor)*28��@��H��b=gradient_tape/ears_model/max_pooling2d_74/MaxPool/MaxPoolGradh
�
�void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)*28��@��H��Xb?gradient_tape/ears_model/conv2d_227/Conv2D/Conv2DBackpropFilterh
a
'volta_scudnn_128x32_relu_interior_nn_v1*28��@��H��Xbears_model/conv2d_224/Conv2Dh
�
�void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)*28��@��H��Xb?gradient_tape/ears_model/conv2d_222/Conv2D/Conv2DBackpropFilterh
V
volta_sgemm_32x32_sliced1x4_nn*28��@��H��Xbears_model/dense_13/MatMulh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28��@��H��Xbears_model/conv2d_237/Conv2Dh
�
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28��@��H��Xb>gradient_tape/ears_model/conv2d_231/Conv2D/Conv2DBackpropInputh
�
+volta_scudnn_128x64_stridedB_interior_nn_v1*28��@��H��Xb>gradient_tape/ears_model/conv2d_226/Conv2D/Conv2DBackpropInputh
�
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28��@��H��Xb>gradient_tape/ears_model/conv2d_229/Conv2D/Conv2DBackpropInputh
�
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28��@��H��b7gradient_tape/ears_model/conv2d_223/BiasAdd/BiasAddGradh
�
�void cudnn::detail::wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, true, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, int, float, int, int, int, int)*28��@��H��Xb?gradient_tape/ears_model/conv2d_228/Conv2D/Conv2DBackpropFilterh
�
�
�
�void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28��@��H��Xbears_model/conv2d_231/Conv2Dh
�
+volta_scudnn_128x64_stridedB_interior_nn_v1*28��@��H��Xb>gradient_tape/ears_model/conv2d_224/Conv2D/Conv2DBackpropInputh
�
�void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28�@�H�b7gradient_tape/ears_model/conv2d_237/BiasAdd/BiasAddGradh
�
�void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28��@��H��Xbears_model/conv2d_229/Conv2Dh
�
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28��@��H��bears_model/conv2d_222/BiasAddh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28��@��H��Xb?gradient_tape/ears_model/conv2d_237/Conv2D/Conv2DBackpropFilterh
q
volta_cgemm_32x32_tn*28��@��H��Xb?gradient_tape/ears_model/conv2d_224/Conv2D/Conv2DBackpropFilterh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)*28��@��H��b:categorical_crossentropy/softmax_cross_entropy_with_logitsh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28��@��H��b%Adam/Adam/update_30/ResourceApplyAdamh
�
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)*28��@��H��Xb?gradient_tape/ears_model/conv2d_225/Conv2D/Conv2DBackpropFilterh
�
�void cudnn::detail::pooling_bw_kernel_avg<float, float, cudnn::detail::averpooling_func<float>, 2, false>(cudnnTensorStruct, float const*, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28��@��H��b@gradient_tape/ears_model/average_pooling2d_9/AvgPool/AvgPoolGradh
�
�
�
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)*28��@��H��Xb?gradient_tape/ears_model/conv2d_227/Conv2D/Conv2DBackpropFilterh
�
�void cudnn::detail::pooling_fw_4d_kernel<float, float, cudnn::detail::maxpooling_func<float, (cudnnNanPropagation_t)0>, 0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28��@��H��b#ears_model/max_pooling2d_73/MaxPoolh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*28��@� H�dbears_model/concatenate_8/concath
[
volta_sgemm_32x128_nt*28��@��H��b*gradient_tape/ears_model/dense_13/MatMul_1h
�
�
�
�
�
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28�~@�~H�~b7gradient_tape/ears_model/conv2d_236/BiasAdd/BiasAddGradh
�
�void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)*28�~@�~H�~Xb?gradient_tape/ears_model/conv2d_224/Conv2D/Conv2DBackpropFilterh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�z@�zH�zXbears_model/conv2d_236/Conv2Dh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�y@�yH�yXb>gradient_tape/ears_model/conv2d_236/Conv2D/Conv2DBackpropInputh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�y@�yH�yXb>gradient_tape/ears_model/conv2d_237/Conv2D/Conv2DBackpropInputh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�x@�xH�xXbears_model/conv2d_231/Conv2Dh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�x@�xH�xXbears_model/conv2d_229/Conv2Dh
�
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)*28�w@�wH�wXb?gradient_tape/ears_model/conv2d_229/Conv2D/Conv2DBackpropFilterh
�
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)*28�w@�wH�wXb?gradient_tape/ears_model/conv2d_231/Conv2D/Conv2DBackpropFilterh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�s@�sH�sXb>gradient_tape/ears_model/conv2d_231/Conv2D/Conv2DBackpropInputh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�s@�sH�sbngradient_tape/ears_model/conv2d_225/Conv2D/Conv2DBackpropInput-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�r@�rH�rb%Adam/Adam/update_14/ResourceApplyAdamh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�q@�qH�qb%Adam/Adam/update_18/ResourceApplyAdamh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�q@�qH�qbogradient_tape/ears_model/max_pooling2d_74/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�q@�qH�qbogradient_tape/ears_model/max_pooling2d_75/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28�p@�pH�pb7gradient_tape/ears_model/conv2d_225/BiasAdd/BiasAddGradh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�p@�pH�pbmgradient_tape/ears_model/max_pooling2d_75/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�p@�pH�pXb>gradient_tape/ears_model/conv2d_229/Conv2D/Conv2DBackpropInputh
�
�void cudnn::detail::pooling_fw_4d_kernel<float, float, cudnn::detail::averpooling_func<float>, 2, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28�o@�oH�ob&ears_model/average_pooling2d_9/AvgPoolh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�o@�oH�oXb?gradient_tape/ears_model/conv2d_231/Conv2D/Conv2DBackpropFilterh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�o@�oH�obpgradient_tape/ears_model/conv2d_226/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
X
volta_sgemm_128x32_tn*28�o@�oH�oXb(gradient_tape/ears_model/dense_13/MatMulh
�
�void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28�l@�lH�lb7gradient_tape/ears_model/conv2d_235/BiasAdd/BiasAddGradh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�l@�lH�lbOears_model/conv2d_225/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28�l@�lH�lb7gradient_tape/ears_model/conv2d_233/BiasAdd/BiasAddGradh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 4, 512, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�l@�lH�lbogradient_tape/ears_model/conv2d_234/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�k@�kH�kbOears_model/conv2d_223/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 4, 512, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�k@�kH�kbogradient_tape/ears_model/conv2d_235/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�j@�jH�jbOears_model/conv2d_227/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�j@�jH�jXb?gradient_tape/ears_model/conv2d_236/Conv2D/Conv2DBackpropFilterh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 4, 512, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�j@�jH�jbogradient_tape/ears_model/conv2d_236/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 4, 512, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�i@�iH�ibogradient_tape/ears_model/conv2d_233/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�h@�hH�hbpgradient_tape/ears_model/conv2d_230/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�h@�hH�hbOears_model/conv2d_231/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 8, 256, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�h@�hH�hbpgradient_tape/ears_model/average_pooling2d_9/AvgPool/AvgPoolGrad-1-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�g@�gH�gXb?gradient_tape/ears_model/conv2d_229/Conv2D/Conv2DBackpropFilterh
�
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28�f@�fH�fbears_model/conv2d_225/BiasAddh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�f@�fH�fbogradient_tape/ears_model/conv2d_227/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�void cudnn::detail::pooling_fw_4d_kernel<float, float, cudnn::detail::maxpooling_func<float, (cudnnNanPropagation_t)0>, 0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28�f@�fH�fb#ears_model/max_pooling2d_75/MaxPoolh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�f@�fH�fbmgradient_tape/ears_model/max_pooling2d_74/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28�e@�eH�eb7gradient_tape/ears_model/conv2d_227/BiasAdd/BiasAddGradh
�
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28�d@�dH�dbears_model/conv2d_227/BiasAddh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�d@�dH�dbOears_model/conv2d_229/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�c@�cH�cbngradient_tape/ears_model/conv2d_223/Conv2D/Conv2DBackpropInput-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�c@�cH�cbogradient_tape/ears_model/conv2d_226/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28�b@�bH�bb7gradient_tape/ears_model/conv2d_224/BiasAdd/BiasAddGradh
�
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28�a@�aH�abears_model/conv2d_223/BiasAddh
�
�void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28�a@�aH�ab7gradient_tape/ears_model/conv2d_231/BiasAdd/BiasAddGradh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�`@�`H�`bogradient_tape/ears_model/max_pooling2d_76/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�void cudnn::detail::pooling_fw_4d_kernel<float, float, cudnn::detail::maxpooling_func<float, (cudnnNanPropagation_t)0>, 0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28�`@�`H�`b#ears_model/max_pooling2d_74/MaxPoolh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 256, 8, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�`@�`H�`bXears_model/average_pooling2d_9/AvgPool-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�
�
�
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�_@�_H�_b%Adam/Adam/update_28/ResourceApplyAdamh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�^@�^H�^bogradient_tape/ears_model/conv2d_230/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28�^@�^H�^b7gradient_tape/ears_model/conv2d_226/BiasAdd/BiasAddGradh
�
�
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�[@�[H�[Xb>gradient_tape/ears_model/conv2d_234/Conv2D/Conv2DBackpropInputh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�Z@�ZH�Zbmgradient_tape/ears_model/max_pooling2d_76/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�Z@�ZH�Zbogradient_tape/ears_model/conv2d_229/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�Z@�ZH�ZXbears_model/conv2d_234/Conv2Dh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�V@�VH�Vbogradient_tape/ears_model/conv2d_231/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28�U@�UH�Ub7gradient_tape/ears_model/conv2d_229/BiasAdd/BiasAddGradh
�
�
�
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28�P@�PH�PXb?gradient_tape/ears_model/conv2d_225/Conv2D/Conv2DBackpropFilterh
�
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28�P@�PH�PXb?gradient_tape/ears_model/conv2d_227/Conv2D/Conv2DBackpropFilterh
�
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28�P@�PH�Pb7gradient_tape/ears_model/conv2d_232/BiasAdd/BiasAddGradh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 512, 4, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�P@�PH�PbOears_model/conv2d_233/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 512, 4, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�O@�OH�Obpgradient_tape/ears_model/conv2d_234/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�
�
�void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28�O@�OH�OXb>gradient_tape/ears_model/conv2d_231/Conv2D/Conv2DBackpropInputh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 512, 4, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�O@�OH�Obpgradient_tape/ears_model/conv2d_236/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28�O@�OH�OXb>gradient_tape/ears_model/conv2d_235/Conv2D/Conv2DBackpropInputh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 512, 4, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�O@�OH�ObOears_model/conv2d_235/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28�O@�OH�OXb>gradient_tape/ears_model/conv2d_233/Conv2D/Conv2DBackpropInputh
�
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28�O@�OH�OXb?gradient_tape/ears_model/conv2d_229/Conv2D/Conv2DBackpropFilterh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�O@�OH�Obpgradient_tape/ears_model/conv2d_225/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28�O@�OH�Ob7gradient_tape/ears_model/conv2d_234/BiasAdd/BiasAddGradh
�
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28�N@�NH�Nbears_model/conv2d_229/BiasAddh
�
�void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28�M@�MH�MXbears_model/conv2d_229/Conv2Dh
�
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28�L@�LH�LXb?gradient_tape/ears_model/conv2d_231/Conv2D/Conv2DBackpropFilterh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�L@�LH�LXb?gradient_tape/ears_model/conv2d_234/Conv2D/Conv2DBackpropFilterh
�
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28�L@�LH�Lbears_model/conv2d_231/BiasAddh
�
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28�L@�LH�Lbears_model/conv2d_237/BiasAddh
�
�void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28�L@�LH�LXb>gradient_tape/ears_model/conv2d_229/Conv2D/Conv2DBackpropInputh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�K@�KH�KXb>gradient_tape/ears_model/conv2d_232/Conv2D/Conv2DBackpropInputh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�J@�JH�Jb%Adam/Adam/update_32/ResourceApplyAdamh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�I@�IH�Ib%Adam/Adam/update_24/ResourceApplyAdamh
�
�
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�H@�HH�Hb$Adam/Adam/update_8/ResourceApplyAdamh
�
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28�H@�HH�HXbears_model/conv2d_229/Conv2Dh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�H@�HH�Hb%Adam/Adam/update_10/ResourceApplyAdamh
�
�void splitKreduce_kernel<float, float, float>(cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*)*28�G@�GH�GXbears_model/dense_13/MatMulh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�G@�GH�Gb$Adam/Adam/update_6/ResourceApplyAdamh
�
�void cudnn::detail::pooling_fw_4d_kernel<float, float, cudnn::detail::maxpooling_func<float, (cudnnNanPropagation_t)0>, 0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28�G@�GH�Gb#ears_model/max_pooling2d_76/MaxPoolh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_quotient_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_quotient_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28�F@�FH�Fb:categorical_crossentropy/softmax_cross_entropy_with_logitsh
�
�void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28�E@�EH�EXbears_model/conv2d_231/Conv2Dh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�D@�DH�Db%Adam/Adam/update_23/ResourceApplyAdamh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�C@�CH�CXbears_model/conv2d_232/Conv2Dh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�C@�CH�Cb$Adam/Adam/update_2/ResourceApplyAdamh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�C@�CH�CXb>gradient_tape/ears_model/conv2d_225/Conv2D/Conv2DBackpropInputh
�
Ovoid scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*28�B@�BH�BXb?gradient_tape/ears_model/conv2d_233/Conv2D/Conv2DBackpropFilterh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�B@�BH�BXbears_model/conv2d_227/Conv2Dh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�B@�BH�Bb%Adam/Adam/update_27/ResourceApplyAdamh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�A@�AH�Ab%Adam/Adam/update_20/ResourceApplyAdamh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�A@�AH�AXb>gradient_tape/ears_model/conv2d_227/Conv2D/Conv2DBackpropInputh
�
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28�A@�AH�AXbears_model/conv2d_231/Conv2Dh
�
�void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28�A@�AH�AXb>gradient_tape/ears_model/conv2d_231/Conv2D/Conv2DBackpropInputh
�
Ovoid scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*28�A@�AH�AXb?gradient_tape/ears_model/conv2d_235/Conv2D/Conv2DBackpropFilterh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�A@�AH�AXbears_model/conv2d_223/Conv2Dh
�
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28�@@�@H�@b7gradient_tape/ears_model/conv2d_230/BiasAdd/BiasAddGradh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�@@�@H�@b%Adam/Adam/update_31/ResourceApplyAdamh
�
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28�@@�@H�@b7gradient_tape/ears_model/conv2d_228/BiasAdd/BiasAddGradh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long)*28�@@�@H�@b:categorical_crossentropy/softmax_cross_entropy_with_logitsh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�@@�@H�@b"Adam/Adam/update/ResourceApplyAdamh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�@@�@H�@Xb>gradient_tape/ears_model/conv2d_230/Conv2D/Conv2DBackpropInputh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 9, 128, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�@@�@H�@bogradient_tape/ears_model/conv2d_236/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�@@�@H�@b$Adam/Adam/update_4/ResourceApplyAdamh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�@@�@H�@b$Adam/Adam/update_5/ResourceApplyAdamh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�@@�@H�@Xbears_model/conv2d_225/Conv2Dh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�@@�@H�@Xb?gradient_tape/ears_model/conv2d_230/Conv2D/Conv2DBackpropFilterh
�
�
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 2> const, Eigen::DSizes<int, 2> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 2> const, Eigen::DSizes<int, 2> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28�?@�?H�?b,gradient_tape/ears_model/concatenate_8/Sliceh
�
�void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28�?@�?H�?Xb>gradient_tape/ears_model/conv2d_229/Conv2D/Conv2DBackpropInputh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�?@�?H�?b%Adam/Adam/update_16/ResourceApplyAdamh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�?@�?H�?Xb>gradient_tape/ears_model/conv2d_223/Conv2D/Conv2DBackpropInputh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�?@�?H�?Xb?gradient_tape/ears_model/conv2d_225/Conv2D/Conv2DBackpropFilterh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�?@�?H�?bpgradient_tape/ears_model/conv2d_231/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�?@�?H�?Xbears_model/conv2d_230/Conv2Dh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�?@�?H�?b%Adam/Adam/update_12/ResourceApplyAdamh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�?@�?H�?Xbears_model/conv2d_228/Conv2Dh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�>@�>H�>Xb>gradient_tape/ears_model/conv2d_228/Conv2D/Conv2DBackpropInputh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�>@�>H�>bpgradient_tape/ears_model/conv2d_227/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�>@�>H�>Xb>gradient_tape/ears_model/conv2d_226/Conv2D/Conv2DBackpropInputh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�=@�=H�=b$Adam/Adam/update_9/ResourceApplyAdamh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�=@�=H�=Xb?gradient_tape/ears_model/conv2d_223/Conv2D/Conv2DBackpropFilterh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�=@�=H�=Xb?gradient_tape/ears_model/conv2d_228/Conv2D/Conv2DBackpropFilterh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�=@�=H�=Xb?gradient_tape/ears_model/conv2d_232/Conv2D/Conv2DBackpropFilterh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�<@�<H�<bngradient_tape/ears_model/conv2d_224/Conv2D/Conv2DBackpropInput-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�;@�;H�;bpgradient_tape/ears_model/conv2d_229/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�;@�;H�;bogradient_tape/ears_model/conv2d_225/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�;@�;H�;b%Adam/Adam/update_25/ResourceApplyAdamh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�;@�;H�;Xbears_model/conv2d_226/Conv2Dh
�
�void tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�;@�;H�;bogradient_tape/ears_model/conv2d_233/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 9, 128, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�;@�;H�;bogradient_tape/ears_model/conv2d_237/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�:@�:H�:b%Adam/Adam/update_17/ResourceApplyAdamh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�:@�:H�:bOears_model/conv2d_230/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<long long, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<long long, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28�:@�:H�:bArgMaxh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�:@�:H�:bOears_model/conv2d_226/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�
�
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28�:@�:H�:bears_model/conv2d_233/BiasAddh
�
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28�:@�:H�:bears_model/conv2d_235/BiasAddh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�:@�:H�:b%Adam/Adam/update_33/ResourceApplyAdamh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�:@�:H�:b$Adam/Adam/update_3/ResourceApplyAdamh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�:@�:H�:bngradient_tape/ears_model/conv2d_226/Conv2D/Conv2DBackpropInput-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int)*28�9@�9H�9b5gradient_tape/ears_model/dense_13/BiasAdd/BiasAddGradh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�9@�9H�9b%Adam/Adam/update_11/ResourceApplyAdamh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�9@�9H�9b%Adam/Adam/update_19/ResourceApplyAdamh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�9@�9H�9b$Adam/Adam/update_1/ResourceApplyAdamh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�9@�9H�9b%Adam/Adam/update_15/ResourceApplyAdamh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�9@�9H�9b%Adam/Adam/update_29/ResourceApplyAdamh
�	
�	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28�9@�9H�9b:categorical_crossentropy/softmax_cross_entropy_with_logitsh
�
�void tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�8@�8H�8bOears_model/conv2d_232/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�8@�8H�8b%Adam/Adam/update_13/ResourceApplyAdamh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�7@�7H�7b$Adam/Adam/update_7/ResourceApplyAdamh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�7@�7H�7Xb?gradient_tape/ears_model/conv2d_227/Conv2D/Conv2DBackpropFilterh
�
�void tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�7@�7H�7bOears_model/conv2d_234/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 128, 9, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�7@�7H�7bpgradient_tape/ears_model/conv2d_237/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 128, 9, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�7@�7H�7bOears_model/conv2d_236/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28�7@�7H�7b%Adam/Adam/update_21/ResourceApplyAdamh
�
�void tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�7@�7H�7bpgradient_tape/ears_model/conv2d_235/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�7@�7H�7bpgradient_tape/ears_model/conv2d_233/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�7@�7H�7bogradient_tape/ears_model/conv2d_227/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�6@�6H�6bOears_model/conv2d_228/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�4@�4H�4bOears_model/conv2d_224/BiasAdd-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�3@�3H�3bogradient_tape/ears_model/conv2d_231/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�
�
�
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�3@�3H�3bogradient_tape/ears_model/conv2d_229/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�2@�2H�2Xb?gradient_tape/ears_model/conv2d_226/Conv2D/Conv2DBackpropFilterh
�
`compute_gemm_pointers(float2**, float2 const*, int, float2 const*, int, float2 const*, int, int)*28�1@�1H�1Xb?gradient_tape/ears_model/conv2d_223/Conv2D/Conv2DBackpropFilterh
�
�
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�1@�1H�1bogradient_tape/ears_model/conv2d_230/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28�0@�0H�0b
div_no_nanh
�
�
�
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)*28�0@�0H�0Xb>gradient_tape/ears_model/conv2d_225/Conv2D/Conv2DBackpropInputh
�
�void tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�0@�0H�0bogradient_tape/ears_model/conv2d_235/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�void tensorflow::functor::RowReduceKernel<cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long>, float*, cub::Sum>(cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long>, float*, int, int, cub::Sum, std::iterator_traits<cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long> >::value_type)*28�0@�0H�0bears_model/dense_13/Softmaxh
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�0@�0H�0bogradient_tape/ears_model/conv2d_228/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�
�
�
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, int)*28�0@�0H�0bLgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mulh
�
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)*28�0@�0H�0Xb>gradient_tape/ears_model/conv2d_223/Conv2D/Conv2DBackpropInputh
�
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)*28�0@�0H�0Xb>gradient_tape/ears_model/conv2d_227/Conv2D/Conv2DBackpropInputh
�
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28�0@�0H�0bears_model/conv2d_236/BiasAddh
�
�void tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�0@�0H�0bogradient_tape/ears_model/conv2d_232/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
�
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�/@�/H�/Xbears_model/conv2d_224/Conv2Dh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28�/@�/H�/bears_model/Casth
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<long long, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<long long, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28�/@�/H�/bArgMax_1h
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::MaxReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::MaxReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)*28�/@�/H�/b:categorical_crossentropy/softmax_cross_entropy_with_logitsh
�
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*28�/@�/H�/Xb?gradient_tape/ears_model/conv2d_224/Conv2D/Conv2DBackpropFilterh
�
Ovoid scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*28�/@�/H�/Xb?gradient_tape/ears_model/conv2d_237/Conv2D/Conv2DBackpropFilterh
�
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28�/@�/H�/bears_model/conv2d_224/BiasAddh
�
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28�/@�/H�/bears_model/conv2d_226/BiasAddh
�
�void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28�/@�/H�/b7gradient_tape/ears_model/conv2d_225/BiasAdd/BiasAddGradh
�
�void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28�/@�/H�/b7gradient_tape/ears_model/conv2d_231/BiasAdd/BiasAddGradh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�/@�/H�/Xb?gradient_tape/ears_model/conv2d_224/Conv2D/Conv2DBackpropFilterh
�
�void tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28�/@�/H�/bogradient_tape/ears_model/conv2d_234/Conv2D/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
�
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28�/@�/H�/bears_model/conv2d_230/BiasAddh
�
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28�/@�/H�/bears_model/conv2d_234/BiasAddh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�/@�/H�/Xb>gradient_tape/ears_model/conv2d_224/Conv2D/Conv2DBackpropInputh
�
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28�.@�.H�.bears_model/conv2d_228/BiasAddh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_pow_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_pow_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28�.@�.H�.b
Adam/Pow_1h
�
�
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28�.@�.H�.bCast_2h
�
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28�.@�.H�.bears_model/conv2d_232/BiasAddh
�
�
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28�-@�-H�-b,categorical_crossentropy/weighted_loss/valueh
�
�
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28�-@�-H�-Xbears_model/conv2d_235/Conv2Dh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_pow_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_pow_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28�-@�-H�-bAdam/Powh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28�,@�,H�,Xbears_model/conv2d_222/Conv2Dh
�
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28�,@�,H�,Xb>gradient_tape/ears_model/conv2d_234/Conv2D/Conv2DBackpropInputh
�
Ovoid scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*28�,@�,H�,Xb?gradient_tape/ears_model/conv2d_236/Conv2D/Conv2DBackpropFilterh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28�,@�,H�,bAssignAddVariableOp_3h
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28�,@�,H�,bAssignAddVariableOp_1h
�
�void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28�,@�,H�,b7gradient_tape/ears_model/conv2d_233/BiasAdd/BiasAddGradh
�
�void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28�,@�,H�,b7gradient_tape/ears_model/conv2d_235/BiasAdd/BiasAddGradh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28�,@�,H�,Xbears_model/conv2d_226/Conv2Dh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28�+@�+H�+bEgradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nanh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28�+@�+H�+Xbears_model/conv2d_227/Conv2Dh
�
�void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28�+@�+H�+b7gradient_tape/ears_model/conv2d_237/BiasAdd/BiasAddGradh
�
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28�+@�+H�+Xb>gradient_tape/ears_model/conv2d_224/Conv2D/Conv2DBackpropInputh
�
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28�+@�+H�+Xb>gradient_tape/ears_model/conv2d_236/Conv2D/Conv2DBackpropInputh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28�+@�+H�+b;gradient_tape/categorical_crossentropy/weighted_loss/Tile_1h
�
�
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28�+@�+H�+bdiv_no_nan_1h
�
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28�*@�*H�*Xb>gradient_tape/ears_model/conv2d_232/Conv2D/Conv2DBackpropInputh
�
Ocudnn::gemm::computeWgradSplitKOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28�*@�*H�*Xb?gradient_tape/ears_model/conv2d_236/Conv2D/Conv2DBackpropFilterh
�
�void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28�*@�*H�*b7gradient_tape/ears_model/conv2d_229/BiasAdd/BiasAddGradh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28�*@�*H�*Xbears_model/conv2d_234/Conv2Dh
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*28�)@�)H�)bears_model/dense_13/BiasAddh
�
`compute_gemm_pointers(float2**, float2 const*, int, float2 const*, int, float2 const*, int, int)*28�)@�)H�)Xb?gradient_tape/ears_model/conv2d_224/Conv2D/Conv2DBackpropFilterh
�
Kcudnn::gemm::computeWgradBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28�)@�)H�)Xb?gradient_tape/ears_model/conv2d_232/Conv2D/Conv2DBackpropFilterh
�
�
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long)*28�)@�)H�)b:categorical_crossentropy/softmax_cross_entropy_with_logitsh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28�)@�)H�)Xbears_model/conv2d_228/Conv2Dh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�)@�)H�)Xb?gradient_tape/ears_model/conv2d_222/Conv2D/Conv2DBackpropFilterh
�
Kcudnn::gemm::computeWgradBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28�(@�(H�(Xb?gradient_tape/ears_model/conv2d_234/Conv2D/Conv2DBackpropFilterh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28�(@�(H�(bAssignAddVariableOp_2h
�
�void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*28�(@�(H�(bSum_2h
�
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28�(@�(H�(Xb>gradient_tape/ears_model/conv2d_228/Conv2D/Conv2DBackpropInputh
�
Kcudnn::gemm::computeWgradBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28�(@�(H�(Xb?gradient_tape/ears_model/conv2d_236/Conv2D/Conv2DBackpropFilterh
�
Fcudnn::gemm::computeBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28�(@�(H�(Xb>gradient_tape/ears_model/conv2d_234/Conv2D/Conv2DBackpropInputh
�
�void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28�(@�(H�(Xbears_model/conv2d_222/Conv2Dh
�
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28�'@�'H�'Xb>gradient_tape/ears_model/conv2d_226/Conv2D/Conv2DBackpropInputh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28�&@�&H�&Xbears_model/conv2d_225/Conv2Dh
�
�void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Max>(float const*, float*, int, int, cub::Max, std::iterator_traits<float const*>::value_type)*28�&@�&H�&bears_model/dense_13/Softmaxh
�
�
�
Ocudnn::gemm::computeWgradSplitKOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28�%@�%H�%Xb?gradient_tape/ears_model/conv2d_232/Conv2D/Conv2DBackpropFilterh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_boolean_and_op, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_boolean_and_op, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28�%@�%H�%b
LogicalAndh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28�%@�%H�%bCasth
�
�void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28�%@�%H�%b7gradient_tape/ears_model/conv2d_227/BiasAdd/BiasAddGradh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28�$@�$H�$bAssignAddVariableOph
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long, long long>, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long, long long>, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28�$@�$H�$bAdam/addh
�
�void tensorflow::(anonymous namespace)::GenerateNormalizedProb<float, float, 4>(float const*, float const*, float const*, float*, int, int, bool)*28�$@�$H�$bears_model/dense_13/Softmaxh
�
�
�
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28�$@�$H�$Xb>gradient_tape/ears_model/conv2d_230/Conv2D/Conv2DBackpropInputh
�
Ocudnn::gemm::computeWgradSplitKOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28�$@�$H�$Xb?gradient_tape/ears_model/conv2d_237/Conv2D/Conv2DBackpropFilterh
�
Ocudnn::gemm::computeWgradSplitKOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28�#@�#H�#Xb?gradient_tape/ears_model/conv2d_234/Conv2D/Conv2DBackpropFilterh
�
Ovoid scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*28�#@�#H�#Xb?gradient_tape/ears_model/conv2d_232/Conv2D/Conv2DBackpropFilterh
�
Ovoid scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*28�#@�#H�#Xb?gradient_tape/ears_model/conv2d_234/Conv2D/Conv2DBackpropFilterh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28�#@�#H�#Xbears_model/conv2d_232/Conv2Dh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28�#@�#H�#bMulh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::equal_to<long long>, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::equal_to<long long>, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28�#@�#H�#bEqualh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28�#@�#H�#bCast_3h
�
Kcudnn::gemm::computeWgradBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28�"@�"H�"Xb?gradient_tape/ears_model/conv2d_233/Conv2D/Conv2DBackpropFilterh
�
Kcudnn::gemm::computeWgradBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28�"@�"H�"Xb?gradient_tape/ears_model/conv2d_235/Conv2D/Conv2DBackpropFilterh
�
Fcudnn::gemm::computeBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28�"@�"H�"Xb>gradient_tape/ears_model/conv2d_230/Conv2D/Conv2DBackpropInputh
�
Ocudnn::gemm::computeWgradSplitKOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28�"@�"H�"Xb?gradient_tape/ears_model/conv2d_233/Conv2D/Conv2DBackpropFilterh
�
Fcudnn::gemm::computeBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28�!@�!H�!Xb>gradient_tape/ears_model/conv2d_224/Conv2D/Conv2DBackpropInputh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28�!@�!H�!Xbears_model/conv2d_224/Conv2Dh
�
Fcudnn::gemm::computeBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28� @� H� Xb>gradient_tape/ears_model/conv2d_236/Conv2D/Conv2DBackpropInputh
�
Kcudnn::gemm::computeWgradBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28� @� H� Xb?gradient_tape/ears_model/conv2d_237/Conv2D/Conv2DBackpropFilterh
�
Ocudnn::gemm::computeWgradSplitKOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28� @� H� Xb?gradient_tape/ears_model/conv2d_235/Conv2D/Conv2DBackpropFilterh
{
Dcudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)*28� @� H� Xbears_model/conv2d_233/Conv2Dh
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28� @� H� b8categorical_crossentropy/weighted_loss/num_elements/Casth
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28� @� H� bAdam/Cast_1h
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28� @� H� bcategorical_crossentropy/Casth
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long const, long long const>, Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long const, long long const>, Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28� @� H� bAdam/Adam/AssignAddVariableOph
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long const, long long const>, Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long const, long long const>, Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28� @� H� bAssignAddVariableOp_4h
�
�void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*28� @� H� b*categorical_crossentropy/weighted_loss/Sumh
�
Fcudnn::gemm::computeBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28�@�H�Xb>gradient_tape/ears_model/conv2d_226/Conv2D/Conv2DBackpropInputh
�
Fcudnn::gemm::computeBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28�@�H�Xb>gradient_tape/ears_model/conv2d_228/Conv2D/Conv2DBackpropInputh
�
Fcudnn::gemm::computeBOffsetsKernel(cudnn::gemm::ComputeBOffsetsParams)*28�@�H�Xb>gradient_tape/ears_model/conv2d_232/Conv2D/Conv2DBackpropInputh
�
Ovoid scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*28�@�H�Xb>gradient_tape/ears_model/conv2d_237/Conv2D/Conv2DBackpropInputh