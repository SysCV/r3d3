#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

torch::Tensor depth_filter_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ix,
    torch::Tensor thresh);

torch::Tensor frame_distance_cuda(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics,
  torch::Tensor ii,
  torch::Tensor jj,
  const float beta);

std::vector<torch::Tensor> projmap_cuda(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics,
  torch::Tensor ii,
  torch::Tensor jj);

torch::Tensor iproj_cuda(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics);

std::vector<torch::Tensor> ba_cuda(
    torch::Tensor poses,
    torch::Tensor poses_rel,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor eta,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor cii,
    torch::Tensor cjj,
    const int t0,
    const int t1,
    const int iterations,
    const float lm,
    const float ep,
    const bool motion_only);

std::vector<torch::Tensor> schur_block_cuda(
  torch::Tensor E,
  torch::Tensor Q,
  torch::Tensor w,
  torch::Tensor ii,
  torch::Tensor jj,
  torch::Tensor kk,
  const int t0,
  const int t1);

std::vector<torch::Tensor> projective_transform_cuda(
  torch::Tensor poses,
  torch::Tensor poses_rel,
  torch::Tensor disps,
  torch::Tensor intrinsics,
  torch::Tensor targets,
  torch::Tensor weights,
  torch::Tensor ii,
  torch::Tensor jj,
  torch::Tensor cii,
  torch::Tensor cjj);

std::vector<torch::Tensor> corr_index_cuda_forward(
  torch::Tensor volume,
  torch::Tensor coords,
  int radius);

std::vector<torch::Tensor> corr_index_cuda_backward(
  torch::Tensor volume,
  torch::Tensor coords,
  torch::Tensor corr_grad,
  int radius);

std::vector<torch::Tensor> altcorr_cuda_forward(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  int radius);

std::vector<torch::Tensor> altcorr_cuda_backward(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  torch::Tensor corr_grad,
  int radius);


#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> ba(
    torch::Tensor poses,
    torch::Tensor poses_rel,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor eta,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor cii,
    torch::Tensor cjj,
    const int t0,
    const int t1,
    const int iterations,
    const float lm,
    const float ep,
    const bool motion_only) {

  CHECK_INPUT(targets);
  CHECK_INPUT(weights);
  CHECK_INPUT(poses);
  CHECK_INPUT(poses_rel);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);
  CHECK_INPUT(cii);
  CHECK_INPUT(cjj);

  return ba_cuda(poses, poses_rel, disps, intrinsics, targets, weights,
                 eta, ii, jj, cii, cjj, t0, t1, iterations, lm, ep, motion_only);

}

torch::Tensor frame_distance(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj,
    const float beta) {

  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);

  return frame_distance_cuda(poses, disps, intrinsics, ii, jj, beta);

}


std::vector<torch::Tensor> projmap(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj) {

  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);

  return projmap_cuda(poses, disps, intrinsics, ii, jj);

}


torch::Tensor iproj(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics) {
  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);

  return iproj_cuda(poses, disps, intrinsics);
}


// c++ python binding
std::vector<torch::Tensor> corr_index_forward(
    torch::Tensor volume,
    torch::Tensor coords,
    int radius) {
  CHECK_INPUT(volume);
  CHECK_INPUT(coords);

  return corr_index_cuda_forward(volume, coords, radius);
}

std::vector<torch::Tensor> corr_index_backward(
    torch::Tensor volume,
    torch::Tensor coords,
    torch::Tensor corr_grad,
    int radius) {
  CHECK_INPUT(volume);
  CHECK_INPUT(coords);
  CHECK_INPUT(corr_grad);

  auto volume_grad = corr_index_cuda_backward(volume, coords, corr_grad, radius);
  return {volume_grad};
}

std::vector<torch::Tensor> altcorr_forward(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    int radius) {
  CHECK_INPUT(fmap1);
  CHECK_INPUT(fmap2);
  CHECK_INPUT(coords);

  return altcorr_cuda_forward(fmap1, fmap2, coords, radius);
}

std::vector<torch::Tensor> altcorr_backward(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    torch::Tensor corr_grad,
    int radius) {
  CHECK_INPUT(fmap1);
  CHECK_INPUT(fmap2);
  CHECK_INPUT(coords);
  CHECK_INPUT(corr_grad);

  return altcorr_cuda_backward(fmap1, fmap2, coords, corr_grad, radius);
}


torch::Tensor depth_filter(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ix,
    torch::Tensor thresh) {

    CHECK_INPUT(poses);
    CHECK_INPUT(disps);
    CHECK_INPUT(intrinsics);
    CHECK_INPUT(ix);
    CHECK_INPUT(thresh);

    return depth_filter_cuda(poses, disps, intrinsics, ix, thresh);
}


/* DEBUGGING */

std::vector<torch::Tensor> schur_block_test(
  torch::Tensor E,
  torch::Tensor Q,
  torch::Tensor w,
  torch::Tensor ii,
  torch::Tensor jj,
  torch::Tensor kk,
  const int t0,
  const int t1) {

  printf("TEST A\n");

  CHECK_INPUT(E);
  CHECK_INPUT(Q);
  CHECK_INPUT(w);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);
  CHECK_INPUT(kk);

  printf("TEST B\n");

  return schur_block_cuda(E, Q, w, ii, jj, kk, t0, t1);
}


std::vector<torch::Tensor> projective_transform_test(
  torch::Tensor poses,
  torch::Tensor poses_rel,
  torch::Tensor disps,
  torch::Tensor intrinsics,
  torch::Tensor targets,
  torch::Tensor weights,
  torch::Tensor ii,
  torch::Tensor jj,
  torch::Tensor cii,
  torch::Tensor cjj) {

  printf("TEST A\n");

  CHECK_INPUT(poses);
  CHECK_INPUT(poses_rel);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(targets);
  CHECK_INPUT(weights);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);
  CHECK_INPUT(cii);
  CHECK_INPUT(cjj);

  printf("TEST B\n");

  return projective_transform_cuda(poses, poses_rel, disps, intrinsics, targets, weights, ii, jj, cii, cjj);
}
/*************/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // bundle adjustment kernels
  m.def("ba", &ba, "bundle adjustment");
  m.def("schur_block_test", &schur_block_test, "schur_block test");
  m.def("projective_transform_test", &projective_transform_test, "projective_transform test");
  m.def("frame_distance", &frame_distance, "frame_distance");
  m.def("projmap", &projmap, "projmap");
  m.def("depth_filter", &depth_filter, "depth_filter");
  m.def("iproj", &iproj, "back projection");

  // correlation volume kernels
  m.def("altcorr_forward", &altcorr_forward, "ALTCORR forward");
  m.def("altcorr_backward", &altcorr_backward, "ALTCORR backward");
  m.def("corr_index_forward", &corr_index_forward, "INDEX forward");
  m.def("corr_index_backward", &corr_index_backward, "INDEX backward");
}