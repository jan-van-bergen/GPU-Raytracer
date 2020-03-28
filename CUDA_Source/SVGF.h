// __device__ float edge_stopping_weights(
//     float2 center_depth_gradient,
//     float2 delta,
//     float center_depth,
//     float depth,
//     float & w_l_direct, 
//     float & w_l_indirect
// ) {
//     float d = 
//         center_depth_gradient.x * float(delta.x) + 
//         center_depth_gradient.y * float(delta.y); // ∇z(p)·(p−q)
//     //float w_z = (phi_depth == 0.0f) ? 0.0f : abs(center_depth - depth) / phi_depth; // exp(-abs(center_depth - depth) / (sigma_z * abs(d) + epsilon));
//     float w_z = exp(-abs(center_depth - depth) / (sigma_z * abs(d) + epsilon));

//     float w_n = powf(fmaxf(0.0f, dot(center_normal, normal)), sigma_n);

//     w_l_direct   = exp(-abs(center_luminance_direct   - luminance_direct)   * luminance_denom_direct);
//     w_l_indirect = exp(-abs(center_luminance_indirect - luminance_indirect) * luminance_denom_indirect);

//     return w_z * w_n;
// }