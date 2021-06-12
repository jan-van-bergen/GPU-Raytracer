#define epsilon 1e-8f // To avoid division by 0

struct Matrix4x4 {
	float4 row_0;
	float4 row_1;
	float4 row_2;
	float4 row_3;
};

__device__ inline void matrix4x4_transform(const Matrix4x4 & matrix, float4 & position) {
	position = make_float4( 
		matrix.row_0.x * position.x + matrix.row_0.y * position.y + matrix.row_0.z * position.z + matrix.row_0.w * position.w,
		matrix.row_1.x * position.x + matrix.row_1.y * position.y + matrix.row_1.z * position.z + matrix.row_1.w * position.w,
		matrix.row_2.x * position.x + matrix.row_2.y * position.y + matrix.row_2.z * position.z + matrix.row_2.w * position.w,
		matrix.row_3.x * position.x + matrix.row_3.y * position.y + matrix.row_3.z * position.z + matrix.row_3.w * position.w
	);
}

struct SVGFData {
	Matrix4x4 view_projection;
	Matrix4x4 view_projection_prev;
};

__device__ __constant__ SVGFData svgf_data;

__device__ inline SVGFData svgf_get_data() {
	SVGFData data;
	data.view_projection     .row_0 = __ldg(&svgf_data.view_projection     .row_0);
	data.view_projection     .row_1 = __ldg(&svgf_data.view_projection     .row_1);
	data.view_projection     .row_2 = __ldg(&svgf_data.view_projection     .row_2);
	data.view_projection     .row_3 = __ldg(&svgf_data.view_projection     .row_3);
	data.view_projection_prev.row_0 = __ldg(&svgf_data.view_projection_prev.row_0);
	data.view_projection_prev.row_1 = __ldg(&svgf_data.view_projection_prev.row_1);
	data.view_projection_prev.row_2 = __ldg(&svgf_data.view_projection_prev.row_2);
	data.view_projection_prev.row_3 = __ldg(&svgf_data.view_projection_prev.row_3);

	return data;
}

__device__ inline void svgf_set_gbuffers(int x, int y, const RayHit & hit, const float3 & hit_point, const float3 & hit_normal, const float3 & hit_point_prev) {
	float4 projected_hit_point      = make_float4(hit_point,      1.0f);
	float4 projected_hit_point_prev = make_float4(hit_point_prev, 1.0f);

	SVGFData svgf_data = svgf_get_data();

	matrix4x4_transform(svgf_data.view_projection,      projected_hit_point);
	matrix4x4_transform(svgf_data.view_projection_prev, projected_hit_point_prev);

	float depth      = projected_hit_point     .z;
	float depth_prev = projected_hit_point_prev.z;

	float2 normal_oct = oct_encode_normal(hit_normal);

	gbuffer_normal_and_depth       .set(x, y, make_float4(normal_oct.x, normal_oct.y, depth, depth_prev));
	gbuffer_mesh_id_and_triangle_id.set(x, y, make_int2(hit.mesh_id, hit.triangle_id));
	gbuffer_screen_position_prev   .set(x, y, make_float2(
		projected_hit_point_prev.x / projected_hit_point_prev.w,
		projected_hit_point_prev.y / projected_hit_point_prev.w
	));
}

__device__ inline bool is_tap_consistent(int x, int y, const float3 & normal, float depth) {
	if (x < 0 || x >= screen_width)  return false;
	if (y < 0 || y >= screen_height) return false;

	float4 prev_normal_and_depth = history_normal_and_depth[x + y * screen_pitch];
	float3 prev_normal = oct_decode_normal(make_float2(prev_normal_and_depth.x, prev_normal_and_depth.y));
	float  prev_depth  = prev_normal_and_depth.z;

	const float THRESHOLD_NORMAL = 0.95f;
	const float THRESHOLD_DEPTH  = 2.0f;

	bool consistent_normal = dot(normal, prev_normal)  > THRESHOLD_NORMAL;
	bool consistent_depth  = fabsf(depth - prev_depth) < THRESHOLD_DEPTH;

	return consistent_normal && consistent_depth;
}

__device__ inline float2 edge_stopping_weights(
	int delta_x, 
	int delta_y,
	const float2 & center_depth_gradient,
	float center_depth,
	float depth,
	const float3 & center_normal,
	const float3 & normal,
	float center_luminance_direct,
	float center_luminance_indirect,
	float luminance_direct,
	float luminance_indirect,
	float luminance_denom_direct,
	float luminance_denom_indirect
) {
	// ∇z(p)·(p−q) (Actually the negative of this but we take its absolute value)
	float d = 
		center_depth_gradient.x * float(delta_x) + 
		center_depth_gradient.y * float(delta_y); 

	float ln_w_z = fabsf(center_depth - depth) / (settings.sigma_z * fabsf(d) + epsilon);

	float w_n = powf(fmaxf(0.0f, dot(center_normal, normal)), settings.sigma_n);

	float w_l_direct   = w_n * expf(-fabsf(center_luminance_direct   - luminance_direct)   * luminance_denom_direct   - ln_w_z);
	float w_l_indirect = w_n * expf(-fabsf(center_luminance_indirect - luminance_indirect) * luminance_denom_indirect - ln_w_z);

	return make_float2(w_l_direct, w_l_indirect);
}

extern "C" __global__ void kernel_svgf_reproject(int sample_index) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= screen_width || y >= screen_height) return;

	int pixel_index = x + y * screen_pitch;

	float4 direct   = frame_buffer_direct  [pixel_index];
	float4 indirect = frame_buffer_indirect[pixel_index];

	// First two raw moments of luminance
	float4 moment;
	moment.x = luminance(direct.x,   direct.y,   direct.z);
	moment.y = luminance(indirect.x, indirect.y, indirect.z);
	moment.z = moment.x * moment.x;
	moment.w = moment.y * moment.y;

	float4 normal_and_depth     = gbuffer_normal_and_depth    .get(x, y);
	float2 screen_position_prev = gbuffer_screen_position_prev.get(x, y);

	float3 normal = oct_decode_normal(make_float2(normal_and_depth.x, normal_and_depth.y));
	float  depth      = normal_and_depth.z;
	float  depth_prev = normal_and_depth.w;

	// Check if this pixel belongs to the Skybox
	if (depth == 0.0f) {
		frame_buffer_direct  [pixel_index] = direct;
		frame_buffer_indirect[pixel_index] = indirect;

		return;
	}

	// Convert from [-1, 1] to [0, 1]
	float u_prev = 0.5f + 0.5f * screen_position_prev.x;
	float v_prev = 0.5f + 0.5f * screen_position_prev.y;

	float s_prev = u_prev * float(screen_width) ;
	float t_prev = v_prev * float(screen_height);
	
	int x_prev = int(s_prev - 0.5f);
	int y_prev = int(t_prev - 0.5f);

	// Calculate bilinear weights
	float fractional_s = s_prev - floor(s_prev);
	float fractional_t = t_prev - floor(t_prev);

	float one_minus_fractional_s = 1.0f - fractional_s;
	float one_minus_fractional_t = 1.0f - fractional_t;

	float w0 = one_minus_fractional_s * one_minus_fractional_t;
	float w1 =           fractional_s * one_minus_fractional_t;
	float w2 = one_minus_fractional_s *           fractional_t;
	float w3 = 1.0f - w0 - w1 - w2;

	float weights[4] = { w0, w1, w2, w3 };
	float consistent_weights_sum = 0.0f;

	// For each tap in a 2x2 bilinear filter, check if the re-projection is consistent
	// We sum the consistent bilinear weights for normalization purposes later on (weights should always add up to 1)
	for (int j = 0; j < 2; j++) {
		for (int i = 0; i < 2; i++) {
			int tap = i + j * 2;

			if (is_tap_consistent(x_prev + i, y_prev + j, normal, depth_prev)) {
				consistent_weights_sum += weights[tap];
			} else {
				weights[tap] = 0.0f;
			}
		}
	}

	float4 prev_direct   = make_float4(0.0f);
	float4 prev_indirect = make_float4(0.0f);
	float4 prev_moment   = make_float4(0.0f);

	// If we already found at least 1 consistent tap
	if (consistent_weights_sum > 0.0f) {
		// Add consistent taps using their bilinear weight
		for (int j = 0; j < 2; j++) {
			for (int i = 0; i < 2; i++) {
				int tap = i + j * 2;

				if (weights[tap] != 0.0f) {
					int tap_x = x_prev + i;
					int tap_y = y_prev + j;
					int tap_index = tap_x + tap_y * screen_pitch;

					float4 tap_direct   = history_direct  [tap_index];
					float4 tap_indirect = history_indirect[tap_index];
					float4 tap_moment   = history_moment  [tap_index];

					prev_direct   += weights[tap] * tap_direct;
					prev_indirect += weights[tap] * tap_indirect;
					prev_moment   += weights[tap] * tap_moment;
				}
			}
		}
	} else {
		// If we haven't yet found a consistent tap in a 2x2 region, try a 3x3 region
		for (int j = -1; j <= 1; j++) {
			for (int i = -1; i <= 1; i++) {
				int tap_x = x_prev + i;
				int tap_y = y_prev + j;

				if (is_tap_consistent(tap_x, tap_y, normal, depth_prev)) {
					int tap_index = tap_x + tap_y * screen_pitch;

					prev_direct   += history_direct  [tap_index];
					prev_indirect += history_indirect[tap_index];
					prev_moment   += history_moment  [tap_index];

					consistent_weights_sum += 1.0f;
				}
			}
		}
	}

	if (consistent_weights_sum > 0.0f) {
		// Normalize
		prev_direct   /= consistent_weights_sum;
		prev_indirect /= consistent_weights_sum;
		prev_moment   /= consistent_weights_sum;

		int history = ++history_length[pixel_index]; // Increase History Length by 1 step

		float inv_history = 1.0f / float(history);
		float alpha_colour = fmaxf(settings.alpha_colour, inv_history);
		float alpha_moment = fmaxf(settings.alpha_moment, inv_history);

		// Integrate using exponential moving average
		direct   = lerp(prev_direct,   direct,   alpha_colour);
		indirect = lerp(prev_indirect, indirect, alpha_colour);
		moment   = lerp(prev_moment,   moment,   alpha_moment);
		
		if (history >= 4 || !settings.enable_spatial_variance) {
			float variance_direct   = fmaxf(0.0f, moment.z - moment.x * moment.x);
			float variance_indirect = fmaxf(0.0f, moment.w - moment.y * moment.y);
			
			// Store the Variance in the alpha channels
			direct  .w = variance_direct;
			indirect.w = variance_indirect;
		}
	} else {
		history_length[pixel_index] = 0; // Reset History Length

		direct.w   = 1.0f;
		indirect.w = 1.0f;
	}

	frame_buffer_direct  [pixel_index] = direct;
	frame_buffer_indirect[pixel_index] = indirect;
	frame_buffer_moment  [pixel_index] = moment;
}

extern "C" __global__ void kernel_svgf_variance(
	float4 const * colour_direct_in,
	float4 const * colour_indirect_in,
	float4       * colour_direct_out,
	float4       * colour_indirect_out
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= screen_pitch || y >= screen_height) return;

	int pixel_index = x + y * screen_pitch;

	int history = history_length[pixel_index];

	if (history >= 4) {
		colour_direct_out  [pixel_index] = colour_direct_in  [pixel_index];
		colour_indirect_out[pixel_index] = colour_indirect_in[pixel_index];

		return;
	}

	float luminance_denom = 1.0f / settings.sigma_l;

	float4 center_colour_direct   = colour_direct_in  [pixel_index];
	float4 center_colour_indirect = colour_indirect_in[pixel_index];

	float center_luminance_direct   = luminance(center_colour_direct.x,   center_colour_direct.y,   center_colour_direct.z);
	float center_luminance_indirect = luminance(center_colour_indirect.x, center_colour_indirect.y, center_colour_indirect.z);

	float4 center_normal_and_depth = gbuffer_normal_and_depth.get(x, y);

	float3 center_normal = oct_decode_normal(make_float2(center_normal_and_depth.x, center_normal_and_depth.y));
	float  center_depth  = center_normal_and_depth.z;

	float2 center_depth_gradient = make_float2(
		gbuffer_normal_and_depth.get(x + 1, y    ).z - center_depth,
		gbuffer_normal_and_depth.get(x,     y + 1).z - center_depth
	);

	// Check if this pixel belongs to the Skybox
	if (center_depth == 0.0f) {
		colour_direct_out  [pixel_index] = center_colour_direct;
		colour_indirect_out[pixel_index] = center_colour_indirect;

		return;
	}

	float sum_weight_direct   = 1.0f;
	float sum_weight_indirect = 1.0f;

	float4 sum_colour_direct   = center_colour_direct;
	float4 sum_colour_indirect = center_colour_indirect;

	float4 sum_moment = make_float4(0.0f);

	const int radius = 3; // 7x7 filter

	for (int j = -radius; j <= radius; j++) {
		int tap_y = y + j;

		if (tap_y < 0 || tap_y >= screen_height) continue;

		for (int i = -radius; i <= radius; i++) {
			int tap_x = x + i;

			if (tap_x < 0 || tap_x >= screen_width) continue;

			if (i == 0 && j == 0) continue; // Center pixel is treated separately

			int tap_index = tap_x + tap_y * screen_pitch;

			float4 colour_direct   = colour_direct_in   [tap_index];
			float4 colour_indirect = colour_indirect_in [tap_index];
			float4 moment          = frame_buffer_moment[tap_index];

			float luminance_direct   = luminance(colour_direct.x,   colour_direct.y,   colour_direct.z);
			float luminance_indirect = luminance(colour_indirect.x, colour_indirect.y, colour_indirect.z);

			float4 normal_and_depth = gbuffer_normal_and_depth.get(tap_x, tap_y);

			float3 normal = oct_decode_normal(make_float2(normal_and_depth.x, normal_and_depth.y));
			float depth = normal_and_depth.z;

			float2 w = edge_stopping_weights(
				i, j,
				center_depth_gradient,
				center_depth, depth,
				center_normal, normal,
				center_luminance_direct, center_luminance_indirect,
				luminance_direct, luminance_indirect,
				luminance_denom, luminance_denom
			);

			float w_direct   = w.x;
			float w_indirect = w.y;

			sum_weight_direct   += w_direct;
			sum_weight_indirect += w_indirect;

			sum_colour_direct   += w_direct   * colour_direct;
			sum_colour_indirect += w_indirect * colour_indirect;

			sum_moment += moment * make_float4(w_direct, w_indirect, w_direct, w_indirect);
		}
	}

	sum_weight_direct   = fmaxf(sum_weight_direct,   1e-6f);
	sum_weight_indirect = fmaxf(sum_weight_indirect, 1e-6f);
	
	sum_colour_direct   /= sum_weight_direct;
	sum_colour_indirect /= sum_weight_indirect;
	
	sum_moment /= make_float4(sum_weight_direct, sum_weight_indirect, sum_weight_direct, sum_weight_indirect);

	float variance_direct   = fmaxf(0.0f, sum_moment.z - sum_moment.x * sum_moment.x);
	float variance_indirect = fmaxf(0.0f, sum_moment.w - sum_moment.y * sum_moment.y);

	// float inv_history  = 1.0f / float(history + 1);
	// variance_direct   *= 4.0f * inv_history;
	// variance_indirect *= 4.0f * inv_history;

	sum_colour_direct  .w = variance_direct;
	sum_colour_indirect.w = variance_indirect;
		
	// Store the Variance in the alpha channel
	colour_direct_out  [pixel_index] = sum_colour_direct;
	colour_indirect_out[pixel_index] = sum_colour_indirect;
}

// Determines which iterations' colour buffers are used as history colour buffers for the next frame
// Can be used to balance between temporal stability and bias from spatial filtering
const int feedback_iteration = 1;

extern "C" __global__ void kernel_svgf_atrous(
	float4 const * colour_direct_in,
	float4 const * colour_indirect_in,
	float4       * colour_direct_out,
	float4       * colour_indirect_out,
	int step_size
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= screen_width || y >= screen_height) return;

	int pixel_index = x + y * screen_pitch;

	float variance_blurred_direct   = 0.0f;
	float variance_blurred_indirect = 0.0f;

	const float kernel_gaussian[2][2] = {
		{ 1.0f / 4.0f, 1.0f / 8.0f  },
		{ 1.0f / 8.0f, 1.0f / 16.0f }
	};

	// Filter Variance using a 3x3 Gaussian Blur
	for (int j = -1; j <= 1; j++) {
		int tap_y = clamp(y + j, 0, screen_height - 1);
		
		for (int i = -1; i <= 1; i++) {
			int tap_x = clamp(x + i, 0, screen_width - 1);

			// Read the Variance of Direct/Indirect Illumination
			// The Variance is stored in the alpha channel (w coordinate)
			float variance_direct   = colour_direct_in  [tap_x + tap_y * screen_pitch].w;
			float variance_indirect = colour_indirect_in[tap_x + tap_y * screen_pitch].w;

			float kernel_weight = kernel_gaussian[abs(i)][abs(j)];

			variance_blurred_direct   += variance_direct   * kernel_weight;
			variance_blurred_indirect += variance_indirect * kernel_weight;
		}
	}

	// Precompute denominators that are loop invariant
	float luminance_denom_direct   = rsqrtf(settings.sigma_l * settings.sigma_l * fmaxf(0.0f, variance_blurred_direct)   + epsilon);
	float luminance_denom_indirect = rsqrtf(settings.sigma_l * settings.sigma_l * fmaxf(0.0f, variance_blurred_indirect) + epsilon);

	float4 center_colour_direct   = colour_direct_in  [pixel_index];
	float4 center_colour_indirect = colour_indirect_in[pixel_index];

	float center_luminance_direct   = luminance(center_colour_direct.x,   center_colour_direct.y,   center_colour_direct.z);
	float center_luminance_indirect = luminance(center_colour_indirect.x, center_colour_indirect.y, center_colour_indirect.z);

	float4 center_normal_and_depth = gbuffer_normal_and_depth.get(x, y);

	float3 center_normal = oct_decode_normal(make_float2(center_normal_and_depth.x, center_normal_and_depth.y));
	float  center_depth  = center_normal_and_depth.z;

	// Check if the pixel belongs to the Skybox
	if (center_depth == 0.0f) return;

	float2 center_depth_gradient = make_float2(
		gbuffer_normal_and_depth.get(x + 1, y    ).z - center_depth,
		gbuffer_normal_and_depth.get(x,     y + 1).z - center_depth
	);

	float  sum_weight_direct   = 1.0f;
	float  sum_weight_indirect = 1.0f;
	float4 sum_colour_direct   = center_colour_direct;
	float4 sum_colour_indirect = center_colour_indirect;

	// Use a 3x3 box filter, as recommended in the A-SVGF paper
	const int radius = 1;

	for (int j = -radius; j <= radius; j++) {
		int tap_y = y + j * step_size;

		if (tap_y < 0 || tap_y >= screen_height) continue;

		for (int i = -radius; i <= radius; i++) {
			int tap_x = x + i * step_size;
			
			if (tap_x < 0 || tap_x >= screen_width) continue;
			
			if (i == 0 && j == 0) continue; // Center pixel is treated separately

			float4 colour_direct   = colour_direct_in  [tap_x + tap_y * screen_pitch];
			float4 colour_indirect = colour_indirect_in[tap_x + tap_y * screen_pitch];

			float luminance_direct   = luminance(colour_direct.x,   colour_direct.y,   colour_direct.z);
			float luminance_indirect = luminance(colour_indirect.x, colour_indirect.y, colour_indirect.z);

			float4 normal_and_depth = gbuffer_normal_and_depth.get(tap_x, tap_y);

			float3 normal = oct_decode_normal(make_float2(normal_and_depth.x, normal_and_depth.y));
			float depth = normal_and_depth.z;

			float2 w = edge_stopping_weights(
				i * step_size, 
				j * step_size,
				center_depth_gradient,
				center_depth, depth,
				center_normal, normal,
				center_luminance_direct, center_luminance_indirect,
				luminance_direct,       luminance_indirect,
				luminance_denom_direct, luminance_denom_indirect
			);

			float weight_direct   = w.x;
			float weight_indirect = w.y;

			sum_weight_direct   += weight_direct;
			sum_weight_indirect += weight_indirect;

			// Filter Colour using the weights, filter Variance using the square of the weights
			sum_colour_direct   += make_float4(weight_direct,   weight_direct,   weight_direct,   weight_direct   * weight_direct)   * colour_direct;
			sum_colour_indirect += make_float4(weight_indirect, weight_indirect, weight_indirect, weight_indirect * weight_indirect) * colour_indirect;
		}
	}

	ASSERT(sum_weight_direct   > 10e-6f, "Divide by 0!");
	ASSERT(sum_weight_indirect > 10e-6f, "Divide by 0!");

	float inv_sum_weight_direct   = 1.0f / sum_weight_direct;
	float inv_sum_weight_indirect = 1.0f / sum_weight_indirect;

	// Normalize
	sum_colour_direct   *= inv_sum_weight_direct;
	sum_colour_indirect *= inv_sum_weight_indirect;

	// Alpha channel contains Variance, and needs to be divided by the square of the weights
	sum_colour_direct  .w *= inv_sum_weight_direct; 
	sum_colour_indirect.w *= inv_sum_weight_indirect;

	colour_direct_out  [pixel_index] = sum_colour_direct;
	colour_indirect_out[pixel_index] = sum_colour_indirect;

	if (step_size == (1 << feedback_iteration)) {
		history_direct  [pixel_index] = sum_colour_direct;
		history_indirect[pixel_index] = sum_colour_indirect;
	}
}

// Updating the Colour History buffer needs a separate kernel because
// multiple pixels may read from the same texel,
// thus we can only update it after all reads are done
extern "C" __global__ void kernel_svgf_finalize(
	const float4 * colour_direct,
	const float4 * colour_indirect
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= screen_width || y >= screen_height) return;

	int pixel_index = x + y * screen_pitch;

	float4 direct   = colour_direct  [pixel_index];
	float4 indirect = colour_indirect[pixel_index];

	float4 colour = direct + indirect;

	if (settings.modulate_albedo) {
		colour *= frame_buffer_albedo[pixel_index];
	}

	accumulator.set(x, y, colour);

	if (settings.enable_taa) {
		// "Pseudo" Reinhard (uses luma)
		colour = colour / (1.0f + luminance(colour.x, colour.y, colour.z));

		// Convert to gamma space
		colour.x = sqrtf(fmaxf(0.0f, colour.x));
		colour.y = sqrtf(fmaxf(0.0f, colour.y));
		colour.z = sqrtf(fmaxf(0.0f, colour.z));

		taa_frame_curr[pixel_index] = colour;
	}

	float4 moment = frame_buffer_moment[pixel_index];

	float4 normal_and_depth = gbuffer_normal_and_depth.get(x, y);

	if (settings.atrous_iterations <= feedback_iteration) {
		// Normally the à-trous filter copies the illumination history,
		// but in case the filter was skipped we need to do this here
		history_direct  [pixel_index] = direct;
		history_indirect[pixel_index] = indirect;
	}

	history_moment          [pixel_index] = moment;
	history_normal_and_depth[pixel_index] = normal_and_depth;

	// Clear frame buffers for next frame
	frame_buffer_albedo  [pixel_index] = make_float4(0.0f);
	frame_buffer_direct  [pixel_index] = make_float4(0.0f);
	frame_buffer_indirect[pixel_index] = make_float4(0.0f);

	gbuffer_normal_and_depth       .set(x, y, make_float4(0.0f));
	gbuffer_mesh_id_and_triangle_id.set(x, y, make_int2(0));
	if (!settings.enable_taa) {
		// TAA previous screen space positions as well, don't clear here if TAA is enabled
		gbuffer_screen_position_prev.set(x, y, make_float2(0.0f));
	}
}
