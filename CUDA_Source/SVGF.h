#define epsilon 1e-8f // To avoid division by 0

// Frame Buffers
__device__ float4 * frame_buffer_albedo;
__device__ float4 * frame_buffer_direct;
__device__ float4 * frame_buffer_indirect;

__device__ float4 * frame_buffer_moment;

surface<void, 2> accumulator; // Final Frame buffer to be displayed on Screen

// GBuffers
texture<float4, cudaTextureType2D> gbuffer_normal_and_depth;
texture<float2, cudaTextureType2D> gbuffer_uv;
texture<int,    cudaTextureType2D> gbuffer_triangle_id;
texture<float2, cudaTextureType2D> gbuffer_screen_position_prev;
texture<float2, cudaTextureType2D> gbuffer_depth_gradient;

// History Buffers (Temporally Integrated)
__device__ int    * history_length;
__device__ float4 * history_direct;
__device__ float4 * history_indirect;
__device__ float4 * history_moment;
__device__ float4 * history_normal_and_depth;
__device__ int    * history_triangle_id;

__device__ inline bool is_tap_consistent(int x, int y, const float3 & normal, float depth) {
	if (x < 0 || x >= SCREEN_WIDTH)  return false;
	if (y < 0 || y >= SCREEN_HEIGHT) return false;

	float4 prev_normal_and_depth = history_normal_and_depth[x + y * SCREEN_WIDTH];
	
	float3 prev_normal = make_float3(prev_normal_and_depth);
	float  prev_depth  = prev_normal_and_depth.w;

	const float threshold_normal = 0.95f;
	const float threshold_depth  = 0.025f * 250.0f; // @HARDCODED @ROBUSTNESS: make this depend on camera near/far

	bool consistent_normals = dot(normal, prev_normal) > threshold_normal;
	bool consistent_depth   = abs(depth - prev_depth)  < threshold_depth;

	return consistent_normals && consistent_depth;
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

	float w_z = exp(-abs(center_depth - depth) / (SIGMA_Z * abs(d) + epsilon));

	// float w_n = pow(max(0.0f, dot(center_normal, normal)), SIGMA_N);
	float w_n1 = max(0.0f, dot(center_normal, normal));
	float w_n2  = w_n1  * w_n1;
	float w_n4  = w_n2  * w_n2;
	float w_n8  = w_n4  * w_n4;
	float w_n16 = w_n8  * w_n8;
	float w_n32 = w_n16 * w_n16;
	float w_n64 = w_n32 * w_n32;
	float w_n   = w_n64 * w_n64;

	float w_l_direct   = exp(-abs(center_luminance_direct   - luminance_direct)   * luminance_denom_direct);
	float w_l_indirect = exp(-abs(center_luminance_indirect - luminance_indirect) * luminance_denom_indirect);

	return w_z * w_n * make_float2(
		w_l_direct, 
		w_l_indirect
	);
}

extern "C" __global__ void kernel_svgf_temporal() {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= SCREEN_WIDTH || y >= SCREEN_HEIGHT) return;

	int pixel_index = x + y * SCREEN_WIDTH;

	float4 direct   = frame_buffer_direct  [pixel_index];
	float4 indirect = frame_buffer_indirect[pixel_index];

	// First two raw moments of luminance
	float4 moment;
	moment.x = luminance(direct.x,   direct.y,   direct.z);
	moment.y = luminance(indirect.x, indirect.y, indirect.z);
	moment.z = moment.x * moment.x;
	moment.w = moment.y * moment.y;

	float u = (float(x) + 0.5f) / float(SCREEN_WIDTH);
	float v = (float(y) + 0.5f) / float(SCREEN_HEIGHT);

	float4 normal_and_depth     = tex2D(gbuffer_normal_and_depth,     u, v);
	float2 screen_position_prev = tex2D(gbuffer_screen_position_prev, u, v);

	float3 normal = make_float3(normal_and_depth);
	float  depth  = normal_and_depth.w;

	// Convert from [-1, 1] to [0, 1]
	float u_prev = 0.5f + 0.5f * screen_position_prev.x;
	float v_prev = 0.5f + 0.5f * screen_position_prev.y;

	float s_prev = u_prev * float(SCREEN_WIDTH)  - 0.5f;
	float t_prev = v_prev * float(SCREEN_HEIGHT) - 0.5f;
	
	int x_prev = int(s_prev);
	int y_prev = int(t_prev);

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

	float consistent_weights[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	float consistent_weights_sum = 0.0f;

	const int2 offsets[4] = {
		{ 0, 0 }, { 1, 0 },
		{ 0, 1 }, { 1, 1 }
	};

	// For each tap in a 2x2 bilinear filter, check if the reprojection is consistent
	// We sum the consistent bilinear weights for normalization purposes later on (weights should always add up to 1)
	for (int tap = 0; tap < 4; tap++) {
		int2 offset = offsets[tap];

		if (is_tap_consistent(x_prev + offset.x, y_prev + offset.y, normal, depth)) {
			float weight = weights[tap];

			consistent_weights[tap] = weight;
			consistent_weights_sum += weight;
		}
	}

	float4 prev_direct   = make_float4(0.0f);
	float4 prev_indirect = make_float4(0.0f);
	float4 prev_moment   = make_float4(0.0f);

	// If we already found at least 1 consistent tap
	if (consistent_weights_sum > 0.0f) {
		// Add consistent taps using their bilinear weight
		for (int tap = 0; tap < 4; tap++) {
			if (consistent_weights[tap] != 0.0f) {
				int2 offset = offsets[tap];
				
				int tap_x = x_prev + offset.x;
				int tap_y = y_prev + offset.y;

				int tap_index = tap_x + tap_y * SCREEN_WIDTH;

				float4 tap_direct   = history_direct  [tap_index];
				float4 tap_indirect = history_indirect[tap_index];
				float4 tap_moment   = history_moment  [tap_index];

				prev_direct   += consistent_weights[tap] * tap_direct;
				prev_indirect += consistent_weights[tap] * tap_indirect;
				prev_moment   += consistent_weights[tap] * tap_moment;
			}
		}

		// Divide by the sum of the consistent weights to renormalize the sum of the consistent weights to 1
		prev_direct   /= consistent_weights_sum;
		prev_indirect /= consistent_weights_sum;
		prev_moment   /= consistent_weights_sum;
	} else {
		// If we haven't yet found a consistent tap in a 2x2 region, try a 3x3 region
		for (int j = -1; j <= 1; j++) {
			for (int i = -1; i <= 1; i++) {
				int tap_x = x_prev + i;
				int tap_y = y_prev + j;

				if (is_tap_consistent(tap_x, tap_y, normal, depth)) {
					int tap_index = tap_x + tap_y * SCREEN_WIDTH;

					prev_direct   += history_direct  [tap_index];
					prev_indirect += history_indirect[tap_index];
					prev_moment   += history_moment  [tap_index];

					consistent_weights_sum += 1.0f;
				}
			}
		}

		if (consistent_weights_sum > 0.0f) {
			prev_direct   /= consistent_weights_sum;
			prev_indirect /= consistent_weights_sum;
			prev_moment   /= consistent_weights_sum;
		}
	}

	if (consistent_weights_sum > 0.0f) {
		int history = ++history_length[pixel_index]; // Increase History Length by 1 step

		float inv_history = 1.0f / float(history);
		float alpha_colour = max(ALPHA_COLOUR, inv_history);
		float alpha_moment = max(ALPHA_COLOUR, inv_history);

		// Integrate using exponential moving average
		direct   = alpha_colour * direct   + (1.0f - alpha_colour) * prev_direct;
		indirect = alpha_colour * indirect + (1.0f - alpha_colour) * prev_indirect;
		moment   = alpha_moment * moment   + (1.0f - alpha_moment) * prev_moment;
		
		if (history >= 4) {
			float variance_direct   = max(0.0f, moment.z - moment.x * moment.x);
			float variance_indirect = max(0.0f, moment.w - moment.y * moment.y);
			
			// Store the Variance in the alpha channel
			direct.w   = variance_direct;
			indirect.w = variance_indirect;
		}
	} else {
		history_length[pixel_index] = 0; // Reset History Length
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

	if (x >= SCREEN_WIDTH || y >= SCREEN_HEIGHT) return;

	int pixel_index = x + y * SCREEN_WIDTH;

	float u = (float(x) + 0.5f) / float(SCREEN_WIDTH);
	float v = (float(y) + 0.5f) / float(SCREEN_HEIGHT);

	int history = history_length[pixel_index];

	if (history >= 4) {
		// @SPEED
		colour_direct_out  [pixel_index] = colour_direct_in  [pixel_index];
		colour_indirect_out[pixel_index] = colour_indirect_in[pixel_index];

		return;
	}

	// @SPEED: some redundancies here
	float luminance_denom_direct   = 1.0f / (SIGMA_L + epsilon);
	float luminance_denom_indirect = 1.0f / (SIGMA_L + epsilon);

	float4 center_colour_direct   = colour_direct_in  [pixel_index];
	float4 center_colour_indirect = colour_indirect_in[pixel_index];

	float center_luminance_direct   = luminance(center_colour_direct.x,   center_colour_direct.y,   center_colour_direct.z);
	float center_luminance_indirect = luminance(center_colour_indirect.x, center_colour_indirect.y, center_colour_indirect.z);

	float4 center_normal_and_depth = tex2D(gbuffer_normal_and_depth, u, v);
	float2 center_depth_gradient   = tex2D(gbuffer_depth_gradient,   u, v);

	float3 center_normal = make_float3(center_normal_and_depth);
	float  center_depth  = center_normal_and_depth.w;

	float sum_weight_direct   = 1.0f;
	float sum_weight_indirect = 1.0f;

	float4 sum_colour_direct   = center_colour_direct;
	float4 sum_colour_indirect = center_colour_indirect;

	float4 sum_moment = make_float4(0.0f);

	const int radius = 3; // 7x7 filter
	
	for (int j = -radius; j <= radius; j++) {
		int tap_y = y + j;

		if (tap_y < 0 || tap_y >= SCREEN_HEIGHT) continue;

		for (int i = -radius; i <= radius; i++) {
			int tap_x = x + i;

			if (tap_x < 0 || tap_x >= SCREEN_WIDTH) continue;

			if (i == 0 && j == 0) continue; // Center pixel is treated separately

			int tap_index = tap_x + tap_y * SCREEN_WIDTH;

			float tap_u = (float(tap_x) + 0.5f) / float(SCREEN_WIDTH);
			float tap_v = (float(tap_y) + 0.5f) / float(SCREEN_HEIGHT);

			float4 colour_direct   = colour_direct_in   [tap_index];
			float4 colour_indirect = colour_indirect_in [tap_index];
			float4 moment          = frame_buffer_moment[tap_index];

			float luminance_direct   = luminance(colour_direct.x,   colour_direct.y,   colour_direct.z);
			float luminance_indirect = luminance(colour_indirect.x, colour_indirect.y, colour_indirect.z);

			float4 normal_and_depth = tex2D(gbuffer_normal_and_depth, tap_u, tap_v);

			float3 normal = make_float3(normal_and_depth);
			float  depth  = normal_and_depth.w;

			float2 w = edge_stopping_weights(
				i, j,
				center_depth_gradient,
				center_depth, depth,
				center_normal, normal,
				center_luminance_direct, center_luminance_indirect,
				luminance_direct, luminance_indirect,
				luminance_denom_direct, luminance_denom_indirect
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

	sum_weight_direct   = max(sum_weight_direct,   1e-6f);
	sum_weight_indirect = max(sum_weight_indirect, 1e-6f);
	
	sum_colour_direct   /= sum_weight_direct;
	sum_colour_indirect /= sum_weight_indirect;
	
	sum_moment /= make_float4(sum_weight_direct, sum_weight_indirect, sum_weight_direct, sum_weight_indirect);

	float variance_direct   = max(0.0f, sum_moment.z - sum_moment.x * sum_moment.x);
	float variance_indirect = max(0.0f, sum_moment.w - sum_moment.y * sum_moment.y);

	// float inv_history  = 1.0f / float(history);
	// variance_direct   *= 4.0f * inv_history;
	// variance_indirect *= 4.0f * inv_history;

	sum_colour_direct.w   = variance_direct;
	sum_colour_indirect.w = variance_indirect;
		
	// Store the Variance in the alpha channel
	colour_direct_out  [pixel_index] = sum_colour_direct;
	colour_indirect_out[pixel_index] = sum_colour_indirect;
}

extern "C" __global__ void kernel_svgf_atrous(
	float4 const * colour_direct_in,
	float4 const * colour_indirect_in,
	float4       * colour_direct_out,
	float4       * colour_indirect_out,
	int step_size
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= SCREEN_WIDTH || y >= SCREEN_HEIGHT) return;

	int pixel_index = x + y * SCREEN_WIDTH;

	float u = (float(x) + 0.5f) / float(SCREEN_WIDTH);
	float v = (float(y) + 0.5f) / float(SCREEN_HEIGHT);

	float variance_blurred_direct   = 0.0f;
	float variance_blurred_indirect = 0.0f;

	const float kernel_gaussian[2][2] = {
		{ 1.0f / 4.0f, 1.0f / 8.0f  },
		{ 1.0f / 8.0f, 1.0f / 16.0f }
	};

	// Filter Variance using a 3x3 Gaussian Blur
	for (int j = -1; j <= 1; j++) {
		int tap_y = clamp(y + j, 0, SCREEN_HEIGHT - 1);
		
		for (int i = -1; i <= 1; i++) {
			int tap_x = clamp(x + i, 0, SCREEN_WIDTH - 1);

			// Read the Variance of Direct/Indirect Illumination
			// The Variance is stored in the alpha channel (w coordinate)
			float variance_direct   = colour_direct_in  [tap_x + tap_y * SCREEN_WIDTH].w;
			float variance_indirect = colour_indirect_in[tap_x + tap_y * SCREEN_WIDTH].w;

			float kernel_weight = kernel_gaussian[abs(i)][abs(j)];

			variance_blurred_direct   += variance_direct   * kernel_weight;
			variance_blurred_indirect += variance_indirect * kernel_weight;
		}
	}

	// Precompute denominators that are loop invariant
	float luminance_denom_direct   = 1.0f / (SIGMA_L * sqrt(max(0.0f, variance_blurred_direct))   + epsilon);
	float luminance_denom_indirect = 1.0f / (SIGMA_L * sqrt(max(0.0f, variance_blurred_indirect)) + epsilon);

	float4 center_colour_direct   = colour_direct_in  [pixel_index];
	float4 center_colour_indirect = colour_indirect_in[pixel_index];

	float center_luminance_direct   = luminance(center_colour_direct.x,   center_colour_direct.y,   center_colour_direct.z);
	float center_luminance_indirect = luminance(center_colour_indirect.x, center_colour_indirect.y, center_colour_indirect.z);

	float4 center_normal_and_depth = tex2D(gbuffer_normal_and_depth, u, v);
	float2 center_depth_gradient   = tex2D(gbuffer_depth_gradient,   u, v);

	float3 center_normal = make_float3(center_normal_and_depth);
	float  center_depth  = center_normal_and_depth.w;

	// Weights from the SVGF reference implementation,
	// the SVGF paper uses different kernel weights
	const float kernel_atrous[3] = {
		1.0f, 
		2.0f / 3.0f, 
		1.0f / 6.0f 
	};

	float  sum_weight_direct   = 1.0f;
	float  sum_weight_indirect = 1.0f;
	float4 sum_colour_direct   = center_colour_direct;
	float4 sum_colour_indirect = center_colour_indirect;

	// 5x5 À-Trous Filter
	const int radius = 2;

	for (int j = -radius; j <= radius; j++) {
		int tap_y = y + j * step_size;

		if (tap_y < 0 || tap_y >= SCREEN_HEIGHT) continue;

		for (int i = -radius; i <= radius; i++) {
			int tap_x = x + i * step_size;
			
			if (tap_x < 0 || tap_x >= SCREEN_WIDTH) continue;
			
			if (i == 0 && j == 0) continue; // Center pixel is treated separately

			float tap_u = (float(tap_x) + 0.5f) / float(SCREEN_WIDTH);
			float tap_v = (float(tap_y) + 0.5f) / float(SCREEN_HEIGHT);

			float4 colour_direct   = colour_direct_in  [tap_x + tap_y * SCREEN_WIDTH];
			float4 colour_indirect = colour_indirect_in[tap_x + tap_y * SCREEN_WIDTH];

			float luminance_direct   = luminance(colour_direct.x,   colour_direct.y,   colour_direct.z);
			float luminance_indirect = luminance(colour_indirect.x, colour_indirect.y, colour_indirect.z);

			float4 normal_and_depth = tex2D(gbuffer_normal_and_depth, tap_u, tap_v);

			float3 normal = make_float3(normal_and_depth);
			float  depth  = normal_and_depth.w;
			
			float2 w = edge_stopping_weights(
				i * step_size, 
				j * step_size,
				center_depth_gradient,
				center_depth, depth,
				center_normal, normal,
				center_luminance_direct, center_luminance_indirect,
				luminance_direct, luminance_indirect,
				luminance_denom_direct, luminance_denom_indirect
			);

			float w_kernel = kernel_atrous[abs(i)] * kernel_atrous[abs(j)];

			float w_direct   = w_kernel * w.x;
			float w_indirect = w_kernel * w.y;

			sum_weight_direct   += w_direct;
			sum_weight_indirect += w_indirect;

			// Filter Colour using the weights
			// Filter Variance using the square of the weights
			sum_colour_direct   += make_float4(w_direct,   w_direct,   w_direct,   w_direct   * w_direct)   * colour_direct;
			sum_colour_indirect += make_float4(w_indirect, w_indirect, w_indirect, w_indirect * w_indirect) * colour_indirect;
		}
	}

	if (sum_weight_direct > 10e-6f) {
		sum_colour_direct.x /= sum_weight_direct;
		sum_colour_direct.y /= sum_weight_direct;
		sum_colour_direct.z /= sum_weight_direct;
		sum_colour_direct.w /= sum_weight_direct * sum_weight_direct; // Alpha channel contains Variance
	}
	
	if (sum_weight_indirect > 10e-6f) {
		sum_colour_indirect.x /= sum_weight_indirect;
		sum_colour_indirect.y /= sum_weight_indirect;
		sum_colour_indirect.z /= sum_weight_indirect;
		sum_colour_indirect.w /= sum_weight_indirect * sum_weight_indirect; // Alpha channel contains Variance
	}

	colour_direct_out  [pixel_index] = sum_colour_direct;
	colour_indirect_out[pixel_index] = sum_colour_indirect;
	
	// Determines which iterations' colour buffers are used as history colour buffers for the next frame
	// Can be used to balance between temporal stability and bias from spatial filtering
	const int feedback_iteration = 1;
	
	if (step_size == (1 << feedback_iteration)) {
		history_direct  [pixel_index] = sum_colour_direct;
		history_indirect[pixel_index] = sum_colour_indirect;
	}
}

// Updating the Colour History buffer needs a separate kernel because
// multiple pixels may read from the same texel,
// thus we can only update it after all reads are done
extern "C" __global__ void kernel_svgf_finalize(const float4 * colour_direct, const float4 * colour_indirect) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= SCREEN_WIDTH || y >= SCREEN_HEIGHT) return;

	int pixel_index = x + y * SCREEN_WIDTH;

	float4 albedo   = frame_buffer_albedo[pixel_index];
	float4 direct   = colour_direct      [pixel_index];
	float4 indirect = colour_indirect    [pixel_index];

	float4 colour = albedo * (direct + indirect);
	surf2Dwrite(colour, accumulator, x * sizeof(float4), y);

	float4 moment = frame_buffer_moment[pixel_index];

	float u = (float(x) + 0.5f) / float(SCREEN_WIDTH);
	float v = (float(y) + 0.5f) / float(SCREEN_HEIGHT);

	float4 normal_and_depth = tex2D(gbuffer_normal_and_depth, u, v);

#if ATROUS_ITERATIONS == 0
	history_direct  [pixel_index] = direct;
	history_indirect[pixel_index] = indirect;
#endif
	history_moment          [pixel_index] = moment;
	history_normal_and_depth[pixel_index] = normal_and_depth;

	// @SPEED
	// Clear frame buffers for next frame
	frame_buffer_albedo  [pixel_index] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
	frame_buffer_direct  [pixel_index] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
	frame_buffer_indirect[pixel_index] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
}
