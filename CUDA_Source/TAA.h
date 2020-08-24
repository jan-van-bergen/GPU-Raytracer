
extern "C" __global__ void kernel_taa() {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= screen_width || y >= screen_height) return;

	int pixel_index = x + y * screen_pitch;

	float4 colour = taa_frame_curr[pixel_index];

	float u = (float(x) + 0.5f) / float(screen_width);
	float v = (float(y) + 0.5f) / float(screen_height);

	float2 screen_position_prev = gbuffer_screen_position_prev.get(u, v);

	// Convert from [-1, 1] to [0, 1]
	float u_prev = 0.5f + 0.5f * screen_position_prev.x;
	float v_prev = 0.5f + 0.5f * screen_position_prev.y;

	float s_prev = u_prev * float(screen_width);
	float t_prev = v_prev * float(screen_height);

	int x1 = int(s_prev + 0.5f) - 2;
	int y1 = int(t_prev + 0.5f) - 2;

	float  sum_weight = 0.0f;
	float4 sum        = make_float4(0.0f);

	for (int j = y1; j < y1 + 4; j++) {
		if (j < 0 || j >= screen_height) continue;

		for (int i = x1; i < x1 + 4; i++) {
			if (i < 0 || i >= screen_width) continue;

			float weight = 
				mitchell_netravali(float(i) + 0.5f - s_prev) * 
				mitchell_netravali(float(j) + 0.5f - t_prev);

			sum_weight += weight;
			sum        += weight * taa_frame_prev[i + j * screen_pitch];
		}
	}

	if (sum_weight > 0.0f) {
		sum /= sum_weight;

		float3 colour_curr = rgb_to_ycocg(make_float3(colour));
		float3 colour_prev = rgb_to_ycocg(make_float3(sum));

		float3 colour_avg = colour_curr;
		float3 colour_var = colour_curr * colour_curr;

		if (x >= 1) {
			if (y >= 1) {
				float3 f = rgb_to_ycocg(make_float3(taa_frame_curr[pixel_index - screen_pitch - 1]));

				colour_avg += f;
				colour_var += f * f;
			}

			float3 f = rgb_to_ycocg(make_float3(taa_frame_curr[pixel_index - 1]));

			colour_avg += f;
			colour_var += f * f;

			if (y < screen_height - 1) {
				float3 f = rgb_to_ycocg(make_float3(taa_frame_curr[pixel_index + screen_pitch - 1]));

				colour_avg += f;
				colour_var += f * f;
			}
		}
		
		if (y >= 1) {
			float3 f = rgb_to_ycocg(make_float3(taa_frame_curr[pixel_index - screen_pitch]));

			colour_avg += f;
			colour_var += f * f;
		}

		if (y < screen_height - 1) {
			float3 f = rgb_to_ycocg(make_float3(taa_frame_curr[pixel_index + screen_pitch]));

			colour_avg += f;
			colour_var += f * f;
		}

		if (x < screen_width - 1) {
			if (y >= 1) {
				float3 f = rgb_to_ycocg(make_float3(taa_frame_curr[pixel_index + 1 - screen_pitch]));

				colour_avg += f;
				colour_var += f * f;
			}

			float3 f = rgb_to_ycocg(make_float3(taa_frame_curr[pixel_index + 1]));

			colour_avg += f;
			colour_var += f * f;

			if (y < screen_height - 1) {
				float3 f = rgb_to_ycocg(make_float3(taa_frame_curr[pixel_index + 1 + screen_pitch]));

				colour_avg += f;
				colour_var += f * f;
			}
		}

		// Normalize the 9 taps
		colour_avg *= 1.0f / 9.0f;
		colour_var *= 1.0f / 9.0f;

		// Compute variance and standard deviation
		float3 sigma2 = colour_var - colour_avg * colour_avg;
		float3 sigma = make_float3(
			sqrt(max(0.0f, sigma2.x)),
			sqrt(max(0.0f, sigma2.y)),
			sqrt(max(0.0f, sigma2.z))
		);

		// Clamp based on average and standard deviation
		float3 colour_min = colour_avg - 1.25f * sigma;
		float3 colour_max = colour_avg + 1.25f * sigma;

		colour_prev = clamp(colour_prev, colour_min, colour_max);

		// Integrate temporally
		const float alpha = 0.1f;
		float3 integrated = ycocg_to_rgb(alpha * colour_curr + (1.0f - alpha) * colour_prev);

		colour.x = integrated.x;
		colour.y = integrated.y;
		colour.z = integrated.z;
	}

	accumulator.set(x, y, colour);
}

extern "C" __global__ void kernel_taa_finalize() {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= screen_width || y >= screen_height) return;

	int pixel_index = x + y * screen_pitch;

	float4 colour = accumulator.get(x, y);

	taa_frame_prev[pixel_index] = colour;

	// Inverse of gamma
	colour = colour * colour;

	// Inverse of "Pseudo" Reinhard
	colour = colour / (1.0f - luminance(colour.x, colour.y, colour.z));

	accumulator.set(x, y, colour);
}
