extern "C" __global__ void kernel_taa() {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= SCREEN_WIDTH || y >= SCREEN_HEIGHT) return;

	int pixel_index = x + y * SCREEN_WIDTH;

	float4 colour = taa_frame_curr[pixel_index];

	float u = (float(x) + 0.5f) / float(SCREEN_WIDTH);
	float v = (float(y) + 0.5f) / float(SCREEN_HEIGHT);

	float2 screen_position_prev = tex2D(gbuffer_screen_position_prev, u, v);

	// Convert from [-1, 1] to [0, 1]
	float u_prev = 0.5f + 0.5f * screen_position_prev.x;
	float v_prev = 0.5f + 0.5f * screen_position_prev.y;

	float s_prev = u_prev * float(SCREEN_WIDTH)  - 0.5f;
	float t_prev = v_prev * float(SCREEN_HEIGHT) - 0.5f;

	int x1 = int(s_prev - 2.0f);
	int y1 = int(t_prev - 2.0f);

	float  sum_weight = 0.0f;
	float4 sum        = make_float4(0.0f);

	for (int j = y1; j < y1 + 4; j++) {
		if (j < 0 || j >= SCREEN_HEIGHT) continue;

		for (int i = x1; i < x1 + 4; i++) {
			if (i < 0 || i >= SCREEN_WIDTH) continue;

			float weight = 
				mitchell_netravali(float(i) - s_prev) * 
				mitchell_netravali(float(j) - t_prev);

			sum_weight += weight;
			sum        += weight * taa_frame_prev[i + j * SCREEN_WIDTH];
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
				float3 f = rgb_to_ycocg(make_float3(taa_frame_curr[pixel_index - SCREEN_WIDTH - 1]));

				colour_avg += f;
				colour_var += f * f;
			}

			float3 f = rgb_to_ycocg(make_float3(taa_frame_curr[pixel_index - 1]));

			colour_avg += f;
			colour_var += f * f;

			if (y < SCREEN_HEIGHT - 1) {
				float3 f = rgb_to_ycocg(make_float3(taa_frame_curr[pixel_index + SCREEN_WIDTH - 1]));

				colour_avg += f;
				colour_var += f * f;
			}
		}
		
		if (y >= 1) {
			float3 f = rgb_to_ycocg(make_float3(taa_frame_curr[pixel_index - SCREEN_WIDTH]));

			colour_avg += f;
			colour_var += f * f;
		}

		if (y < SCREEN_HEIGHT - 1) {
			float3 f = rgb_to_ycocg(make_float3(taa_frame_curr[pixel_index + SCREEN_WIDTH]));

			colour_avg += f;
			colour_var += f * f;
		}

		if (x < SCREEN_WIDTH - 1) {
			if (y >= 1) {
				float3 f = rgb_to_ycocg(make_float3(taa_frame_curr[pixel_index + 1 - SCREEN_WIDTH]));

				colour_avg += f;
				colour_var += f * f;
			}

			float3 f = rgb_to_ycocg(make_float3(taa_frame_curr[pixel_index + 1]));

			colour_avg += f;
			colour_var += f * f;

			if (y < SCREEN_HEIGHT - 1) {
				float3 f = rgb_to_ycocg(make_float3(taa_frame_curr[pixel_index + 1 + SCREEN_WIDTH]));

				colour_avg += f;
				colour_var += f * f;
			}
		}

		// Normalize the 9 taps
		colour_avg *= 1.0f / 9.0f;
		colour_var *= 1.0f / 9.0f;

		float3 sigma2 = colour_var - colour_avg * colour_avg;
		float3 sigma = make_float3(
			sqrt(max(0.0f, sigma2.x)),
			sqrt(max(0.0f, sigma2.y)),
			sqrt(max(0.0f, sigma2.z))
		);

		float3 colour_min = colour_avg - 1.25f * sigma;
		float3 colour_max = colour_avg + 1.25f * sigma;

		colour_prev = clamp(colour_prev, colour_min, colour_max);

		if (!isnan(colour_prev.x + colour_prev.y + colour_prev.z)) {
			const float alpha = 0.1f;
			float3 integrated = ycocg_to_rgb(alpha * colour_curr + (1.0f - alpha) * colour_prev);

			colour.x = integrated.x;
			colour.y = integrated.y;
			colour.z = integrated.z;
		}
	}

	surf2Dwrite(colour, accumulator, x * sizeof(float4), y);
}

extern "C" __global__ void kernel_taa_finalize() {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= SCREEN_WIDTH || y >= SCREEN_HEIGHT) return;

	int pixel_index = x + y * SCREEN_WIDTH;

	float4 colour;
	surf2Dread(&colour, accumulator, x * sizeof(float4), y);

	taa_frame_prev[pixel_index] = colour;

	// Inverse of gamma
	colour = colour * colour;

	// Inverse of Reinhard
	colour = colour / (make_float4(1.0f) - colour);

	surf2Dwrite(colour, accumulator, x * sizeof(float4), y);
}
