#pragma once
#include "Pathtracer/Pathtracer.h"

struct PerfTest {
private:
	static constexpr int BUFFER_SIZE = 32;

	static constexpr const char * output_file = "perf.txt";

	struct POV {
		Vector3    position;
		Quaternion rotation;

		float timings[BUFFER_SIZE];
	};

	bool enabled;

	int index_pov;
	int index_buffer;

	Pathtracer * pathtracer;

	Array<POV> * povs;

public:
	Array<POV> povs_sponza = {
		{ Vector3( 18.739738f,  10.332139f, -10.229103f), Quaternion(0.000000f,  0.801883f,  0.000000f,  0.597480f), { } },
		{ Vector3( 31.355043f,  31.696985f,  13.222142f), Quaternion(0.000000f,  0.387925f,  0.000000f, -0.921690f), { } },
		{ Vector3( 70.257584f,   8.347624f,  49.902672f), Quaternion(0.000000f, -0.576111f,  0.000000f, -0.817371f), { } },
		{ Vector3( 24.349691f,  51.417969f, -10.351927f), Quaternion(0.000000f, -0.985181f,  0.000000f,  0.171514f), { } },
		{ Vector3( 24.349691f,  51.417969f, -10.351927f), Quaternion(0.000000f, -0.245309f,  0.000000f, -0.969444f), { } },
		{ Vector3(-15.957721f,  62.806641f, -43.916168f), Quaternion(0.000000f, -0.803925f,  0.000000f,  0.594729f), { } },
		{ Vector3(-52.839905f,  38.513454f,  -8.991060f), Quaternion(0.202261f, -0.729369f, -0.606600f, -0.243197f), { } },
		{ Vector3(-92.179306f,  74.721153f,  12.197323f), Quaternion(0.009840f,  0.621556f,  0.007809f, -0.783262f), { } },
		{ Vector3(-129.707321f, 17.916590f,  43.054050f), Quaternion(0.011467f,  0.408287f,  0.005129f, -0.912762f), { } }
	};

	Array<POV> povs_san_miguel = {
		{ Vector3(24.800940f, 2.231690f, 7.698777f),  Quaternion(0.000000f, 0.276862f, 0.000000f, 0.960908f),    { } },
		{ Vector3(15.381029f, 2.231690f, 5.391366f),  Quaternion(0.000000f, 0.963890f, 0.000000f, 0.266294f),    { } },
		{ Vector3(-8.911288f, 2.231690f, 0.720734f),  Quaternion(0.000000f, 0.708531f, 0.000000f, -0.705675f),   { } },
		{ Vector3(5.776708f, 0.671570f, 1.609853f),   Quaternion(0.000000f, 0.046106f, 0.000000f, -0.998933f),   { } },
		{ Vector3(4.405293f, 7.238101f, 0.628109f),   Quaternion(0.177942f, 0.655648f, 0.163070f, -0.715445f),   { } },
		{ Vector3(12.886882f, 4.282956f, 2.777880f),  Quaternion(0.177942f, 0.655648f, 0.163070f, -0.715445f),   { } },
		{ Vector3(21.197109f, 1.080195f, -2.957915f), Quaternion(-0.010298f, -0.981503f, 0.182976f, -0.055241f), { } }
	};

	Array<POV> povs_bistro = {
		{ Vector3(-7.348903f, 2.480730f, 4.043096f),   Quaternion(0.000000f, -0.772662f, 0.000000f, 0.634818f), { } },
		{ Vector3(41.444153f, 3.789229f, 34.644260f),  Quaternion(0.000000f, 0.450685f, 0.000000f, 0.892683f),  { } },
		{ Vector3(5.012013f, 2.168808f, 4.757593f),    Quaternion(0.000000f, 0.607728f, 0.000000f, 0.794145f),  { } },
		{ Vector3(3.510249f, 2.168808f, -15.540760f),  Quaternion(0.000000f, 0.969852f, 0.000000f, 0.243695f),  { } },
		{ Vector3(5.321108f, 13.875035f, -23.227219f), Quaternion(0.393976f, 0.491117f, 0.264929f, -0.730340f), { } },
		{ Vector3(-14.827924f, 6.492402f, -6.873830f), Quaternion(0.134087f, 0.105233f, 0.014321f, -0.985261f), { } },
		{ Vector3(-7.894484f, 2.674741f, 0.916597f),   Quaternion(0.104225f, 0.628730f, 0.085566f, -0.765840f), { } },
	};

	void init(Pathtracer * pathtracer, bool enabled, const char * scene_name);

	void frame_begin();
	bool frame_end(float frame_time);
};
