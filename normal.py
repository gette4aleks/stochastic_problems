import pyopencl as cl
import numpy as np

context = cl.create_some_context()
queue = cl.CommandQueue(context)
mf = cl.mem_flags

num_samples = 100000
uniform_randoms = np.random.rand(num_samples).astype(np.float32)

program = cl.Program(context, """
__kernel void normal_distribution(__global const float *uniform_randoms, __global float *normals) {
    int gid = get_global_id(0);
    float u1 = uniform_randoms[gid];
    float u2 = uniform_randoms[gid + get_global_size(0)];

    float z0 = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI * u2);
    normals[gid] = z0;
}
""").build()

uniform_randoms_cl = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=uniform_randoms)
normals_cl = cl.Buffer(context, mf.WRITE_ONLY, uniform_randoms.nbytes)

program.normal_distribution(queue, (num_samples // 2,), None, uniform_randoms_cl, normals_cl)

normals = np.empty(num_samples // 2, dtype=np.float32)
cl.enqueue_copy(queue, normals, normals_cl)

"""import matplotlib.pyplot as plt

plt.hist(normals, bins=50, density=True)
plt.title('Гистограмма нормального распределения')
plt.xlabel('Значения')
plt.ylabel('Плотность вероятности')
plt.grid()
plt.show()
"""
