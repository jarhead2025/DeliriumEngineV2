// FILEPATH: kernels/surfel_march.cl

kernel void surfel_march(
    const read_only image2d_t depth_tex,
    write_only image2d_t out_tex,
    __global int* results,
    const float max_distance,
    const float step_size,
    const int num_steps
) {
    int2 id = (int2)(get_global_id(0), get_global_id(1));
    
    // Assuming depth values are in [0, 1]
    float depth_value = read_imagef(depth_tex, CLK_FILTER_NEAREST, id).x;
    
    if (depth_value > max_distance) {
        return;
    }
    
    vec3 start_position = vec3(id.x * step_size, id.y * step_size, depth_value);
    float current_depth = 0.0; 
    for (int i = 0; i < num_steps; i++) {
        // Perform ray-marching logic here
        current_depth += step_size;
        
        if(current_depth > max_distance) {
            break;
        }
    }
    
    results[id.x * get_global_size(1) + id.y] = (int)current_depth;
    write_imagef(out_tex, CLK_WRITE_ONLY, id, vec4(current_depth, 0.0, 0.0, 1.0));
}
