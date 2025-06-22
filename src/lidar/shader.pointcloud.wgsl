@group(0) @binding(0)
var<storage, read_write> v_indices: array<vec4<f32>>;

@group(0) @binding(1)
var acc_struct: acceleration_structure;

struct LidarBeam {
  direction: vec3<f32>,
  lidar_id: u32
};

@group(0) @binding(2)
var<storage, read> lidar_beam: array<LidarBeam>;

@group(0) @binding(3)
var<uniform> lidar_position: mat4x4f;

struct WorkGroupParameters {
  width: u32,
  height: u32,
  depth: u32,
  num_lidar_beams: u32,
};

@group(0) @binding(4)
var<uniform> work_group_params: WorkGroupParameters;

fn global_id_to_index(global_id: vec3<u32>) -> u32 {
    return global_id.x + global_id.y * work_group_params.width + global_id.z * work_group_params.width * work_group_params.height;
}

@compute @workgroup_size(1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id_to_index(global_id);
    if (index >= work_group_params.num_lidar_beams) {
        return; // Out of bounds
    }
    let m_origin = vec3f(lidar_position[0][3], 
            lidar_position[1][3] , 
            lidar_position[2][3]);

    let matrix = mat3x3f(lidar_position[0][0], 
                        lidar_position[0][1], 
                        lidar_position[0][2], 
                        lidar_position[1][0], 
                        lidar_position[1][1], 
                        lidar_position[1][2], 
                        lidar_position[2][0], 
                        lidar_position[2][1], 
                        lidar_position[2][2]);
    let direction = lidar_beam[index].direction * matrix;
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0x0u, 0xFFu, 0.1, 50.0, m_origin, direction));
    rayQueryProceed(&rq);

    let intersection = rayQueryGetCommittedIntersection(&rq);
    if (intersection.kind != RAY_QUERY_INTERSECTION_NONE) {
      v_indices[index] = vec4f(intersection.t * lidar_beam[index].direction.x,
                                      intersection.t * lidar_beam[index].direction.y,
                                      intersection.t * lidar_beam[index].direction.z,
                                      intersection.t); // TODO: Can replace with any thing
                                      // For instance, brightness, semantic class, etc.

    }
    else {
      v_indices[index] = vec4f(0.0, 0.0, 0.0, 0.0); // No intersection
    }
}
