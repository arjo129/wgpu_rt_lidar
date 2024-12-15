@group(0) @binding(0)
var<storage, read_write> v_indices: array<f32>;

@group(0) @binding(1)
var acc_struct: acceleration_structure;

struct LidarBeam {
  direction: vec3<f32>,
  lidar_id: u32
};

@group(0) @binding(2)
var<storage, read> lidar_beam: array<LidarBeam>;

@group(0) @binding(3)
var<storage, read> lidar_position: array<mat4x3f>;

@compute @workgroup_size(256, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let origin = 
      vec3f(lidar_position[lidar_beam[global_id.x].lidar_id][3][0], 
            lidar_position[lidar_beam[global_id.x].lidar_id][3][1], 
            lidar_position[lidar_beam[global_id.x].lidar_id][3][2]);

    let matrix = mat3x3f(lidar_position[lidar_beam[global_id.x].lidar_id][0][0], 
                        lidar_position[lidar_beam[global_id.x].lidar_id][0][1], 
                        lidar_position[lidar_beam[global_id.x].lidar_id][0][2], 
                        lidar_position[lidar_beam[global_id.x].lidar_id][1][0], 
                        lidar_position[lidar_beam[global_id.x].lidar_id][1][1], 
                        lidar_position[lidar_beam[global_id.x].lidar_id][1][2], 
                        lidar_position[lidar_beam[global_id.x].lidar_id][2][0], 
                        lidar_position[lidar_beam[global_id.x].lidar_id][2][1], 
                        lidar_position[lidar_beam[global_id.x].lidar_id][2][2]);
    //let matrix = mat3x3f(1,0,0,0,1,0,0,0,1);                    
    let direction = lidar_beam[global_id.x].direction * matrix;

    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.1, 200.0, origin, direction));
    rayQueryProceed(&rq);

    let intersection = rayQueryGetCommittedIntersection(&rq);
    if (intersection.kind != RAY_QUERY_INTERSECTION_NONE) {
        v_indices[global_id.x] = intersection.t;
    }
}