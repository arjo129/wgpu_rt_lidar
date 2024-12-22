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
var<uniform> lidar_position: mat4x4f;

@compute @workgroup_size(256, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let m_origin = vec3f(lidar_position[0][3], 
            lidar_position[1][3], 
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
    //let matrix = mat3x3f(1,0,0,0,1,0,0,0,1);                    
    let direction = lidar_beam[global_id.x].direction;// * matrix;
   
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0x04u, 0xFFu, 0.1, 50.0, m_origin, direction));
    rayQueryProceed(&rq);

    let intersection = rayQueryGetCommittedIntersection(&rq);
    if (intersection.kind != RAY_QUERY_INTERSECTION_NONE) {
      v_indices[global_id.x] = intersection.t;
    }

    if (global_id.x == 0) {
      v_indices[0] = m_origin.x;
    }
    if (global_id.x == 1) {
      v_indices[1] = m_origin.y;
    }
    if (global_id.x == 2) {
      v_indices[2] = m_origin.z;
    }
    
}