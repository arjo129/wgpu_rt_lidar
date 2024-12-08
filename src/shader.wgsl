@group(0) @binding(0)
var<storage, read_write> v_indices: array<f32>;

@group(0) @binding(1)
var acc_struct: acceleration_structure;

struct LidarBeam {
  direction: vec3<f32>,
  lidar_id: u32
};

@group(0) @binding(2)
var<storage, read_write> lidar_beam: array<LidarBeam>;

@compute @workgroup_size(256, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let origin = vec3<f32>(0.0, 0.0, 0.0);
    let direction = lidar_beam[global_id.x].direction;

    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.1, 200.0, origin, direction));
    rayQueryProceed(&rq);

    let intersection = rayQueryGetCommittedIntersection(&rq);
    if (intersection.kind != RAY_QUERY_INTERSECTION_NONE) {
        v_indices[global_id.x] = intersection.t;
    }
}