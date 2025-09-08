@group(0) @binding(0)
var<storage, read_write> result: array<u32>;

@group(0) @binding(1)
var acc_struct: acceleration_structure;

@group(0) @binding(2)
var<storage, read> from_v: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read> to: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read> mapping: array<u32>;

@compute @workgroup_size(1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let to_check = vec3f(to[mapping[index]].x, to[mapping[index]].y, to[mapping[index]].z);
    let from_check = vec3f(from_v[index].xyz);

    let direction = to_check - from_check;
    let m_origin = from_check;

    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0x0u, 0xFFu, 0.1, 50.0, m_origin, direction));
    rayQueryProceed(&rq);
    let intersection = rayQueryGetCommittedIntersection(&rq);
    if (intersection.kind != RAY_QUERY_INTERSECTION_NONE &&
    intersection.t < distance(from_check, to_check)
    ) {
        result[index] = 0;
    }
    else 
    {
        result[index] = 1;
    }
}