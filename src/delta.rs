//! One GPU kernel for the gated-delta scan, registered on the fusion stream
//! so the core and the next state stay on the device.

use burn::backend::wgpu::{Wgpu, WgpuRuntime};
use burn::tensor::{DType, Shape, Tensor, TensorMetadata, TensorPrimitive};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device;
use burn_cubecl::{CubeBackend, cubecl::prelude::*};
use burn_fusion::stream::{Operation, OperationStreams};
use burn_fusion::FusionRuntime;
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};

use crate::backbone::{LIN_DIM, V_HEADS};

type Inner = CubeBackend<WgpuRuntime, f32, i32, u32>;
type Runtime = FusionCubeRuntime<WgpuRuntime>;

pub fn scan(
    q: Tensor<Wgpu, 3>,
    k: Tensor<Wgpu, 3>,
    v: Tensor<Wgpu, 3>,
    g: Tensor<Wgpu, 2>,
    beta: Tensor<Wgpu, 2>,
    state: Tensor<Wgpu, 3>,
) -> (Tensor<Wgpu, 3>, Tensor<Wgpu, 3>) {
    let seq = q.dims()[0];
    let mut streams = OperationStreams::default();
    let (client, q_ir, hold_q) = input(&mut streams, q);
    let (_, k_ir, hold_k) = input(&mut streams, k);
    let (_, v_ir, hold_v) = input(&mut streams, v);
    let (_, g_ir, hold_g) = input(&mut streams, g);
    let (_, beta_ir, hold_beta) = input(&mut streams, beta);
    let (_, state_ir, hold_state) = input(&mut streams, state);
    let core_ir = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([seq, V_HEADS, LIN_DIM]),
        DType::F32,
    );
    let state_ir_out = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([V_HEADS, LIN_DIM, LIN_DIM]),
        DType::F32,
    );
    let desc = CustomOpIr::new(
        "gated-delta",
        &[q_ir, k_ir, v_ir, g_ir, beta_ir, state_ir],
        &[core_ir, state_ir_out],
    );
    let [core, next] = client
        .register(streams, OperationIr::Custom(desc.clone()), DeltaOp { desc })
        .outputs();
    drop((hold_q, hold_k, hold_v, hold_g, hold_beta, hold_state));
    (
        Tensor::from_primitive(TensorPrimitive::Float(core)),
        Tensor::from_primitive(TensorPrimitive::Float(next)),
    )
}

fn input<const D: usize>(
    streams: &mut OperationStreams,
    tensor: Tensor<Wgpu, D>,
) -> (burn_fusion::Client<Runtime>, TensorIr, burn_fusion::FusionTensor<Runtime>) {
    let fusion = tensor.into_primitive().tensor();
    streams.tensor(&fusion);
    let client = fusion.client.clone();
    let ir = fusion.clone().into_ir();
    (client, ir, fusion)
}

#[derive(Debug)]
struct DeltaOp {
    desc: CustomOpIr,
}

impl Operation<Runtime> for DeltaOp {
    fn execute(&self, handles: &mut HandleContainer<<Runtime as FusionRuntime>::FusionHandle>) {
        let (inputs, outputs) = self.desc.as_fixed::<6, 2>();
        let q = into_contiguous(handles.get_float_tensor::<Inner>(&inputs[0]));
        let k = into_contiguous(handles.get_float_tensor::<Inner>(&inputs[1]));
        let v = into_contiguous(handles.get_float_tensor::<Inner>(&inputs[2]));
        let g = into_contiguous(handles.get_float_tensor::<Inner>(&inputs[3]));
        let beta = into_contiguous(handles.get_float_tensor::<Inner>(&inputs[4]));
        let state = into_contiguous(handles.get_float_tensor::<Inner>(&inputs[5]));
        let seq = q.shape().dims::<3>()[0];
        let client = q.client.clone();
        let device = q.device.clone();
        let core = empty_device::<WgpuRuntime, f32>(
            client.clone(),
            device.clone(),
            Shape::new([seq, V_HEADS, LIN_DIM]),
        );
        let next = empty_device::<WgpuRuntime, f32>(
            client.clone(),
            device,
            Shape::new([V_HEADS, LIN_DIM, LIN_DIM]),
        );
        unsafe {
            gated_delta_scan::launch_unchecked(
                &client,
                CubeCount::Static(V_HEADS as u32, 1, 1),
                CubeDim::new_1d(LIN_DIM as u32),
                q.into_array_arg(),
                k.into_array_arg(),
                v.into_array_arg(),
                g.into_array_arg(),
                beta.into_array_arg(),
                state.into_array_arg(),
                next.clone().into_array_arg(),
                core.clone().into_array_arg(),
                seq as u32,
            );
        }
        handles.register_float_tensor::<Inner>(&outputs[0].id, core);
        handles.register_float_tensor::<Inner>(&outputs[1].id, next);
    }
}

#[cube(launch_unchecked)]
fn gated_delta_scan(
    q: &Array<f32>,
    k: &Array<f32>,
    v: &Array<f32>,
    g: &Array<f32>,
    beta: &Array<f32>,
    state_in: &Array<f32>,
    state_out: &mut Array<f32>,
    out: &mut Array<f32>,
    seq: u32,
) {
    let head = usize::cast_from(CUBE_POS_X);
    let j = usize::cast_from(UNIT_POS_X);
    let dim = comptime!(LIN_DIM);
    let heads = comptime!(V_HEADS);
    if head < heads && j < dim {
        let mut i = 0usize;
        while i < dim {
            let at = (head * dim + i) * dim + j;
            state_out[at] = state_in[at];
            i += 1usize;
        }
        let mut t = 0usize;
        let steps = usize::cast_from(seq);
        while t < steps {
            let token = t * heads + head;
            let decay = Exp::exp(g[token]);
            let scale = beta[token];
            let base = token * dim;
            let mut kv = 0.0f32;
            i = 0usize;
            while i < dim {
                let at = (head * dim + i) * dim + j;
                let value = state_out[at] * decay;
                state_out[at] = value;
                kv += value * k[base + i];
                i += 1usize;
            }
            let delta = (v[base + j] - kv) * scale;
            let mut acc = 0.0f32;
            i = 0usize;
            while i < dim {
                let at = (head * dim + i) * dim + j;
                let value = state_out[at] + k[base + i] * delta;
                state_out[at] = value;
                acc += value * q[base + i];
                i += 1usize;
            }
            out[base + j] = acc;
            t += 1usize;
        }
    }
}
