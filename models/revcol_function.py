# --------------------------------------------------------
# Reversible Column Networks
# Copyright (c) 2022 Megvii Inc.
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Yuxuan Cai
# --------------------------------------------------------

from typing import Any, List, Tuple

import torch


def get_gpu_states(forward_gpu_devices) -> Tuple[List[int], List[torch.Tensor]]:
    gpu_states = []
    for device in forward_gpu_devices:
        with torch.cuda.device(device):
            gpu_states.append(torch.cuda.get_rng_state())
    return gpu_states


def get_gpu_device(*args):
    return list(set(arg.get_device() for arg in args if isinstance(arg, torch.Tensor) and arg.is_cuda))


def set_device_states(cpu_state, devices, states) -> None:
    torch.set_rng_state(cpu_state)
    for device, state in zip(devices, states):
        with torch.cuda.device(device):
            torch.cuda.set_rng_state(state)


def detach_and_grad(inputs: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
    if not isinstance(inputs, tuple):
        raise RuntimeError("Only tuple inputs are supported.")

    outputs = []
    for item in inputs:
        if not isinstance(item, torch.Tensor):
            outputs.append(item)
            continue
        detached = item.detach()
        detached.requires_grad = True
        outputs.append(detached)
    return tuple(outputs)


def get_cpu_and_gpu_states(gpu_devices):
    return torch.get_rng_state(), get_gpu_states(gpu_devices)


class ReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3 = run_functions
        alpha0, alpha1, alpha2, alpha3 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
        ctx.cpu_autocast_kwargs = {
            "enabled": torch.is_autocast_cpu_enabled(),
            "dtype": torch.get_autocast_cpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }

        assert len(args) == 5
        x, c0, c1, c2, c3 = args
        ctx.first_col = isinstance(c0, int)

        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices

            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1) + c0 * alpha0

            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2) + c1 * alpha1

            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3) + c2 * alpha2

            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, None) + c3 * alpha3

        ctx.save_for_backward(x, c0, c1, c2, c3)
        return x, c0, c1, c2, c3

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3 = ctx.saved_tensors
        l0, l1, l2, l3 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right = grad_outputs
        x, c0, c1, c2, c3 = detach_and_grad((x, c0, c1, c2, c3))

        with (
            torch.enable_grad(),
            torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state),
            torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs),
            torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs),
        ):
            g3_up = g3_right
            g3_left = g3_up * alpha3
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)
            output3 = l3(c2, None)
            torch.autograd.backward(output3, g3_up, retain_graph=True)
            with torch.no_grad():
                c3_left = (c3 - output3) / alpha3
            g2_up = g2_right + c2.grad
            g2_left = g2_up * alpha2

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)
            output2 = l2(c1, c3_left)
            torch.autograd.backward(output2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            torch.autograd.backward(c3_left * alpha3, g3_up)

            with torch.no_grad():
                c2_left = (c2 - output2) / alpha2
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right + c1.grad
            g1_left = g1_up * alpha1

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            output1 = l1(c0, c2_left)
            torch.autograd.backward(output1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            torch.autograd.backward(c2_left * alpha2, g2_up)

            with torch.no_grad():
                c1_left = (c1 - output1) / alpha1
            g0_up = g0_right + c0.grad
            g0_left = g0_up * alpha0
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            output0 = l0(x, c1_left)
            torch.autograd.backward(output0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            torch.autograd.backward(c1_left * alpha1, g1_up)

            with torch.no_grad():
                c0_left = (c0 - output0) / alpha0
            gx_up = x.grad
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left
            c0_left.requires_grad = False
            torch.autograd.backward(c0_left * alpha0, g0_up)

        if ctx.first_col:
            return None, None, gx_up, None, None, None, None
        return None, None, gx_up, g0_left, g1_left, g2_left, g3_left
