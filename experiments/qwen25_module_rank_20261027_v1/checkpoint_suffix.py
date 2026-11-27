"""Memory implementation only: checkpoint native decoder-block suffixes.

The caller installs editable_grid.DifferentiableInstall separately.  This helper
never alters weights, requires_grad flags, model training mode, attention code,
position/mask arguments, or CUDA peak statistics.  Native block bound forwards
are retained, including all of their original keyword arguments and outputs.
"""
from __future__ import annotations

import inspect
import types

import torch
from torch.utils.checkpoint import checkpoint


def _grad_flags(model):
    return dict(parameters={name: bool(value.requires_grad) for name, value in model.named_parameters()},
                buffers={name: bool(value.requires_grad) for name, value in model.named_buffers()})


def _training_flags(model):
    return {name: bool(module.training) for name, module in model.named_modules()}


class SuffixCheckpoint:
    """Wrap target and later blocks with fixed non-reentrant checkpointing.

    The model must expose ``model.layers`` and already be entirely in eval mode.
    At the first target, hidden inputs need not require gradients: non-reentrant
    checkpointing still records dependency on a trainable leaf buffer captured
    by the native block.  ``torch.autograd.grad`` remains supported.
    """
    def __init__(self, model, first_layer):
        assert isinstance(first_layer, int) and not isinstance(first_layer, bool)
        self.model = model
        self.decoder = model.get_submodule('model')
        self.layers = model.get_submodule('model.layers')
        assert 0 <= first_layer < len(self.layers), 'Suffix start is outside decoder layers'
        self.first_layer = first_layer
        self.originals = {}
        self.checkpoint_forward_calls = {}
        self.no_grad_direct_calls = {}
        self.installed = False
        self.install_count = 0
        self.cleanup_error = None

    def _check_eval(self, block):
        assert not self.model.training and not self.decoder.training and not block.training, 'Checkpoint helper requires eval mode'

    @staticmethod
    def _check_no_cache(signature, args, kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        if 'use_cache' in bound.arguments:
            assert bound.arguments['use_cache'] in (None, False), 'KV cache is forbidden during fixed-prefix gradients'
        for name in ('past_key_value', 'past_key_values'):
            assert bound.arguments.get(name) is None, 'Mutable KV cache is forbidden'

    def _make_forward(self, block, native, signature, index):
        owner = self

        def suffix_forward(actual_block, *args, **kwargs):
            assert owner.installed and actual_block is block
            owner._check_eval(block)
            owner._check_no_cache(signature, args, kwargs)
            assert not any(name in kwargs for name in ('use_reentrant', 'preserve_rng_state', 'determinism_check', 'context_fn', 'debug')), 'Checkpoint control argument leaked into decoder kwargs'
            if not torch.is_grad_enabled():
                owner.no_grad_direct_calls[index] += 1
                return native(*args, **kwargs)
            owner.checkpoint_forward_calls[index] += 1
            return checkpoint(native, *args, use_reentrant=False,
                              preserve_rng_state=True, determinism_check='default', **kwargs)

        return types.MethodType(suffix_forward, block)

    def install(self):
        assert not self.installed, 'Suffix checkpoint already installed'
        assert not any(_training_flags(self.model).values()), 'All model modules must remain eval'
        for node in (self.model, self.decoder):
            config = getattr(node, 'config', None)
            if config is not None:
                assert getattr(config, 'use_cache', False) is False, 'Disable config.use_cache before installing'
        self.initial_state_keys = tuple(self.model.state_dict())
        self.initial_grad_flags = _grad_flags(self.model)
        self.initial_training_flags = _training_flags(self.model)
        self.originals = {}
        self.checkpoint_forward_calls = {}
        self.no_grad_direct_calls = {}
        self.installed = True
        self.install_count += 1
        try:
            for index in range(self.first_layer, len(self.layers)):
                block = self.layers[index]
                native = block.forward
                signature = inspect.signature(native)
                had_instance_forward = 'forward' in block.__dict__
                raw_instance_forward = block.__dict__.get('forward')
                wrapped = self._make_forward(block, native, signature, index)
                self.originals[index] = (block, native, wrapped, had_instance_forward, raw_instance_forward)
                self.checkpoint_forward_calls[index] = 0
                self.no_grad_direct_calls[index] = 0
                block.forward = wrapped
            assert tuple(self.model.state_dict()) == self.initial_state_keys
            assert _grad_flags(self.model) == self.initial_grad_flags
            assert _training_flags(self.model) == self.initial_training_flags
        except BaseException:
            self.restore()
            raise
        return self

    def restore(self):
        if not self.installed:
            return self
        changed = []
        for index, (block, native, wrapped, had_instance_forward, raw_instance_forward) in self.originals.items():
            if block.forward is not wrapped:
                changed.append(index)
            if had_instance_forward:
                block.forward = raw_instance_forward
            else:
                delattr(block, 'forward')
        self.installed = False
        if changed:
            self.cleanup_error = 'Decoder forwards changed externally: ' + str(changed)
            raise AssertionError(self.cleanup_error)
        return self

    def report(self):
        started = hasattr(self, 'initial_state_keys')
        return dict(status='NATIVE_SUFFIX_NONREENTRANT_CHECKPOINT_INSTALLED' if self.installed else 'NATIVE_SUFFIX_CHECKPOINT_RESTORED_OR_NOT_INSTALLED',
                    installed=self.installed,
                    first_layer=self.first_layer, total_decoder_layers=len(self.layers),
                    wrapped_indices=list(range(self.first_layer, len(self.layers))),
                    wrapped_block_count=len(self.layers)-self.first_layer,
                    use_reentrant=False, preserve_rng_state=True, determinism_check='default',
                    no_grad_calls_use_native_forward=True, native_kwargs_and_output_structure_preserved=True,
                    builtin_training_trigger_used=False, changes_requires_grad=False,
                    state_dict_keys_unchanged=not started or tuple(self.model.state_dict()) == self.initial_state_keys,
                    requires_grad_flags_unchanged=not started or _grad_flags(self.model) == self.initial_grad_flags,
                    training_flags_unchanged=not started or _training_flags(self.model) == self.initial_training_flags,
                    checkpoint_forward_calls=dict(self.checkpoint_forward_calls),
                    no_grad_direct_calls=dict(self.no_grad_direct_calls),
                    counters_scope='Top-level block forward calls only; checkpoint may partially recompute a native block during backward',
                    cuda_peak_statistics_reset=False, install_count=self.install_count,
                    cleanup_error=self.cleanup_error,
                    scope='Memory execution strategy only. Actual real-model forward/gradient equivalence and runtime memory must be verified by the worker.')

    def __enter__(self):
        if not self.installed:
            self.install()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore()
        return False
