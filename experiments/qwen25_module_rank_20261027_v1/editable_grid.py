"""Auditable target-grid observation and differentiable ImpQuant A16 forwards.

The native GPTQ solver/quantize result is returned unchanged. Only a single
requested module's integer codes and original group parameters are retained.
No optimization, code selection, or modification of upstream source occurs.
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
from pathlib import Path
import types

import torch
from torch.nn import functional as F


def tensor_digest(value):
    x = value.detach().cpu().contiguous()
    h = hashlib.sha256(str((str(x.dtype), tuple(x.shape))).encode())
    h.update(x.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _dtype(name):
    mapping = {str(t): t for t in (torch.float32, torch.float16, torch.bfloat16)}
    if isinstance(name, torch.dtype):
        assert name in mapping.values()
        return name
    assert name in mapping, name
    return mapping[name]


def decode_grid(grid, codes=None, *, dtype=None, device=None):
    """Decode authoritative codes; never estimate/recompute scale or zero."""
    q = grid['codes'] if codes is None else codes
    assert tuple(q.shape) == tuple(grid['shape']) and q.dtype == torch.uint8
    group = int(grid['group_size'])
    rows, columns = q.shape
    assert columns % group == 0
    dev = q.device if device is None else torch.device(device)
    s = grid['scales'].to(device=dev)
    z = grid['zeros'].to(device=dev)
    assert s.dtype == z.dtype == torch.float32
    assert tuple(s.shape) == tuple(z.shape) == (rows, columns // group)
    values = s.unsqueeze(-1) * (q.to(device=dev, dtype=torch.float32).reshape(rows, -1, group) - z.unsqueeze(-1))
    return values.reshape(rows, columns).to(dtype=_dtype(grid['weight_dtype'] if dtype is None else dtype))


class GridCapture:
    """Observe the native sequential GPTQ calls in one worker process.

    ``module_name_map`` may be ``dict(model.named_modules())`` or a map from
    module objects/ids to qualified names. ``install`` returns self. ``finish``
    restores the native callables and saves ``target_grid.pt`` plus its receipt.
    Calls must be serial: a nested/concurrent solver is rejected.
    """
    def __init__(self, module_name_map, target_name, out_dir):
        self.names = {}
        for key, value in module_name_map.items():
            if isinstance(key, str):
                self.names[id(value)] = key
            else:
                self.names[key if isinstance(key, int) else id(key)] = value
        assert target_name in self.names.values(), target_name
        self.target_name = target_name
        self.out_dir = Path(out_dir)
        self.records = []
        self.active = None
        self.grid = None
        self.installed = False
        self.target_calls = 0

    def install(self, gptq_module):
        assert not self.installed
        self.native_module = gptq_module
        self.original_fast = gptq_module.GPTQ.fasterquant
        self.original_quantize = gptq_module.quantize
        self.source_sha256 = hashlib.sha256(Path(inspect.getsourcefile(self.original_fast)).read_bytes()).hexdigest()
        signature = inspect.signature(self.original_fast)
        owner = self

        @functools.wraps(self.original_fast)
        def observed_fast(solver, *args, **kwargs):
            assert owner.active is None, 'Nested/concurrent GPTQ observer use'
            bound = signature.bind(solver, *args, **kwargs)
            bound.apply_defaults()
            name = owner.names.get(id(solver.layer))
            assert name is not None, 'Solver module absent from frozen module-name map'
            assert not any(r['name'] == name for r in owner.records), 'Repeated GPTQ solver/module'
            group = int(bound.arguments['groupsize'])
            actorder = bool(bound.arguments['actorder'])
            maxq = int(solver.quantizer.maxq.item())
            assert group == 128 and not actorder and maxq == 7, 'Gate requires W3/group128/actorder=False'
            rows, columns = solver.layer.weight.shape
            record = dict(name=name, shape=[rows, columns], group_size=group,
                          block_size=int(bound.arguments['blocksize']), actorder=actorder,
                          max_code=maxq, min_code=0, actual_symmetric=bool(solver.quantizer.sym),
                          requested_damping=float(bound.arguments['percdamp']),
                          weight_dtype=str(solver.layer.weight.dtype), captured=name == owner.target_name)
            owner.active = dict(name=name, record=record, column=0)
            if record['captured']:
                assert owner.grid is None and columns % group == 0
                dev = solver.dev
                owner.active.update(codes=torch.empty((rows, columns), device=dev, dtype=torch.uint8),
                                    scales=torch.empty((rows, columns // group), device=dev, dtype=torch.float32),
                                    zeros=torch.empty((rows, columns // group), device=dev, dtype=torch.float32),
                                    invalid=torch.zeros((), device=dev, dtype=torch.bool))
            try:
                result = owner.original_fast(solver, *args, **kwargs)
                current = owner.active
                if record['captured']:
                    assert current['column'] == columns, (current['column'], columns)
                    assert not bool(current['invalid'].item()), 'Observer/native quantization or within-group parameters differ'
                    grid = dict(format='authoritative_native_gptq_target_grid_v1', module=name,
                                codes=current['codes'], scales=current['scales'], zeros=current['zeros'],
                                shape=[rows, columns], group_size=group, min_code=0, max_code=7,
                                weight_dtype=str(solver.layer.weight.dtype), actual_symmetric=record['actual_symmetric'],
                                actorder=False, native_scale_dtype='torch.float32', native_zero_dtype='torch.float32')
                    rebuilt = decode_grid(grid, device=solver.layer.weight.device)
                    actual = solver.layer.weight.detach().contiguous()
                    assert torch.equal(rebuilt.contiguous().view(torch.uint8), actual.view(torch.uint8)), 'Grid roundtrip not bit-exact'
                    record['weight_sha256'] = tensor_digest(actual)
                    record['grid_roundtrip_bits_equal'] = True
                    owner.grid = {k: (v.detach().cpu().contiguous() if isinstance(v, torch.Tensor) else v) for k, v in grid.items()}
                    owner.grid['weight_sha256'] = record['weight_sha256']
                    owner.target_calls += 1
                record['effective_damping'] = float(solver.effective_percdamp)
                record['status'] = 'PASS'
                owner.records.append(record)
                return result
            except BaseException:
                record['status'] = 'FAILED'
                owner.records.append(record)
                raise
            finally:
                owner.active = None

        @functools.wraps(self.original_quantize)
        def observed_quantize(x, scale, zero, maxq):
            result = owner.original_quantize(x, scale, zero, maxq)
            current = owner.active
            if current is not None and current['record']['captured']:
                column = current['column']
                record = current['record']
                rows, columns = record['shape']
                assert column < columns and tuple(x.shape) == (rows, 1)
                assert scale.dtype == zero.dtype == x.dtype == torch.float32
                assert tuple(scale.shape) == tuple(zero.shape) == (rows, 1)
                q = torch.clamp(torch.round(x / scale) + zero, 0, 7)
                current['codes'][:, column] = q[:, 0].to(torch.uint8)
                g = column // 128
                if column % 128 == 0:
                    current['scales'][:, g] = scale[:, 0]
                    current['zeros'][:, g] = zero[:, 0]
                else:
                    current['invalid'].logical_or_(torch.any(current['scales'][:, g] != scale[:, 0]))
                    current['invalid'].logical_or_(torch.any(current['zeros'][:, g] != zero[:, 0]))
                current['invalid'].logical_or_(torch.any(result != scale * (q - zero)))
                current['column'] += 1
            return result

        self.fast_wrapper = observed_fast
        self.quant_wrapper = observed_quantize
        gptq_module.GPTQ.fasterquant = observed_fast
        gptq_module.quantize = observed_quantize
        self.installed = True
        return self

    def restore(self):
        if self.installed:
            assert self.active is None
            assert self.native_module.GPTQ.fasterquant is self.fast_wrapper
            assert self.native_module.quantize is self.quant_wrapper
            self.native_module.GPTQ.fasterquant = self.original_fast
            self.native_module.quantize = self.original_quantize
            self.installed = False

    def finish(self):
        self.restore()
        assert self.grid is not None and self.target_calls == 1
        assert all(r['status'] == 'PASS' for r in self.records)
        grid = self.grid
        assert torch.isfinite(grid['scales']).all() and (grid['scales'] > 0).all()
        assert torch.isfinite(grid['zeros']).all()
        assert int(grid['codes'].min()) >= 0 and int(grid['codes'].max()) <= 7
        assert tensor_digest(decode_grid(grid)) == grid['weight_sha256']
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / 'target_grid.pt'
        assert not path.exists(), 'Never overwrite prior grid evidence'
        torch.save(grid, path)
        report = dict(status='AUTHORITATIVE_TARGET_GRID_ROUNDTRIP_PASS', target=self.target_name,
                      modules_observed=len(self.records), target_calls=self.target_calls,
                      target_shape=grid['shape'], group_size=128, bits=3, code_domain=[0, 7],
                      actual_symmetric=grid['actual_symmetric'], actorder=False,
                      grid_roundtrip_bits_equal=True, captured_grid_modules=1,
                      scope='Only target grid exported; other modules run native GPTQ without grid capture.',
                      weight_sha256=grid['weight_sha256'], native_source_sha256=self.source_sha256,
                      file=str(path), bytes=path.stat().st_size,
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest(), modules=self.records)
        with (self.out_dir / 'grid_roundtrip_report.json').open('x') as f:
            json.dump(report, f, indent=2); f.write('\n')
        return report


def legal_code_move(grid, row, column, direction):
    """Describe one fixed-grid move using the actual dequantization dtype."""
    assert direction in (-1, 0, 1)
    assert 0 <= row < grid['shape'][0] and 0 <= column < grid['shape'][1]
    old_q = int(grid['codes'][row, column])
    new_q = old_q + direction
    if not grid['min_code'] <= new_q <= grid['max_code']:
        return dict(row=row, column=column, old_q=old_q, requested_q=new_q,
                    eligible=False, reason='OUTSIDE_CODE_DOMAIN', delta_eff=0.0)
    g = column // grid['group_size']
    s = grid['scales'][row, g].detach().cpu()
    z = grid['zeros'][row, g].detach().cpu()
    dtype = _dtype(grid['weight_dtype'])
    old = (s * (torch.tensor(float(old_q), dtype=torch.float32) - z)).to(dtype)
    new = (s * (torch.tensor(float(new_q), dtype=torch.float32) - z)).to(dtype)
    delta = float(new.float() - old.float())
    return dict(row=row, column=column, old_q=old_q, new_q=new_q, eligible=True,
                old_value=float(old), new_value=float(new), delta_eff=delta,
                native_scale=float(s), native_zero=float(z), no_op=direction == 0,
                effective_change=delta != 0.0, weight_dtype=str(dtype))


def apply_code_move(weight, grid, row, column, direction):
    """Apply a sparse move from the baseline without changing grid metadata."""
    record = legal_code_move(grid, row, column, direction)
    assert record['eligible'], record
    old = torch.tensor(record['old_value'], dtype=weight.dtype, device=weight.device)
    assert torch.equal(weight.detach()[row, column], old), 'Coordinate not at its frozen baseline'
    with torch.no_grad():
        weight[row, column].copy_(torch.tensor(record['new_value'], dtype=weight.dtype, device=weight.device))
    return record


def restore_grid_weight(weight, grid):
    restored = decode_grid(grid, dtype=weight.dtype, device=weight.device)
    with torch.no_grad():
        weight.copy_(restored)
    assert tensor_digest(weight) == grid['weight_sha256']


def _differentiable_a16_forward(self, x):
    # Preserve the original x.float()/FP32 scale -> original dtype order.
    # Native quantize_activation_per_token_outlier returns x unchanged at A16.
    if self.input_scale is not None:
        x = (x.float() / self.input_scale.float()).to(x.dtype)
    return F.linear(x, self.weight, self.bias)


class DifferentiableInstall:
    def __init__(self, model, target_name):
        self.model = model
        self.target_name = target_name
        self.originals = {}
        keys = tuple(model.state_dict())
        model.eval().requires_grad_(False)
        for value in model.buffers():
            if value.is_floating_point() and value.requires_grad:
                assert value.is_leaf
                value.requires_grad_(False)
        target = None
        for name, module in model.named_modules():
            if module.__class__.__name__ != 'ImpQuantLinear':
                continue
            assert module.a_bit == 16 and module.w_bit == 3
            assert module.weight.is_leaf and not module.weight.requires_grad
            if module.input_scale is not None:
                assert tuple(module.input_scale.shape) == (module.in_features,)
                assert torch.isfinite(module.input_scale).all() and (module.input_scale > 0).all()
            self.originals[name] = (module, module.forward)
            module.forward = types.MethodType(_differentiable_a16_forward, module)
            if name == target_name:
                target = module
        assert target is not None, 'Target must be an existing ImpQuantLinear'
        target.weight.requires_grad_(True)
        self.target_weight = target.weight
        self.target_module = target
        self.keys_unchanged = tuple(model.state_dict()) == keys
        assert self.keys_unchanged
        assert not any(p.requires_grad for p in model.parameters())
        active = [name for name, value in model.named_buffers() if value.requires_grad]
        assert active == [target_name + '.weight'], active
        self.restored = False

    def report(self):
        return dict(status='DIFFERENTIABLE_A16_FORWARD_INSTALLED_NOT_YET_FULL_MODEL_VALIDATED',
                    replaced_modules=len(self.originals), target=self.target_name,
                    state_dict_keys_unchanged=self.keys_unchanged,
                    target_shape=list(self.target_weight.shape), target_is_buffer=True,
                    target_only_requires_grad=True, activation_bits=16,
                    input_scale_operation='x.float()/input_scale.float() -> original x dtype -> F.linear',
                    scope='Parent must verify complete real-model forward/loss bit equality before claiming the gate PASS.')

    def restore(self):
        assert not self.restored
        self.target_weight.grad = None
        self.target_weight.requires_grad_(False)
        for module, original in self.originals.values():
            module.forward = original
        self.restored = True


def install_differentiable_impquant(model, target_name):
    return DifferentiableInstall(model, target_name)
