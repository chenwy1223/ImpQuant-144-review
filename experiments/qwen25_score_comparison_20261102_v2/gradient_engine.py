"""Native end-loss gradients projected onto six already frozen sparse edits.

This module never chooses coordinates, modifies a weight, assembles an arm,
saves a full gradient, resets peak memory or retries a failed sample. The caller
must authenticate the old exact32-code receipts, complete baseline state and
prepared inputs, and must measure a native teacher reference for each new row.
"""
from contextlib import contextmanager
import gc
import importlib
import math
from pathlib import Path
import sys
import time

INDICES = (7, 11, 15, 19, 23, 27)
MODULES = tuple('model.layers.%d.self_attn.v_proj' % i for i in INDICES)
SHAPE = (512, 3584)
COUNT = 32


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def validate_edits(edits, shape=SHAPE):
    """Validate the fixed receipt structure; authority-grid checks remain upstream."""
    require(isinstance(edits, list) and len(edits) == COUNT, 'Exactly32 saved moves')
    coordinates = set()
    for edit in edits:
        r, c = edit['row'], edit['column']
        require(type(r) is int and type(c) is int and 0 <= r < shape[0] and
                0 <= c < shape[1], 'Existing integer coordinate')
        require((r, c) not in coordinates, 'Saved coordinates must be distinct')
        coordinates.add((r, c))
        old_q, new_q, direction = edit['old_q'], edit['new_q'], edit['direction']
        require(type(old_q) is int and type(new_q) is int and type(direction) is int and
                direction in (-1, 1) and 1 <= old_q <= 6 and
                new_q == old_q + direction and 0 <= new_q <= 7,
                'Exactly one legal previously selected integer step')
        old, new, delta = edit['old_value'], edit['new_value'], edit['delta_eff']
        require(all(type(x) in (int, float) and math.isfinite(x) for x in (old, new, delta)) and
                delta != 0 and float(new) - float(old) == float(delta),
                'Actual finite nonzero BF16-grid delta')
    return tuple((x['row'], x['column']) for x in edits)


def projected_score(selected_gradients, edits):
    """FP64 scalar arithmetic, retaining all32 signed coordinate contributions."""
    validate_edits(edits)
    require(len(selected_gradients) == COUNT and
            all(type(g) in (int, float) and math.isfinite(g) for g in selected_gradients),
            'Exactly32 finite selected gradients')
    delta = [float(x['delta_eff']) for x in edits]
    terms = [-float(g) * d for g, d in zip(selected_gradients, delta)]
    require(all(math.isfinite(x) for x in terms), 'Finite first-order terms')
    value = math.fsum(terms)
    require(math.isfinite(value), 'Finite projected predicted decrease')
    return dict(selected_gradients_FP32=[float(x) for x in selected_gradients],
                actual_deltas_FP64=delta, negative_gradient_times_delta_FP64=terms,
                predicted_decrease_FP64=value,
                summation='math.fsum of32 FP64 (-FP32_gradient * actual_BF16_delta) terms',
                all_zero_selected_gradient=all(float(x) == 0. for x in selected_gradients),
                new_coordinate_selection_calls=0, new_repair_calls=0)


def aggregate_selected(records, edits):
    """Preserve both rounding conventions rather than silently changing the old rule.

    The selected32 FP32 mean is retained for the mean-gradient interpretation.
    Its reduction layout differs from the old full1.8M-coordinate tensor mean;
    no bit-identity to that historical reduction is claimed. Per-row projection
    means and the rounding residual are always reported alongside it.
    """
    import torch
    require(records, 'Nonempty fixed scoring rows')
    values = torch.tensor([r['selected_gradients_FP32'] for r in records], dtype=torch.float32)
    require(tuple(values.shape) == (len(records), COUNT), 'Complete selected-gradient rows')
    mean = values.mean(dim=0).contiguous()
    result = projected_score(mean.tolist(), edits)
    row_mean = math.fsum(r['predicted_decrease_FP64'] for r in records) / len(records)
    result.update(rows=len(records), mean_gradient_rule='torch.tensor(rows32_FP32).mean0',
                  mean_per_row_FP64_projected_decrease=row_mean,
                  FP32_mean_minus_mean_row_projection=result['predicted_decrease_FP64'] - row_mean,
                  historical_full_gradient_mean_bit_identity_claimed=False)
    return result


def _operators():
    from common import RANK_HERE
    folder = Path(RANK_HERE).resolve()
    # The parent execution lock must bind these exact unchanged helper files.
    if str(folder) not in sys.path:
        sys.path.append(str(folder))
    editable = importlib.import_module('editable_grid')
    suffix = importlib.import_module('checkpoint_suffix')
    require(Path(editable.__file__).resolve() == folder/'editable_grid.py' and
            Path(suffix.__file__).resolve() == folder/'checkpoint_suffix.py',
            'Only authenticated frozen native helper files may be imported')
    return editable.install_differentiable_impquant, suffix.SuffixCheckpoint


def _capture(model):
    modules = dict(model.named_modules())
    require(not any(m.training for m in modules.values()), 'All modules already eval')
    params = dict(model.named_parameters()); buffers = dict(model.named_buffers())
    require(not any(x.requires_grad or x.grad is not None for x in params.values()) and
            not any(x.requires_grad or x.grad is not None for x in buffers.values()),
            'Initially every parameter/buffer is frozen and without accumulated gradients')
    imp = {n: m.forward for n, m in modules.items() if m.__class__.__name__ == 'ImpQuantLinear'}
    require(len(imp) == 196, 'Exactly196 original native quantized modules')
    blocks = {i: model.get_submodule('model.layers.%d' % i).forward for i in range(28)}
    decoder = model.get_submodule('model')
    require(model.config.use_cache is False and decoder.config.use_cache is False,
            'Native end-loss route forbids mutable KV cache')
    return dict(keys=tuple(model.state_dict()),
                parameter_flags={n: x.requires_grad for n, x in params.items()},
                buffer_flags={n: x.requires_grad for n, x in buffers.items()},
                training={n: m.training for n, m in modules.items()},
                imp_forwards=imp, block_forwards=blocks,
                model_cache=model.config.use_cache, decoder_cache=decoder.config.use_cache)


def _check_restored(model, before):
    require(tuple(model.state_dict()) == before['keys'], 'All state_dict keys unchanged')
    require({n: x.requires_grad for n, x in model.named_parameters()} == before['parameter_flags'] and
            {n: x.requires_grad for n, x in model.named_buffers()} == before['buffer_flags'],
            'Exact original requires_grad flags restored')
    require(not any(x.grad is not None for _, x in model.named_parameters()) and
            not any(x.grad is not None for _, x in model.named_buffers()),
            'No accumulated parameter or buffer gradient after projection')
    require({n: m.training for n, m in model.named_modules()} == before['training'],
            'Every original eval flag restored')
    require(model.config.use_cache is before['model_cache'] and
            model.get_submodule('model').config.use_cache is before['decoder_cache'],
            'Original cache flags unchanged')
    require(all(model.get_submodule(n).forward == f for n, f in before['imp_forwards'].items()),
            'All196 ImpQuant native forwards restored')
    require(all(model.get_submodule('model.layers.%d' % i).forward == f
                for i, f in before['block_forwards'].items()),
            'All28 native decoder forwards restored')
    return True


@contextmanager
def _temporary_route(model, module):
    require(module in MODULES, 'Only the six fixed candidate v_proj modules')
    before = _capture(model)
    install, suffix_type = _operators()
    handle = checkpoint = None
    report = {}
    scientific_error = None
    try:
        handle = install(model, module)
        checkpoint = suffix_type(model, int(module.split('.')[2])).install()
        require(handle.report()['replaced_modules'] == 196 and
                handle.report()['target_is_buffer'] is True and
                not any(x.requires_grad for x in model.parameters()),
                'Original one-target buffer route with frozen parameters')
        yield handle, checkpoint, report
    except BaseException as exc:
        scientific_error = exc
        raise
    finally:
        # Restore both independent installations even if the first cleanup fails.
        cleanup_errors = []
        if checkpoint is not None:
            try:
                report['checkpoint_before_restore'] = checkpoint.report()
                checkpoint.restore()
                report['checkpoint_after_restore'] = checkpoint.report()
            except BaseException as exc:
                cleanup_errors.append('checkpoint: '+repr(exc))
        if handle is not None:
            try:
                report['installation'] = handle.report()
                handle.restore()
                report['differentiable_restore_complete'] = handle.restored is True
            except BaseException as exc:
                cleanup_errors.append('differentiable: '+repr(exc))
        try:
            _check_restored(model, before)
            report.update(all196_native_forwards_restored=True,
                          all28_decoder_forwards_restored=True,
                          all_tensor_gradient_flags_and_training_restored=True)
        except BaseException as exc:
            cleanup_errors.append('restored state: '+repr(exc))
        if scientific_error is not None:
            # Caller may serialize this without performing another model call.
            try:
                scientific_error.gradient_route_report = dict(report)
            except (AttributeError, TypeError):
                pass
        if cleanup_errors:
            raise AssertionError('Gradient route cleanup failed; no retry: '+'; '.join(cleanup_errors))


def _teacher(model, adapter, row, module, manifest_sha256, model_state_root):
    import torch
    from common import NAME, utc
    from native_contracts import load_prepared, scalar_diagnostics
    require(not model.training and model.config.use_cache is False and
            not any(x.requires_grad or x.grad is not None for x in model.parameters()),
            'Frozen native gradient-forward context')
    require({n for n, x in model.named_buffers() if x.requires_grad} == {module+'.weight'},
            'Only the current target buffer may require a gradient')
    sample, positions, measured, prefix_records = load_prepared(row)
    eos = adapter.tokenizer.eos_token_id
    require(type(eos) is int and eos not in sample['answer_token_ids'], 'Answer excludes EOS')
    target = model.get_submodule(module).weight
    gpu = {n: x.to(target.device) for n, x in sample['inputs'].items()}
    started = time.monotonic()
    with torch.enable_grad():
        output = model(**gpu, labels=None, use_cache=False, past_key_values=None,
                       output_attentions=False, output_hidden_states=False, return_dict=True)
        require(output.loss is None and output.past_key_values is None and
                output.logits.ndim == 3 and output.logits.shape[:2] == gpu['input_ids'].shape and
                output.logits.shape[-1] == model.config.vocab_size, 'Actual full native answer logits')
        selected = output.logits[0, positions, :].float()
        loss, loss_record = scalar_diagnostics(selected, sample['answer_token_ids'],
                                             sample['option_token_ids'], sample['canonical_target'], eos)
        require(loss.requires_grad is True, 'End loss remains attached to the exact native graph')
    r = dict(attempt=NAME, utc=utc(), arm='quant_baseline', id=row['id'], task=row['task'],
             doc_id=row['doc_id'], source_index=row['source_index'], role=row['role'],
             call_role='score64_gradient', module=module, image_sha256=row['sha256'],
             pixel_sha256=row['pixel_sha256'], target=row['target'], canonical_target=row['canonical_target'],
             prepared_input={k: row['prepared'][k] for k in ('path', 'bytes', 'sha256')},
             prefix_input_identity=row['prefix_input_identity'], full_input_ids=row['full_input_ids'],
             prompt_length=row['prompt_length'], answer_positions=row['answer_positions'],
             causal_prediction_positions=positions, target_token_ids=row['answer_token_ids'],
             labels_sha256=measured['labels']['sha256'], input_tensor_hashes=measured,
             prefix_tensor_hashes=prefix_records, manifest_sha256=manifest_sha256,
             model_state_root=model_state_root, diagnostic_forward_seconds=time.monotonic()-started,
             teacher_forcing=dict(answer_only=True, prompt_image_header_end_not_supervised=True,
                                  labels_passed_to_model=False, use_cache=False, past_key_values=None,
                                  target_positions_shift_minus_one=True), **loss_record)
    # Only loss retains the native autograd graph; output and input containers
    # themselves are not retained in the returned evidence object.
    del output, selected, gpu, sample
    return loss, r


def project_one(model, adapter, row, module, edits, phase_callback, *, native_reference,
                manifest_sha256, model_state_root):
    """Return one real gradient projection plus teacher and restoration evidence.

    Native baseline reference is mandatory and must precede this call. Its
    independent no-gradient forward is accounted by the caller. This function
    performs exactly one forward and one autograd.grad call, then fully restores
    its temporary route. Parent must audit complete full-model fingerprints at
    the declared stage boundaries and preserve exceptions/failure evidence.
    """
    import torch
    from native_contracts import tensor_digest, verify_A_equivalence
    require(module in MODULES, 'Fixed candidate only')
    coordinates = validate_edits(edits)
    require(native_reference['id'] == row['id'] and native_reference['task'] == row['task'] and
            native_reference['role'] == row['role'] and native_reference['arm'] == 'quant_baseline' and
            native_reference['manifest_sha256'] == manifest_sha256 and
            native_reference['model_state_root'] == model_state_root,
            'Authenticated independent native baseline reference for this exact row')
    target = model.get_submodule(module).weight
    require(target.dtype == torch.bfloat16 and tuple(target.shape) == SHAPE,
            'Original BF16 v_proj weight shape')
    for (r, c), edit in zip(coordinates, edits):
        require(float(target.detach()[r, c]) == float(edit['old_value']),
                'Every fixed coordinate starts at the complete baseline value')
        new_value = torch.tensor(edit['new_value'], dtype=torch.bfloat16)
        require(float(new_value) == float(edit['new_value']), 'Actual saved new value is BF16 representable')
    weight_before = tensor_digest(target)
    start = time.monotonic(); phase_callback('score64_gradient_projection', module=module,
                                           task=row['task'], id=row['id'], role=row['role'])
    with _temporary_route(model, module) as (handle, checkpoint, route):
        loss, teacher = _teacher(model, adapter, row, module, manifest_sha256, model_state_root)
        verify_A_equivalence(native_reference, teacher)
        gradient = torch.autograd.grad(loss, handle.target_weight, create_graph=False,
                                       retain_graph=False, allow_unused=False)[0]
        require(gradient.dtype == torch.bfloat16 and tuple(gradient.shape) == SHAPE and
                bool(torch.isfinite(gradient).all()), 'Finite actual native BF16 target gradient')
        cpu = gradient.detach().float().cpu().contiguous()
        selected = [float(cpu[r, c]) for r, c in coordinates]
        projection = projected_score(selected, edits)
        bits = gradient.detach().cpu().contiguous().view(torch.int16)
        projection.update(module=module, id=row['id'], task=row['task'], role=row['role'],
                          coordinates=[list(x) for x in coordinates],
                          selected_native_BF16_bits=[int(bits[r, c]) & 0xffff for r, c in coordinates],
                          gradient_native_dtype='torch.bfloat16', gradient_saved_selected_dtype='torch.float32',
                          complete_gradient_FP32_sha256=tensor_digest(cpu),
                          complete_gradient_nonzero=int((cpu != 0).sum()),
                          complete_gradient_norm_FP64=float(cpu.double().norm()),
                          complete_gradient_saved=False, native_forward_bits_equal=True,
                          new_gradient_calls=1, teacher_forced_forward_calls=1,
                          cuda_peak_statistics_reset=False)
        require(handle.target_weight.grad is None and
                not any(x.grad is not None for x in model.parameters()), 'No gradient accumulation')
        checkpoint_report = checkpoint.report()
        require(all(v > 0 for v in checkpoint_report['checkpoint_forward_calls'].values()),
                'Every declared native suffix block was reached by the gradient forward')
        del gradient, cpu, bits, loss
    require(tensor_digest(target) == weight_before, 'Gradient projection never modifies the target weight')
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    teacher.update(gradient_native_dtype='torch.bfloat16', native_A_forward_bits_equal=True)
    return dict(teacher=teacher, projection=projection, route=route,
                costs=dict(seconds=time.monotonic()-start, teacher_forced_forward_calls=1,
                           new_gradient_calls=1, full_gradient_files=0, new_coordinate_selections=0,
                           new_repair_calls=0, new_generation_calls=0))
