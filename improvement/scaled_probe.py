"""Engineering-only fit/gradient probe; no decoded performance or target access."""
from pathlib import Path
import json
import time
import traceback
import torch
import execution_runtime as rt
from scaled_components import load_sources, BigVGAN16, calibrate_sources, ensemble_gradient

RUN = rt.OUT / 'execution_scaled'


def main():
    if (RUN / 'probe-result.json').exists(): raise FileExistsError('Preserve earlier real probe')
    assets = rt.read(RUN / 'assets.json')
    for model in assets['models']:
        for path, value in model['files'].items():
            if rt.digest(path) != value['sha256']: raise ValueError('Changed new source/decoder asset')
    plan = rt.read(rt.RUN / 'plan.json'); rt.check(plan)
    cohort = sorted(rt.read_jsonl(rt.OUT / 'execution48/development-48.jsonl'), key=lambda r:(r['duration'],r['id']))
    selected = [cohort[0], cohort[len(cohort)//2], cohort[-1]]
    files = [Path(__file__), rt.OUT / 'scaled_components.py', RUN / 'component-tests.log', RUN / 'assets.json',
        rt.ROOT / 'artifacts/whisper/large-v2.pt']
    rt.atomic_json(RUN / 'probe-registration.json', {'created_at_utc': rt.stamp(), 'ids': [r['id'] for r in selected],
        'scope': 'Three duration-selected engineering fit cases, not a pilot performance estimate; no decoding',
        'target_models_evaluated': [], 'bindings': {str(p):rt.digest(p) for p in files}})
    result = {'status': 'running', 'scope': 'engineering_only', 'target_models_evaluated': [], 'cases': []}
    began = time.perf_counter()
    try:
        with rt.backend_mode('strict_math'):
            torch.set_num_threads(4); rt.seed_everything(2026)
            models, metadata = load_sources(plan)
            vocoder, _ = rt.load_frozen_vocoder(plan['base_config']['features'], plan['feature_artifacts'], 'cuda')
            big = BigVGAN16()
            prior = {r['id']: r for r in rt.read_jsonl(rt.OUT / 'execution48/mechanism-pilot48/generated.jsonl')}
            for row in selected:
                uid=row['id']; start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
                audio=rt.load_audio(row['audio']).cuda()
                asset=prior[uid]['clean_features']
                if rt.digest(asset['path']) != asset['sha256']: raise ValueError('Changed pilot reference')
                q=torch.load(asset['path'],map_location='cuda',weights_only=True)
                with torch.no_grad(): reconstructed=vocoder(q)
                scales, calibration=calibrate_sources(models,reconstructed,row['text'])
                gradients=[]; transforms=[]
                for augmentation in ('identity','fir6k','resample8k'):
                    before=time.perf_counter()
                    gradient,values=ensemble_gradient(models,reconstructed,row['text'],scales,augmentation)
                    torch.cuda.synchronize()
                    transforms.append({'name':augmentation,'seconds':time.perf_counter()-before,'components':values})
                    gradients.append(gradient)
                live=q.detach().requires_grad_(); rendered=vocoder(live)
                if not torch.equal(rendered,reconstructed): raise ValueError('Vocoder detached/VJP forwards differ')
                derivative,=torch.autograd.grad(rendered,live,grad_outputs=gradients[0])
                if not torch.isfinite(derivative).all() or derivative.norm()==0: raise ValueError('Invalid feature VJP')
                fv_peak=torch.cuda.max_memory_allocated()
                del live,rendered,derivative,gradients
                torch.cuda.reset_peak_memory_stats()
                mel=big.encode(audio)
                with torch.no_grad(): big_audio=big(mel)
                mel_live=mel.detach().requires_grad_()
                output=big(mel_live)
                if not torch.equal(output.detach(),big_audio): raise ValueError('BigVGAN detached/gradient forwards differ')
                derivative,=torch.autograd.grad(output.square().mean(),mel_live)
                if not torch.isfinite(derivative).all() or derivative.norm()==0: raise ValueError('Invalid BigVGAN derivative')
                if any(p.grad is not None for m in [*models.values(),vocoder,big] for p in m.parameters()):
                    raise ValueError('Frozen weights acquired parameter gradients')
                case={'id':uid,'duration':row['duration'],'status':'pass','calibration':calibration,'transforms':transforms,
                    'fv_peak_allocated_bytes':fv_peak,'bigvgan_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'bigvgan_parameters':sum(p.numel() for p in big.parameters()),'bigvgan_original_samples':len(audio),
                    'bigvgan_output_samples':len(output),'bigvgan_peak':float(output.detach().abs().max()),
                    'bigvgan_clipped_after_resample':big.last_clipped_samples,'wall_seconds':time.perf_counter()-start}
                result['cases'].append(case); rt.atomic_json(RUN/'probe-result.json',result)
                print(json.dumps(case),flush=True)
                del q,audio,reconstructed,mel,mel_live,output,derivative,big_audio
            result.update(status='pass',source_metadata=metadata,backend=rt.backend_manifest(),wall_seconds=time.perf_counter()-began)
            rt.atomic_json(RUN/'probe-result.json',result)
    except Exception as error:
        result.update(status='failed_execution',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-began)
        rt.atomic_json(RUN/'probe-result.json',result); raise


if __name__ == '__main__': main()
