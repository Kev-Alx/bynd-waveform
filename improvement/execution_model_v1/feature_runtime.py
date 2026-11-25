import sys,pathlib,json,time,os,random,gc,resource
P=pathlib.Path('/workspace/improvement');D=P/'execution_model_v1';sys.path[:0]=[str(P),str(D),'/workspace','/workspace/scripts']
import torch
import execution_runtime as rt
from feature_policy import FeaturePolicy,RADII,spectral_proxy,envelope_proxy,quality_gradient_mix
from scaled_components import checkpoint_blocks,input_gradient,transform

def status(**values):rt.atomic_json(D/'progress.json',{'updated_at_utc':rt.stamp(),**values});print(json.dumps(values),flush=True)
def check():
    plan=rt.read(D/'plan.json')
    for p,h in plan['bindings'].items():
        if rt.digest(p)!=h:raise ValueError(f'Changed registered file: {p}')
    old=rt.read(rt.RUN/'plan.json');exceptions=plan.get('preexisting_unrelated_baseline_drift',{})
    for path,h in old['frozen_files'].items():
        current=rt.digest(path)
        if current!=h:
            allowed=exceptions.get(path,{})
            if allowed.get('registered_sha256')!=h or allowed.get('current_sha256')!=current:raise ValueError(f'Unexpected original registration drift: {path}')
    return plan

def measured(fn):
    torch.cuda.synchronize();b=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);b.record();t=time.perf_counter();value=fn();e.record();torch.cuda.synchronize();return value,{'wall_seconds':time.perf_counter()-t,'gpu_stream_elapsed_seconds':b.elapsed_time(e)/1000}
def sources():
    old=rt.read(rt.RUN/'plan.json');vocoder,source,native,metadata=rt.models(old);checkpoint_blocks([*source.model.encoder.blocks,*source.model.decoder.blocks]);return vocoder,source,native,metadata

def microbatch(policy,q,reference,text,source,vocoder,rho,augmentation,quality,accumulation,caps):
    delta=policy(q,rho);rendered=vocoder(q+delta);live=rendered.detach().requires_grad_(True)
    nll=source.nll(transform(live,augmentation),text);tokens=len(source.tokenizer.encode(text.strip()));loss=-nll/tokens
    sg,=torch.autograd.grad(loss,live);sg=sg.detach()
    qlive=rendered.detach().requires_grad_(True);sp=spectral_proxy(qlive,reference);ep=envelope_proxy(qlive,reference);hinge=(sp/caps['spectral']-1).clamp_min(0)+(ep/caps['envelope']-1).clamp_min(0);active=bool(quality and hinge.detach()>0)
    if active:qg,=torch.autograd.grad(hinge,qlive)
    else:qg=torch.zeros_like(sg)
    mixed,mix=quality_gradient_mix(sg,qg.detach(),active);torch.autograd.backward(rendered,mixed/accumulation)
    if not torch.isfinite(mixed).all():raise ValueError('Nonfinite mixed gradient')
    return {'token_mean_nll':float(nll.detach())/tokens,'source_sum_nll':float(nll.detach()),'spectral':float(sp.detach()),'envelope':float(ep.detach()),'quality_hinge':float(hinge.detach()),'quality_active':active,'rho':rho,'actual_relative_radius':float(delta.detach().norm()/q.norm()),'source_audio_gradient_norm':float(sg.double().norm()),'quality_audio_gradient_norm':float(qg.detach().double().norm()),'combined_audio_gradient_norm':float(mixed.double().norm()),**mix,'source_backward_calls':1,'decoder_backward_calls':1,'decoder_forward_calls':1,'augmentation':augmentation}

def make_policy():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11);model=FeaturePolicy()
    return model.cuda()

def save_policy(policy,path,**metadata):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');torch.save({'model':{k:v.detach().cpu() for k,v in policy.state_dict().items()},'registration':policy.registration(),**metadata},temp);temp.replace(path);return {'path':str(path),'sha256':rt.digest(path)}
def load_policy(path,device='cuda'):
    saved=torch.load(path,map_location='cpu',weights_only=True);m=FeaturePolicy();m.load_state_dict(saved['model'],strict=True);return m.to(device),saved

def profile():
    plan=check();began=time.perf_counter();rows=rt.read_jsonl(P/'execution/data/train-10h.jsonl');chosen=[min(rows,key=lambda r:r['duration']),sorted(rows,key=lambda r:r['duration'])[len(rows)//2],max(rows,key=lambda r:r['duration'])];old=rt.read(rt.RUN/'plan.json')
    with rt.backend_mode('strict_math'):
        torch.set_num_threads(4);rt.seed_everything(11)
        from r3_real_runner_v2 import load_encoder_only
        native,encoder,meta=load_encoder_only(old);probe=[]
        for r in chosen:
            x=rt.load_audio(r['audio']).cuda()
            def encode():
                with torch.no_grad():return encoder(x).detach()
            q,t=measured(encode);probe.append((r,q.cpu(),x.cpu(),t))
        del native,encoder;gc.collect();torch.cuda.empty_cache();vocoder,source,native,metadata=sources();model=make_policy();optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0);caps=plan['quality_caps'];results=[];inference_probe=[];torch.cuda.reset_peak_memory_stats()
        for r,q,x,et in probe:
            q=q.cuda()
            with torch.no_grad():reference=vocoder(q).detach()
            for quality in [False,True]:
                for step in range(4):
                    optimizer.zero_grad(set_to_none=True)
                    row,t=measured(lambda:microbatch(model,q,reference,r['text'],source,vocoder,.05,'identity',quality,1,caps));norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));assert norm>0,'Policy gradient vanished in engineering profile';optimizer.step();results.append({'id':r['id'],'duration':r['duration'],'quality':quality,'warmup':step==0,'microbatch':row,'timing':t,'feature_encode':et,'parameter_preclip_norm':norm});status(stage='profile',id=r['id'],quality=quality,step=step+1)
            with torch.no_grad():
                same=model(q,.05);again=model(q,.05);assert torch.equal(same,again);assert abs(float(same.norm()/q.norm())-.05)<1e-5
                _,inference=measured(lambda:vocoder(q+model(q,.05)).detach());inference_probe.append({'id':r['id'],'audio_seconds':r['duration'],'encoding':et,'policy_decoder':inference})
            del q,reference
        peak=torch.cuda.max_memory_allocated();assert peak<20*1024**3,'Proposed20GiB engineering headroom exceeded'
        timings=[r['timing']['wall_seconds'] for r in results if not r['warmup']];mean=sum(timings)/len(timings);target_updates=min(6000,max(100,int(7200/(mean*4*1.30))));forecast={'measured_profile_microbatch_mean_seconds':mean,'microbatches':len(timings),'estimated_updates_per7200s_arm_with30pct_margin':target_updates,'proposed_arm_wall_seconds':7200,'common_warmup_updates':300,'forecast_warmup_seconds':300*4*mean,'forecast_two_training_arms_seconds':2*target_updates*4*mean,'forecast_pilot_scoring_seconds':48*8*5.31*1.5,'forecast_feature_preprocessing_seconds':len(rows)*(sum(r[3]['wall_seconds'] for r in probe)/3+.12),'inference_engineering_probe':inference_probe,'forecast_new1000_compute_seconds_excluding_io_loading':1000*sum(v['encoding']['wall_seconds']+v['policy_decoder']['wall_seconds'] for v in inference_probe)/3,'new_audio_forecast_limits':'Three engineering points/unfinished model, excludes IO/loading; actual48 trained timing and WER/quality must be measured before speed claims','larger_data_forecasts':[{'audio_hours':h,'assumed_records':len(rows)*h/10.0036557,'one_full_training_pass_compute_hours':len(rows)*h/10.0036557*mean/3600,'preprocessing_hours':len(rows)*h/10.0036557*(sum(r[3]['wall_seconds'] for r in probe)/3+.12)/3600,'basis':'Measured short/median/long training microbatch/encoding throughput; same duration distribution, batch1; training loading/optimizer overhead and convergence benefit are not established'} for h in [10,50,100]],'teacher_optimized_target_generation_seconds':0,'training_records':2807,'training_audio_hours':sum(r['duration'] for r in rows)/3600,'memory_peak_allocated_bytes':peak,'memory_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'wall_seconds':time.perf_counter()-began,'scope':'engineering correctness/memory/throughput only; no pilot performance claim','results':results,'status':'pass','source_metadata':metadata,'encoder_metadata':meta,'backend':rt.backend_manifest()};rt.atomic_json(D/'profile.json',forecast);status(stage='profile',status='complete',target_updates=target_updates)

def cache_data():
    plan=check();old=rt.read(rt.RUN/'plan.json');start=time.perf_counter();rows=rt.read_jsonl(P/'execution/data/train-10h.jsonl');folder=D/'training_cache';folder.mkdir(exist_ok=False);records=[]
    with rt.backend_mode('strict_math'):
        torch.set_num_threads(4);rt.seed_everything(11)
        from r3_real_runner_v2 import load_encoder_only
        native,encoder,meta=load_encoder_only(old);vocoder,_=rt.load_frozen_vocoder(old['base_config']['features'],old['feature_artifacts'],'cuda');torch.cuda.reset_peak_memory_stats()
        for i,r in enumerate(rows):
            if rt.digest(r['audio'])!=r['audio_sha256']:raise ValueError('Changed training recording')
            x=rt.load_audio(r['audio']).cuda()
            def generate_reference():
                with torch.no_grad():
                    q=encoder(x).detach();neutral=vocoder(q).detach();return q,neutral
            (q,neutral),timing=measured(generate_reference);qpath=folder/'features'/f'{r["id"]}.pt';qpath.parent.mkdir(exist_ok=True);torch.save(q.cpu(),qpath);wav=rt.save_audio(folder/'neutral'/f'{r["id"]}.wav',neutral);records.append({**r,'q':{'path':str(qpath),'sha256':rt.digest(qpath)},'neutral':wav,'preprocessing_timing':timing});
            if i%20==0 or i+1==len(rows):status(stage='preprocessing',completed=i+1,total=len(rows))
            del x,q,neutral
        rt.write_jsonl(folder/'manifest.jsonl',records);result={'status':'complete','count':len(rows),'training_hours':sum(r['duration'] for r in rows)/3600,'wall_seconds':time.perf_counter()-start,'compute_wall_seconds':sum(r['preprocessing_timing']['wall_seconds'] for r in records),'gpu_stream_elapsed_seconds':sum(r['preprocessing_timing']['gpu_stream_elapsed_seconds'] for r in records),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'manifest_sha256':rt.digest(folder/'manifest.jsonl'),'optimized_teacher_targets_generated':False,'encoder_metadata':meta};rt.atomic_json(folder/'receipt.json',result)

def train(arm):
    plan=check();profile=rt.read(D/'profile.json');rows=rt.read_jsonl(D/'training_cache/manifest.jsonl');assert len(rows)==2807;assert rt.digest(D/'training_cache/manifest.jsonl')==rt.read(D/'training_cache/receipt.json')['manifest_sha256'];folder=D/'training'/arm;folder.mkdir(parents=True,exist_ok=False);warm=arm=='warmup';maximum_updates=300 if warm else profile['estimated_updates_per7200s_arm_with30pct_margin'];maximum_seconds=1200 if warm else 7200;quality=arm=='quality';started=time.perf_counter();metrics=[];trace=folder/'trace.jsonl'
    with rt.backend_mode('strict_math'):
        torch.set_num_threads(4);rt.seed_everything(11);vocoder,source,native,meta=sources()
        if warm:
            model=make_policy();save_policy(model,D/'training/initial.pt',arm='initial',updates=0,seed=11)
        else:model,_=load_policy(D/'training/warmup/final.pt')
        model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0);optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();rng=__import__('numpy').random.default_rng(20261027 if warm else 20261028);order=[];steps=0;micro=0;data_time=0.;gpu_time=0.;clip_count=0;diagnostics={};source_model_frozen=not any(p.requires_grad for p in source.parameters())
        while steps<maximum_updates and time.perf_counter()-started<maximum_seconds-30:
            update_start=time.perf_counter();batch=[]
            for slot in range(4):
                if not order:order=rng.permutation(len(rows)).tolist()
                r=rows[order.pop()];t=time.perf_counter();q=torch.load(r['q']['path'],map_location='cuda',weights_only=True);ref=rt.load_audio(r['neutral']['path']).cuda();data_time+=time.perf_counter()-t;rho=RADII[int(rng.integers(0,3))];aug=['identity','identity','identity','fir6k','resample8k'][int(rng.integers(0,5))]
                row,timing=measured(lambda:microbatch(model,q,ref,r['text'],source,vocoder,rho,aug,quality,4,plan['quality_caps']));gpu_time+=timing['gpu_stream_elapsed_seconds'];micro+=1;batch.append(row);del q,ref
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));clip_count+=norm>5;optimizer.step();optimizer.zero_grad(set_to_none=True);steps+=1
            record={'update':steps,'microbatches':micro,'wall_seconds_cumulative':time.perf_counter()-started,'update_wall_seconds':time.perf_counter()-update_start,'parameter_preclip_norm':norm,'gradient_clipped':norm>5,'means':{k:sum(b[k] for b in batch)/4 for k in ['token_mean_nll','source_sum_nll','spectral','envelope','quality_hinge','actual_relative_radius','source_audio_gradient_norm','quality_audio_gradient_norm','combined_audio_gradient_norm','projection_fraction']},'quality_active_fraction':sum(b['quality_active'] for b in batch)/4};metrics.append(record)
            with trace.open('a') as f:f.write(json.dumps(record)+'\n')
            if steps==1 or steps%20==0:status(stage='training',arm=arm,updates=steps,max_updates=maximum_updates,elapsed_seconds=time.perf_counter()-started,last=record)
            if steps%500==0:save_policy(model,folder/'checkpoints'/f'update{steps:05d}.pt',arm=arm,updates=steps,microbatches=micro,seed=11)
        saved=save_policy(model,folder/'final.pt',arm=arm,updates=steps,microbatches=micro,seed=11);assert source_model_frozen and not any(p.grad is not None for p in vocoder.parameters())
        result={'status':'complete','arm':arm,'updates':steps,'microbatches':micro,'wall_seconds':time.perf_counter()-started,'data_loading_wall_seconds':data_time,'microbatch_gpu_stream_elapsed_seconds':gpu_time,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'gradient_clip_fraction':clip_count/max(1,steps),'maximum_wall_seconds':maximum_seconds,'stop_reason':'update_cap' if steps==maximum_updates else 'declared_training_wall_cap','model':saved,'source_metadata':meta,'policy':model.registration(),'frozen_source_decoder_gradients_absent':True,'teacher_optimized_data_cost_seconds':0,'formal_evaluation_cost_excluded':True,'training_manifest_sha256':rt.digest(D/'training_cache/manifest.jsonl')};rt.atomic_json(folder/'receipt.json',result);status(stage='training',arm=arm,status='complete',updates=steps)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['profile','cache','train']);p.add_argument('--arm',choices=['warmup','source','quality']);a=p.parse_args()
    if a.stage=='profile':profile()
    elif a.stage=='cache':cache_data()
    else:train(a.arm)
