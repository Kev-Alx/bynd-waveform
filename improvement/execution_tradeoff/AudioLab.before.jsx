import React, { useEffect, useMemo, useRef, useState } from 'react';
import { DataComponent, SectionHeader, useDataApp, useDashboardTabs } from '../../data-app-public.jsx';
import './audio.css';

const TABS=[{id:'listen',label:'Listen & compare'},{id:'results',label:'All results'},{id:'explanation',label:'Changes & next steps'}];
const DEFAULTS=['scaled-fv--clean','scaled-fv--fv_quality50','scaled-fv--ensemble_eot_200'];
const fmt=(n,d=2)=>Number.isFinite(n)?n.toFixed(d):'—';
const clock=n=>`${Math.floor((n||0)/60)}:${String(Math.floor((n||0)%60)).padStart(2,'0')}`;
const groups=rows=>[...new Set(rows.map(r=>r.study))];
const fileUrl=path=>new URL(path,window.location.origin+'/').href;

function downloadCsv(rows){
  const keys=['study','name','count','referenceWords','wer','dnsmos','nisqa','utmos','qualityPass','goalPass','failedMetrics'];
  const quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';
  const blob=new Blob([[keys,...rows.map(r=>keys.map(k=>r[k]))].map(r=>r.map(quote).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='speech-results.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
}

function Transcript({reference,hypothesis}){
  // Illustrative display alignment. Official WER comes from saved score receipts.
  const a=reference.toLowerCase().replace(/[^a-z0-9' ]/g,' ').split(/\s+/).filter(Boolean);
  const b=hypothesis.toLowerCase().replace(/[^a-z0-9' ]/g,' ').split(/\s+/).filter(Boolean);
  const d=Array.from({length:a.length+1},(_,i)=>Array.from({length:b.length+1},(_,j)=>i?j?0:i:j));
  for(let i=1;i<=a.length;i++)for(let j=1;j<=b.length;j++)d[i][j]=Math.min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(a[i-1]===b[j-1]?0:1));
  let i=a.length,j=b.length;const out=[];
  while(i||j){
    if(i&&j&&d[i][j]===d[i-1][j-1]+(a[i-1]===b[j-1]?0:1)){out.push({word:b[j-1],type:a[i-1]===b[j-1]?'equal':'substitution',original:a[i-1]});i--;j--;}
    else if(j&&d[i][j]===d[i][j-1]+1){out.push({word:b[j-1],type:'insertion'});j--;}
    else{out.push({word:a[i-1],type:'deletion'});i--;}
  }
  return <p className="lab-transcript">{out.reverse().map((v,n)=><React.Fragment key={n}><span className={'word-'+v.type} title={v.type==='substitution'?`Reference: ${v.original}`:v.type==='deletion'?'Missing from recognizer transcript':v.type==='insertion'?'Extra recognizer word':undefined}>{v.word}</span>{' '}</React.Fragment>)}{!b.length&&<em>No words recognized.</em>}</p>;
}

function Metric({label,value,bad=false,suffix=''}){return <div className={'lab-metric '+(bad?'is-low':'')}><span>{label}</span><strong>{fmt(value)}{suffix}</strong></div>;}

function AudioPane({slot,runId,run,runs,score,clip,onRun,audioRefs,onStart,rate,linked}){
  const audio=useRef(null);const [time,setTime]=useState(0),[error,setError]=useState(''),[playing,setPlaying]=useState(false);
  useEffect(()=>{audioRefs.current[slot]=audio.current;return()=>{delete audioRefs.current[slot];};},[slot,score?.audio]);
  useEffect(()=>{setTime(0);setError('');setPlaying(false);},[score?.audio]);
  useEffect(()=>{if(audio.current)audio.current.playbackRate=rate;},[rate,score?.audio]);
  const seek=value=>{if(audio.current&&Number.isFinite(audio.current.duration)){audio.current.currentTime=Math.max(0,Math.min(value,audio.current.duration));setTime(audio.current.currentTime);}};
  return <DataComponent id={'audio-pane-'+slot} queryId="recordings" kind="custom" title={'Run '+['A','B','C'][slot]} variant="card" displayRows={score?[score]:[]} sourceRows={score?[score]:[]}>
    <div className="lab-pane" data-reviewed-rows>
      <div className="lab-run-name">{run?.name}</div><label className="lab-run-select"><span>Condition</span><select aria-label={'Run '+['A','B','C'][slot]} title={run?.name} value={runId} onChange={e=>onRun(slot,e.target.value)}>{groups(runs).map(group=><optgroup key={group} label={group}>{runs.filter(r=>r.study===group).map(r=><option key={r.id} value={r.id}>{r.name} · {r.count} clips</option>)}</optgroup>)}</select></label>
      <p className="lab-run-description">{run?.description}</p>
      <div className="lab-run-summary"><span>{run?.count}-clip mean WER <b>{fmt(run?.wer)}%</b></span><span className={'lab-badge '+(run?.qualityPass?'pass':'fail')}>{run?.qualityPass?'Quality passes':'Quality fails'}</span></div>
      {!score?<div className="lab-empty">This sentence was not part of this run’s {run?.count}-clip cohort. Choose another sentence or run.</div>:<>
        <div className="lab-wave"><svg viewBox="0 0 480 82" role="img" aria-label="Actual waveform peak amplitudes, common zero-to-one scale">{score.waveform.map((v,i)=><rect key={i} x={i*5} y={41-Math.max(1,v*38)} width="3" height={Math.max(2,v*76)} rx="1" fill={i/score.waveform.length<=time/score.duration?'var(--accent)':'var(--border-strong)'}/>)}<line x1="0" x2="480" y1="41" y2="41" stroke="var(--border)"/></svg><input className="lab-seek" type="range" aria-label={'Seek run '+['A','B','C'][slot]} min="0" max={score.duration} step="0.05" value={Math.min(time,score.duration)} onChange={e=>seek(Number(e.target.value))}/><div className="lab-time"><span>{clock(time)}</span><span>{clock(score.duration)}</span></div></div>
        <audio key={score.audio} ref={audio} controls preload="metadata" src={fileUrl(score.audio)} onLoadedMetadata={()=>{audio.current.playbackRate=rate;}} onTimeUpdate={()=>setTime(audio.current?.currentTime||0)} onPlay={()=>{setPlaying(true);onStart(slot,linked);}} onPause={()=>setPlaying(false)} onEnded={()=>setPlaying(false)} onError={()=>setError('Audio could not load. Retry or download the WAV below.')} aria-label={`${run?.name}: ${clip.id}`}/>
        {error&&<p role="alert" className="lab-error">{error}</p>}
        <div className="lab-audio-actions"><button onClick={()=>seek(0)}>↺ Restart</button><a href={fileUrl(score.audio)} download={`${clip.id}-${run?.variant}.wav`}>Download WAV ↗</a><span aria-live="polite">{playing?'Playing':''}</span></div>
        <div className="lab-metrics"><Metric label="Clip WER" value={score.wer} suffix="%"/><Metric label="DNSMOS" value={score.dnsmos} bad={score.dnsmos<3}/><Metric label="NISQA" value={score.nisqa} bad={score.nisqa<3.75}/><Metric label="UTMOS" value={score.utmos} bad={score.utmos<3.5}/></div>
        <div className="lab-transcript-heading">Recognizer transcript</div><Transcript reference={clip.reference} hypothesis={score.hypothesis}/>
        <div className="lab-errors">{score.substitutions} substitutions · {score.deletions} deletions · {score.insertions} insertions</div><details className="lab-original-transcript"><summary>Exact saved transcript</summary><p>{score.hypothesis||'(empty)'}</p></details>
      </>}
    </div>
  </DataComponent>;
}

function Listening({runs,recordings,clips,chosen,setChosen}){
  const [selected,setSelected]=useState('84-121123-0023'),[search,setSearch]=useState(''),[onlyShared,setOnlyShared]=useState(false),[panes,setPanes]=useState(3),[rate,setRate]=useState(1),[linked,setLinked]=useState(true);
  const audioRefs=useRef({});const scoreMap=useMemo(()=>new Map(recordings.map(r=>[r.runId+'|'+r.clipId,r])),[recordings]);
  const active=chosen.slice(0,panes);const runMap=useMemo(()=>new Map(runs.map(r=>[r.id,r])),[runs]);
  const visible=clips.filter(c=>(!onlyShared||active.every(id=>scoreMap.has(id+'|'+c.id)))&&(`${c.id} ${c.reference} ${c.speaker}`.toLowerCase().includes(search.toLowerCase())));
  const clip=clips.find(c=>c.id===selected)||clips[0];
  function stop(){Object.values(audioRefs.current).forEach(a=>a?.pause());}
  useEffect(()=>{stop();return stop;},[selected,panes]);useEffect(()=>{stop();},[chosen]);
  function onRun(slot,id){stop();setChosen(v=>v.map((r,i)=>i===slot?id:r));}
  function onStart(slot,carry){const other=Object.entries(audioRefs.current).find(([i,a])=>+i!==slot&&a&&!a.paused);if(carry&&other&&audioRefs.current[slot])audioRefs.current[slot].currentTime=Math.min(other[1].currentTime,audioRefs.current[slot].duration||Infinity);Object.entries(audioRefs.current).forEach(([i,a])=>{if(+i!==slot)a?.pause();});}
  const index=visible.findIndex(c=>c.id===selected);function next(delta){const n=visible[index+delta];if(n)setSelected(n.id);}
  return <><div className="lab-listen-toolbar"><div className="lab-presets"><button onClick={()=>{stop();setPanes(3);setChosen(DEFAULTS);}}>Clean / original / ensemble</button><button onClick={()=>{stop();setPanes(3);setChosen(['scaled-mel--clean','scaled-fv--fv_quality50','scaled-mel--mel_eot_200']);}}>Compare BigVGAN</button></div><div className="lab-pane-count" role="group" aria-label="Comparison panels">{[1,2,3].map(n=><button key={n} aria-pressed={panes===n} onClick={()=>{stop();setPanes(n);}}>{n} {n===1?'panel':'panels'}</button>)}</div></div>
    <div className="lab-listening-layout"><aside className="lab-library"><div className="lab-library-heading"><strong>Sentences</strong><span>{visible.length} / {clips.length}</span></div><label className="lab-search"><span className="sr-only">Search sentence, ID or speaker</span><input type="search" placeholder="Search words, ID, speaker…" value={search} onChange={e=>setSearch(e.target.value)}/></label><label className="lab-check"><input type="checkbox" checked={onlyShared} onChange={e=>setOnlyShared(e.target.checked)}/>Only sentences in every selected run</label><div className="lab-clip-list">{visible.map((c,i)=><button key={c.id} className={'lab-clip '+(c.id===selected?'selected':'')} aria-pressed={c.id===selected} onClick={()=>setSelected(c.id)}><span className="lab-clip-number">{String(i+1).padStart(2,'0')}</span><span><strong>{c.reference.toLowerCase()}</strong><small>{c.id} · {fmt(c.duration,1)}s</small></span></button>)}{!visible.length&&<p className="lab-empty">No matching sentences. Clear the search or shared-sentence filter.</p>}</div></aside>
      <section className="lab-comparison" aria-label="Audio comparison"><div className="lab-sentence"><div className="lab-sentence-top"><span>{clip.id} · speaker {clip.speaker}</span><div><button disabled={index<=0} onClick={()=>next(-1)} aria-label="Previous sentence">←</button><button disabled={index<0||index>=visible.length-1} onClick={()=>next(1)} aria-label="Next sentence">→</button></div></div><div className="lab-transcript-heading">Reference transcript</div><p>{clip.reference.toLowerCase()}</p></div>
        <div className="lab-playback-bar"><button onClick={stop}>■ Stop audio</button><label>Speed <select aria-label="Playback speed" value={rate} onChange={e=>setRate(Number(e.target.value))}><option value="0.75">0.75×</option><option value="1">1×</option><option value="1.25">1.25×</option></select></label><label className="lab-check"><input type="checkbox" checked={linked} onChange={e=>setLinked(e.target.checked)}/>Keep position when switching playback</label></div>
        <div className="lab-audio-grid" style={{'--panes':panes}}>{active.map((id,i)=><AudioPane key={i} slot={i} runId={id} run={runMap.get(id)} runs={runs} clip={clip} score={scoreMap.get(id+'|'+clip.id)} onRun={onRun} audioRefs={audioRefs} onStart={onStart} rate={rate} linked={linked}/>)}</div>
        <p className="lab-footnote">Original WAV samples and saved volume. One recording plays at a time. Waveforms share an amplitude scale. Quality thresholds apply to each full run’s mean; the card metrics describe this sentence.</p><div className="lab-word-key"><span className="word-substitution">Changed word</span><span className="word-insertion">Extra word</span><span className="word-deletion">Missing word</span><span>Display alignment is illustrative; exact WER is from the saved scorer.</span></div>
      </section></div></>;
}

function Results({runs}){
  const [cohort,setCohort]=useState('48'),[study,setStudy]=useState('all'),[quality,setQuality]=useState('all'),[query,setQuery]=useState(''),[sort,setSort]=useState('study');
  const options=runs.filter(r=>cohort==='all'||r.count===Number(cohort));
  const rows=options.filter(r=>(study==='all'||r.study===study)&&(quality==='all'||(quality==='pass'?r.qualityPass:!r.qualityPass))&&(`${r.name} ${r.variant} ${r.study}`.toLowerCase().includes(query.toLowerCase()))).sort((a,b)=>sort==='wer'?b.wer-a.wer:sort==='utmos'?b.utmos-a.utmos:a.study.localeCompare(b.study)||a.name.localeCompare(b.name));
  return <section className="lab-results"><SectionHeader id="all-results-heading" title="Complete results"/><p className="lab-intro">Higher WER means more source-recognition errors; higher quality scores are better. The goal requires WER &gt; 75%, DNSMOS ≥ 3, NISQA ≥ 3.75 and UTMOS ≥ 3.5 together.</p><div className="lab-table-controls"><label>Cohort<select aria-label="Cohort" value={cohort} onChange={e=>{setCohort(e.target.value);setStudy('all');}}><option value="48">48 recordings</option><option value="24">24 recordings</option><option value="all">Both · shown separately</option></select></label><label>Study<select aria-label="Study" value={study} onChange={e=>setStudy(e.target.value)}><option value="all">All studies</option>{groups(options).map(s=><option key={s}>{s}</option>)}</select></label><label>Quality<select aria-label="Quality" value={quality} onChange={e=>setQuality(e.target.value)}><option value="all">All conditions</option><option value="pass">Passes all floors</option><option value="fail">Fails a floor</option></select></label><label>Sort<select aria-label="Sort" value={sort} onChange={e=>setSort(e.target.value)}><option value="study">Study and name</option><option value="wer">Highest WER</option><option value="utmos">Highest UTMOS</option></select></label><label>Find run<input aria-label="Find run" type="search" value={query} onChange={e=>setQuery(e.target.value)} placeholder="Name or variant…"/></label><button className="lab-primary" onClick={()=>downloadCsv(rows)}>Export {rows.length} rows ↓</button><button onClick={()=>{setCohort('48');setStudy('all');setQuality('all');setQuery('');setSort('study');}}>Reset</button></div>
    <DataComponent id="complete-results-table" queryId="runs" title="Recognition and audio quality by condition" kind="table" displayRows={rows} sourceRows={rows} variant="card"><div className="lab-table-scroll"><table className="lab-table" data-reviewed-rows><thead><tr><th>Run / condition</th><th>N</th><th>WER % ↑</th><th>DNSMOS ↑</th><th>NISQA ↑</th><th>UTMOS ↑</th><th>Quality floors</th><th>Joint goal</th></tr></thead><tbody>{rows.map(r=><tr key={r.id}><td><strong>{r.name}</strong><small>{r.study}</small><code>{r.variant}</code></td><td>{r.count}</td><td>{fmt(r.wer)}</td><td className={r.dnsmos<3?'is-low':''}>{fmt(r.dnsmos,3)}</td><td className={r.nisqa<3.75?'is-low':''}>{fmt(r.nisqa,3)}</td><td className={r.utmos<3.5?'is-low':''}>{fmt(r.utmos,3)}</td><td><span className={'lab-badge '+(r.qualityPass?'pass':'fail')}>{r.qualityPass?'Pass':'Fail'}</span>{r.failedMetrics&&<small>{r.failedMetrics}</small>}</td><td>{r.goalPass?'Met':'Not met'}</td></tr>)}</tbody></table>{!rows.length&&<p className="lab-empty">No conditions match these filters.</p>}</div></DataComponent><p className="lab-footnote">{rows.length} of {runs.length} conditions shown. WER pools S/D/I over reference words; MOS averages recordings. WER can exceed 100% because insertions count. The 24-clip and 48-clip cohorts have different denominators. Reused controls and checkpoint evaluations are retained per study and are not independent replications.</p></section>;
}

const CHANGES=[
 ['Word-risk objective','Optimize word mistakes in candidate transcripts rather than token loss.','Some candidate sets had no useful errors; the repaired version attacked but failed quality.'],
 ['Local feature mixtures','Restrict edits to nearby, similar speech features.','Quality stayed high, but attack strength almost disappeared.'],
 ['Learned waveform generator','Train a shared network to predict edits instead of optimizing each recording.','One seed and about one hour of training produced an almost-zero edit.'],
 ['Direct waveform optimization','Edit audio samples with and without spectral/content penalties.','Recognition errors increased, but stronger settings lost quality.'],
 ['Hard constraints and quiet-region budgets','Limit proxy distortion or protect quiet regions more tightly.','More quality was preserved, but the original quality-admissible tradeoff was not beaten.'],
 ['Larger models and longer runs','Add Whisper-large-v2 and Data2Vec CTC; run up to 500 ensemble or 1,500 single-model updates.','Ensembles reached roughly 90% source WER; stronger settings failed UTMOS.'],
 ['Transformations during optimization','Include filtering and resampling while calculating attack gradients.','Designed to encourage persistence; transfer and defense evaluation remain unmeasured.'],
 ['Larger BigVGAN decoder','Edit a mel spectrogram through a different 112.4M-parameter decoder.','Reconstruction was good; 200/500-step attacks damaged all three quality scores.'],
];

function Explanation(){return <div className="lab-explanation"><SectionHeader id="changes-title" title="What changed, in plain language"/><div className="lab-prose"><p>The original implementation encodes speech with WavLM, edits the features separately for each recording, and reconstructs audio with HiFi-GAN. Whisper-small tells the optimizer how to make recognition harder. The models themselves stay frozen.</p><p>The experiments changed the objective, the space we edit, the constraints, the source models, or the amount of optimization. The original implementation and completed experiment records remain preserved.</p></div><div className="lab-changes-table"><table className="lab-table"><thead><tr><th>Change</th><th>What it means</th><th>What happened</th></tr></thead><tbody>{CHANGES.map(([name,meaning,result])=><tr key={name}><td><strong>{name}</strong></td><td>{meaning}</td><td>{result}</td></tr>)}</tbody></table></div><SectionHeader id="current-outcome-title" title="Current outcome"/><div className="lab-prose"><p>The expanded cycle finished on <strong>24 October at 22:02, Taipei time</strong>, after 45 hours 51 minutes. Every scheduled stage completed. The scientific gate selected <strong>stop</strong>: no condition achieved WER above 75% while passing all three quality floors.</p><p>The original quality-admissible 50-step control remains at <strong>49.80% WER</strong> on the 48-recording cohort. Some attacks exceeded 100% WER because the recognizer inserted extra words. That does not establish that people could still understand the speech.</p></div><SectionHeader id="future-title" title="Next steps · proposed, not running"/><ol className="lab-next-steps"><li><strong>Listen across the tradeoff.</strong> Compare clean speech, the original quality control and stronger settings on the same sentences. Check whether lower UTMOS corresponds to audible damage, changed words, or a scorer mismatch.</li><li><strong>Inspect the failure mechanism.</strong> Use saved substitutions, deletions, insertions and trajectories to distinguish recognition disruption from erased or corrupted speech. Listening here is exploratory, not a blinded human score.</li><li><strong>Prioritize the quality bottleneck.</strong> Register one quality-focused change with a fixed baseline and 48-point pilot. More steps already increased quality damage; a new run needs a specific hypothesis.</li><li><strong>Advance after a joint pass.</strong> Freeze a qualifying candidate, run the 76-recording development validation, then unseen recognizers and named preprocessing comparisons. Follow with untouched confirmation/final data and a blinded human study.</li></ol><p className="lab-footnote">Transferability and defense persistence are unmeasured for these candidates. The 76-point validation shares development speakers and is not a final holdout. The prepared larger training corpus has not produced a new successful learned model. No new GPU experiment was launched to build this interface.</p></div>;}

export function DashboardContent(){
  const {reviewedRows}=useDataApp();const {activeTabId}=useDashboardTabs(TABS);
  const runs=reviewedRows('runs'),recordings=reviewedRows('recordings'),clips=reviewedRows('clips');const [chosen,setChosen]=useState(DEFAULTS);
  return <article className="audio-lab"><div className="lab-status-line"><span className="lab-status-dot"/>Expanded run complete · 45h 51m<span>{runs.length} conditions · {clips.length} sentences</span></div><div className="lab-outcome"><strong>No candidate met WER &gt; 75% and all quality floors.</strong><span>The original quality-admissible control remains the reference. Transfer and defense evaluation were not triggered.</span></div>{activeTabId==='results'?<Results runs={runs}/>:activeTabId==='explanation'?<Explanation/>:<Listening runs={runs} recordings={recordings} clips={clips} chosen={chosen} setChosen={setChosen}/>}</article>;
}
