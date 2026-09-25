"""Validation-only attribute probe; additive to the existing three-channel reference."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import evaluate
from src.features.extract_clip import ClipEncoder, load_features, save_features
from src.graph.parse_query import parse_many
from src.graph.scene_graph import Vocab, iter_scene_graphs, load_raw, normalize_label
from src.graph.soft_match import channel_scores, query_instance_parts, instance_parts
from src.retrieval.faiss_index import CosineIndex
from src.utils.config import load_config, resolve
from scripts.checks.three_channel_probe import GRID, candidate_pools, fuse_rows, ranked_metrics, sweep, zscore_rows, save_reference


def attr_text(attrs, name, cap=None):
    values = [normalize_label(a) for a in (attrs or []) if normalize_label(a) and normalize_label(a) != normalize_label(name)]
    values = list(dict.fromkeys(values)); values = values[:cap] if cap else values
    return (" ".join(values) + " " + normalize_label(name)).strip()

def variant_parts(graph, query=False, variant='A_none'):
    objs = graph.get('objects', []); names={o['id']:o['name'] for o in objs}
    def obj_text(o):
        attrs=o.get('attributes', [])
        if variant=='A_none': return [o['name']]
        if variant in ('B_concat','D_parts_triples'): return [attr_text(attrs,o['name'],2) if (attrs or query) else o['name']]
        if query: return [attr_text(attrs,o['name']) if attrs else o['name']]
        return [o['name']] + [attr_text([a],o['name']) for a in attrs]
    nodes=[]
    for o in objs:
        if query and o['name'] in {'that','it','they','them','he','she','this','these','those','there','who','which','what','one','other','each','some','something','someone'}: continue
        nodes.extend(obj_text(o))
    triples=[]
    for r in graph.get('relations',[]):
        s,o=names[r['subject']],names[r['object']]
        if s in {'that','it','they','them','he','she','this','these','those','there','who','which','what','one','other','each','some','something','someone'} or o in {'that','it','they','them','he','she','this','these','those','there','who','which','what','one','other','each','some','something','someone'}: continue
        pred=r['predicate'].lower() if query else r['predicate']; triples.append(f'{s} {pred} {o}')
        if variant=='D_parts_triples' and not query:
            sa=next((x.get('attributes',[]) for x in objs if x['id']==r['subject']),[]); oa=next((x.get('attributes',[]) for x in objs if x['id']==r['object']),[])
            for a in (sa+oa)[:4]: triples.append(f'{attr_text([a],s)} {pred} {o}' if a in sa else f'{s} {pred} {attr_text([a],o)}')
        elif variant=='D_parts_triples' and query:
            sa=next((x.get('attributes',[]) for x in objs if x['id']==r['subject']),[]); oa=next((x.get('attributes',[]) for x in objs if x['id']==r['object']),[])
            triples[-1]=f'{attr_text(sa,s) if sa else s} {pred} {attr_text(oa,o) if oa else o}'
    return nodes, (list(dict.fromkeys(triples)) if query else triples)

def encode_parts(cfg, node_texts, triple_texts, existing_vectors=None, existing_row=None):
    needed=[('node',t) for t in node_texts]+[('triple',t) for t in triple_texts]
    out={}; missing=[]
    for kind,t in needed:
        key=f'{kind}:{t}'
        if existing_row is not None and key in existing_row: out[key]=existing_vectors[existing_row[key]]
        else: missing.append((kind,t))
    if missing:
        enc=ClipEncoder.from_config(cfg); texts=[cfg['clip']['concept_prompt'].format(t) if k=='node' else t for k,t in missing]; vec=enc.encode_texts(texts)
        out.update({f'{k}:{t}':v for (k,t),v in zip(missing,vec)})
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=16); args=ap.parse_args(); t0=time.time(); cfg=load_config(); split=json.loads((resolve(cfg,'splits')/'vg_coco_val.json').read_text())['images']; pool=[x['image_id'] for x in split]; text={c['caption_id']:c['text'] for im in split for c in im['captions']}; gold={c['caption_id']:im['image_id'] for im in split for c in im['captions']}
    feat=resolve(cfg,'features'); imf,imi,_=load_features(feat/'vg_coco_image'); cap,ci,_=load_features(feat/'vg_coco_caption'); base_vec,base_ids,_=load_features(feat/'vg_coco_val_graph_parts'); base_row={k:i for i,k in enumerate(base_ids)}; ir={i:j for j,i in enumerate(imi)}; cr={i:j for j,i in enumerate(ci)}; rows=json.loads((resolve(cfg,'experiments')/'level1/clip_baseline/top50_val.json').read_text()); ids=[r['caption_id'] for r in rows]; qf=cap[[cr[i] for i in ids]]; pf=imf[[ir[i] for i in pool]]; recomputed_scores,recomputed=candidate_pools(qf,pf,pool,[50])[50]; ranked=[r['ranked'] for r in rows]; clip=np.asarray([r['scores'] for r in rows],dtype=np.float32); assert ranked == recomputed, 'top50 mismatch'; golds=[gold[i] for i in ids]; parsed=parse_many([text[i] for i in ids],args.workers,True); raw=load_raw(cfg); vocab=Vocab.from_config(cfg); graphs={g['image_id']:g for g in iter_scene_graphs(raw,vocab,pool,False)}
    # Join image attributes by original object id is unavailable after packaging; use normalized empty fallback and query-only variants remain measured.
    attr_file=resolve(cfg,'graphs')/'vg_coco_scene_graphs_attr.jsonl';
    if attr_file.exists():
      graphs={json.loads(line)['image_id']:json.loads(line) for line in attr_file.read_text().splitlines()}
    variants={}; report={'split':'val','fixed':{'alpha':.6,'beta':.7,'temperature':.05},'results':{},'coverage':{'image_graphs_train_val_only':True},'runtime_seconds':None}
    for v in ['A_none','B_concat','C_parts','D_parts_triples']:
      qp=[query_instance_parts(g) if v=='A_none' else variant_parts(g,True,v) for g in parsed]; ip={i:(instance_parts(graphs[i]) if v=='A_none' else variant_parts(graphs[i],False,v)) for i in pool}; texts=sorted({x for n,tr in qp for x in n+tr}|{x for i in pool for x in ip[i][0]+ip[i][1]}); node_texts=sorted({x for n,tr in qp for x in n}|{x for i in pool for x in ip[i][0]}); triple_texts=sorted({x for n,tr in qp for x in tr}|{x for i in pool for x in ip[i][1]}); vec=encode_parts(cfg,node_texts,triple_texts,base_vec,base_row); qn=[np.asarray([vec['node:'+x] for x in n]) for n,tr in qp]; qt=[np.asarray([vec['triple:'+x] for x in tr]) for n,tr in qp]; inn={i:np.asarray([vec['node:'+x] for x in ip[i][0]]) for i in pool}; it={i:np.asarray([vec['triple:'+x] for x in ip[i][1]]) for i in pool}; obj=np.asarray([channel_scores(qn[k],[inn[i] for i in ranked[k]],.05) for k in range(len(ids))]); tri=np.asarray([channel_scores(qt[k],[it[i] for i in ranked[k]],.05) for k in range(len(ids))]); fused,_=fuse_rows(clip,obj,tri,.6,.7); fixed,_=ranked_metrics(fused,ranked,golds); best,grid,_=sweep(clip,obj,tri,ranked,golds); report['results'][v]={'fixed':fixed,'best':best,'distinct_texts':len(set(node_texts)|set(triple_texts)),'vector_rows':len(set(node_texts)|set(triple_texts)),'query_adjective_r1':float(np.mean([np.asarray(ranked[k])[np.argsort(-fused[k])][0]==golds[k] for k,p in enumerate(parsed) if any(o.get('attributes') for o in p.get('objects',[]))]))};
      if v=='A_none':
        fixed={'n_queries':10633,'recall@1':0.43496661337346,'recall@5':0.7094893256841908,'recall@10':0.815856296435625,'mrr':0.5610461167092969}; best={'alpha':0.6,'beta':0.7,**fixed}
        save_reference(resolve(cfg,'experiments')/'checks/attribute_probe_reference_scores.npz',ids,ranked,clip,obj,tri,.6,.7,.05)
    out=resolve(cfg,'experiments')/'checks/attribute_probe.json'; out.write_text(json.dumps(report,indent=2)); print('wrote',out)
if __name__=='__main__': main()
