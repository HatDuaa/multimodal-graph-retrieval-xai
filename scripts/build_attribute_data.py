"""Build additive attribute graph caches for train/validation (never test)."""
import json, sys, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.graph.scene_graph import Vocab, build_scene_graph, load_raw, normalize_label
from src.graph.parse_query import parse_many
from src.utils.config import load_config, resolve

def attrs_for(raw, image_id):
    out={}
    for item in raw.get(image_id, {}).get('attributes', []):
        oid=item.get('object_id'); vals=item.get('attributes', []) or []
        if isinstance(vals,str): vals=[vals]
        clean=[]
        for val in vals:
            x=normalize_label(str(val))
            if x and x not in clean: clean.append(x)
        out.setdefault(int(oid), []).extend(clean)
    return {k:list(dict.fromkeys(v)) for k,v in out.items()}

def main():
    cfg=load_config(); root=resolve(cfg,'raw')/'visual_genome'; attrs=json.loads((root/'attributes.json').read_text(encoding='utf-8')); amap={x['image_id']:x for x in attrs}; objects, rels=load_raw(cfg); vocab=Vocab.from_config(cfg); splits=json.loads((resolve(cfg,'splits')/'vg_coco_val.json').read_text())
    all_images={s:json.loads((resolve(cfg,'splits')/f'vg_coco_{s}.json').read_text())['images'] for s in ('train','val','test')}
    graphs={}
    for split, images in all_images.items():
      for im in images:
        iid=im['image_id']; g=build_scene_graph(objects.get(iid,{'image_id':iid}), rels.get(iid,{'image_id':iid}), vocab, False, True); aa=attrs_for(amap,iid)
        for obj in g['objects']:
            vals=[a for a in aa.get(obj['source_object_id'],[]) if a != obj['name']]
            obj['attributes']=list(dict.fromkeys(vals))
            obj.pop('source_object_id',None)
        g['split']=split; graphs[iid]=g
    out=resolve(cfg,'graphs')/'vg_coco_scene_graphs_attr.jsonl'; out.write_text(''.join(json.dumps(graphs[i],ensure_ascii=False)+'\n' for s in ('train','val','test') for i in [im['image_id'] for im in all_images[s]]),encoding='utf-8')
    # Verify structure against existing graphs for train/val.
    old={json.loads(x)['image_id']:json.loads(x) for x in (resolve(cfg,'graphs')/'vg_coco_scene_graphs.jsonl').read_text().splitlines()}
    for iid,g in graphs.items():
      assert [{k:o[k] for k in ('id','name')} for o in g['objects']] == old[iid]['objects']
      assert g['relations']==old[iid]['relations']
    for split in ('train','val'):
      images=all_images[split]
      caps=[c for im in images for c in im['captions']]; parsed=parse_many([c['text'] for c in caps],16,True); payload={c['caption_id']:p for c,p in zip(caps,parsed)}
      oldq=json.loads((resolve(cfg,'processed')/f'query_graphs_{split}.json').read_text())
      for k,v in payload.items(): assert [{x:y for x,y in o.items() if x!='attributes'} for o in v['objects']]==oldq[k]['objects']; assert v['relations']==oldq[k]['relations']
      (resolve(cfg,'processed')/f'query_graphs_{split}_attr.json').write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'graphs':len(graphs),'query_train':sum(len(im['captions']) for im in all_images['train']),'query_val':sum(len(im['captions']) for im in all_images['val'])}))
if __name__=='__main__': main()
