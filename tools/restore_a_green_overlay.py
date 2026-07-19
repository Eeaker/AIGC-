"""Restore one-pixel production green strokes on top of strict LOFO geometry."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.io_utils import imread,imwrite
from src.metrics import chamfer_and_f1,closure_metrics
from src.task_a import restore_production_green

GREEN=np.array((0,255,0),np.uint8)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--strict',type=Path,required=True); ap.add_argument('--source',type=Path,required=True); ap.add_argument('--reference',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True); report={}
 mapping={'A1.jpg':'A0001.tga','A2.jpg':'A0006.tga','A3.jpg':'A0009.tga'}
 for src_name,name in mapping.items():
  base=imread(args.strict/name)[:,:,:3]; rough=imread(args.source/src_name)[:,:,:3]; ref=imread(args.reference/name)[:,:,:3]
  out=restore_production_green(base,rough); imwrite(args.output/name,out)
  rg=np.all(ref==GREEN,axis=2); pg=np.all(out==GREEN,axis=2)
  dt_r=cv2.distanceTransform((~rg).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE); dt_p=cv2.distanceTransform((~pg).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
  precision=float((dt_r[pg]<=2).mean()) if pg.any() else 0.; recall=float((dt_p[rg]<=2).mean()) if rg.any() else 0.
  report[name]={**chamfer_and_f1(out,ref),**closure_metrics(out,ref),'pixels_changed_from_strict':int(np.any(out!=base,axis=2).sum()),'green_pred':int(pg.sum()),'green_ref':int(rg.sum()),'green_precision_2px':precision,'green_recall_2px':recall,'green_f1_2px':2*precision*recall/max(precision+recall,1e-12)}
 (args.output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
