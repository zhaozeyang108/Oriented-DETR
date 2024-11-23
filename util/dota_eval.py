import numpy as np
from DOTA_devkit import polyiou
def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
def compute_metric(gt,predict,class_names,img_ids,ovthresh=0.5,use_07_metric=True):
    aps={}
    for i,class_name in enumerate(class_names):
        npos = 0
        class_recs={}
        class_id=i+1
        for img_id in img_ids:
            R = [obj for obj in gt[img_id] if obj["category_id"] == class_id]
            npos += len(R)
            bbox=np.array([x["bbox"] for x in R])
            class_recs[img_id]={"bbox":bbox,
                               "det":np.array([False for x in R ])}
        class_det=predict[class_id]
        det_image_ids=np.array([x[0] for x in class_det])
        confidence=np.array([x[1] for x in class_det])
        BB=np.array([x[2] for x in class_det])

        sorted_ind=np.argsort(-confidence)
        sorted_scores=np.sort(-confidence)
        BB=BB[sorted_ind]
        det_image_ids=det_image_ids[sorted_ind]
        nd=len(det_image_ids)
        tp=np.zeros(nd)
        fp=np.zeros(nd)
        for d in range(nd):
            R = class_recs[det_image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            ## compute det bb with each BBGT
            if BBGT.size > 0:
                # compute overlaps
                # intersection

                # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
                # pdb.set_trace()
                BBGT_xmin = np.min(BBGT[:, 0::2], axis=1)
                BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
                BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
                BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
                bb_xmin = np.min(bb[0::2])
                bb_ymin = np.min(bb[1::2])
                bb_xmax = np.max(bb[0::2])
                bb_ymax = np.max(bb[1::2])

                ixmin = np.maximum(BBGT_xmin, bb_xmin)
                iymin = np.maximum(BBGT_ymin, bb_ymin)
                ixmax = np.minimum(BBGT_xmax, bb_xmax)
                iymax = np.minimum(BBGT_ymax, bb_ymax)
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                       (BBGT_xmax - BBGT_xmin + 1.) *
                       (BBGT_ymax - BBGT_ymin + 1.) - inters)

                overlaps = inters / uni

                BBGT_keep_mask = overlaps > 0
                BBGT_keep = BBGT[BBGT_keep_mask, :]
                BBGT_keep_index = np.where(overlaps > 0)[0]

                def calcoverlaps(BBGT_keep, bb):
                    overlaps = []
                    for index, GT in enumerate(BBGT_keep):
                        overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                        overlaps.append(overlap)
                    return overlaps

                if len(BBGT_keep) > 0:
                    overlaps = calcoverlaps(BBGT_keep, bb)

                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)
                    # pdb.set_trace()
                    jmax = BBGT_keep_index[jmax]

            if ovmax > ovthresh:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall

        print('check fp:', fp)
        print('check tp', tp)

        print('npos num:', npos)
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        rec = tp / float(npos)  # recall
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)  # 准确率
        ap = voc_ap(rec, prec, use_07_metric)
        aps[class_names[class_id-1]["name"]]=ap
        #print("ok")
    return aps